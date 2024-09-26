import copy
import torch
from collections import OrderedDict
from itertools import chain
from model_merging.task_vectors import MTLTaskVector

# from model_merging.tallmask_utils import construct_consensus_mask, construct_tall_mask, load_tall_mask
# from model_merging.ties_utils import ties_merging


def state_dict_to_vector(state_dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]

    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for key, value in sorted_shared_state_dict.items()])


def vector_to_state_dict(vector, state_dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    # add back the encoder and decoder embedding weights.
    if "transformer.shared.weight" in sorted_reference_dict:
        for key in remove_keys:
            sorted_reference_dict[key] = sorted_reference_dict["transformer.shared.weight"]
    return sorted_reference_dict


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False
    return True


def topk_values_mask(M, K=0.7, return_mask=False, reshape_mask=False):
    if K == 100:
        # print("Not applying mask")
        if return_mask:
            return M, torch.ones_like(M), None
        else:
            return M, torch.ones_like(M)

    if K >= 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if reshape_mask:
        final_mask = final_mask.reshape(M.shape)

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    else:
        return M * final_mask, final_mask.float().mean(dim=1)
    

def aggregate_task_vectors(task_vectors, mm_config):
    # Flattening out Checkpoints
    remove_keys = []
    flat_task_vectors = torch.vstack([state_dict_to_vector(task_vector.theta, remove_keys) for task_vector in task_vectors])

    # Aggregate Task Vectors
    merge_method = mm_config["model_merging"]["method"]
    merge_config = mm_config[merge_method]
    if merge_config["name"] == "ties":
        # TIES Merging
        merge_func = "dis-mean"
        # merged_tv = ties_merging(flat_task_vectors, reset_thresh=merge_config["k"], merge_func=merge_func)
    elif merge_config["name"] in ["sum", "zeroshot", "average"]:
        # "sum" corresponds to Task Arithmetic (TA)
        # TA, zeroshot, weight average all construct the task vector with sum, but use different scaling factors.
        flat_task_vectors, _ = topk_values_mask(flat_task_vectors, K=merge_config["k"], return_mask=False)
        merged_tv = flat_task_vectors.sum(dim=0)
    # TODO:
    # elif merge_config["name"] == "tall_mask":
    #     # construct multi-task vector
    #     if merge_config["use_ties"]:
    #         print(f"Using TIES for constructing multi-task vector")
    #         merged_tv = ties_merging(flat_task_vectors, reset_thresh=20, merge_func=f"dis-sum")
    #     else:
    #         print(f"Using Task Arithmetic for constructing multi-task vector")
    #         flat_task_vectors, _ = topk_values_mask(flat_task_vectors, K=merge_config["k"], return_mask=False)
    #         merged_tv = flat_task_vectors.sum(dim=0)
    #     # get TALL masks
    #     if merge_config["load_masks"]:
    #         # load tall masks directly from storage
    #         eval_masks = load_tall_mask(remove_keys, ptm_check, config)
    #     else:
    #         print(f"=== Constructing TALL Mask ===")
    #         # construct tall masks
    #         eval_masks = construct_tall_mask(
    #             flat_task_vectors, flat_ft, flat_ptm, merged_tv, ptm_check, remove_keys, config
    #         )
    # elif merge_config["name"] == "consensus":  # consensus merging
    #     # construct consensus mask (assuming the TALL masks have already been constructed)
    #     consensus_mask = construct_consensus_mask(ptm_check, merge_config["prun_thre_k"], config, remove_keys)
    #     # construct multi-task vector
    #     if merge_config["use_ties"]:
    #         merged_tv = ties_merging(flat_task_vectors, reset_thresh=20, merge_func="dis-sum")
    #     else:
    #         flat_task_vectors, _ = topk_values_mask(
    #             flat_task_vectors, K=merge_config["k"], return_mask=False
    #         )  # top-k mag filtering
    #         merged_tv = flat_task_vectors.sum(dim=0)
    #     # apply the consensus mask to filter multi-task vector
    #     merged_tv = merged_tv * consensus_mask
    # elif merge_config["name"] == "mag_masking":
    #     # Magnitude masking baseline
    #     print(f"=== Using Magnitude Masking ===")
    #     merged_tv = flat_task_vectors.sum(dim=0)
    #     _, _, eval_masks = topk_values_mask(flat_task_vectors, K=merge_config["k"], return_mask=True)
    #     eval_masks = [vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys) for mask in eval_masks]
    #     eval_masks = {key: value for key, value in zip(mm_config.DATASETS, eval_masks)}
    else:
        raise ValueError(f"Method {mm_config['model_merging']['name']} not defined.")

    shared_tv_state_dict = vector_to_state_dict(merged_tv, task_vectors[0].theta, remove_keys=remove_keys)
    task_specific_tv_state_dict = dict(chain(*(task_vector.tau.items() for task_vector in task_vectors)))
    mtl_task_vector = MTLTaskVector(theta=shared_tv_state_dict, tau=task_specific_tv_state_dict)
    mtl_task_vector.tasks = dict(chain(*(task_vector.tasks.items() for task_vector in task_vectors))) # Assume one ft model for each task (e.g. there can't be two seg tasks)
    print("Norm of shared task vector: ", mtl_task_vector.norm())

    # if merge_config["name"] not in ["tall_mask", "mag_masking"]:
    #     eval_masks = None

    # return task_vector, eval_masks
    return mtl_task_vector
