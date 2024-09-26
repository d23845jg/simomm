import torch
from create_network import *
from typing import Union
from utils import torch_load

_Checkpoint = Union[str, torch.nn.Module]


def symmetric_difference(A, B):
    """Returns the symmetric difference between two lists."""
    return list(set(A) ^ set(B))


class _TaskVector():
    def __init__(
        self,
        pretrained_state_dict=None,
        finetuned_state_dict=None,
        theta=None,
        tau=None,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if theta is not None:
            self.theta = theta
        else:
            assert pretrained_state_dict is not None and finetuned_state_dict is not None

            # the final task vector is the difference between finetuned and pretrained vectors.
            assert (pretrained_state_dict.keys() == finetuned_state_dict.keys()), f"State dicts have different keys: {symmetric_difference(pretrained_state_dict.keys(), finetuned_state_dict.keys())}."
            
            self.theta = {}
            for key in pretrained_state_dict:
                if pretrained_state_dict[key].dtype == torch.int64:
                    continue
                if pretrained_state_dict[key].dtype == torch.uint8:
                    continue
                self.theta[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.theta:
                if key not in other.theta:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.theta[key] + other.theta[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.theta:
                new_vector[key] = -self.theta[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.theta:
                new_vector[key] = self.theta[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.theta:
                new_vector[key] = other * self.theta[key]
        return self.__class__(vector=new_vector)

    def dot(self, other):
        """Dot product of two task vectors."""
        with torch.no_grad():
            dot_product = 0.0
            for key in self.theta:
                if key not in other.theta:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.theta[key] * other.theta[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))
    
    # def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
    #     """Apply a task vector to a pretrained model."""
    #     with torch.no_grad():
    #         pretrained_model = self._load_checkpoint(pretrained_checkpoint)
    #         pretrained_state_dict = pretrained_model.state_dict()

    #         missing_keys = []
            
    #         for key, param in pretrained_state_dict.items():
    #             vector_value = self.theta.get(key)
    #             if vector_value is None:
    #                 missing_keys.append(key)
    #             else:
    #                 # In-place addition for efficiency
    #                 param.add_(scaling_coef * vector_value)

    #         if missing_keys:
    #             print(f"Warning: the following keys are present in the pretrained state dict but not in the task vector: {missing_keys}")  # noqa: E501
            
    #         pretrained_model.load_state_dict(pretrained_state_dict)        
    #     return pretrained_model



class MTLTaskVector(_TaskVector):
    def __init__(
        self,
        pretrained_checkpoint: _Checkpoint = None,
        finetuned_checkpoint: _Checkpoint = None,
        theta=None,
        tau=None,
    ):
        if theta is not None and tau is not None:
            super().__init__(None, None, theta, None)
            self.tau = tau
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_model = self._safe_load(pretrained_checkpoint)
                finetuned_model = self._safe_load(finetuned_checkpoint)
                self.tasks = finetuned_model.tasks

                # parse pretrained_checkpoint
                pretrained_state_dict = {
                    k: v for k, v in pretrained_model.state_dict().items()
                    if not any(task in k for task in pretrained_model.tasks)
                }

                # parse finetuned_checkpoint
                finetuned_state_dict = {}
                task_specific_state_dict = {}
                for k, v in finetuned_model.state_dict().items():
                    if any(task in k for task in self.tasks):
                        task_specific_state_dict[k] = v
                    else:
                        finetuned_state_dict[k] = v

                super().__init__(pretrained_state_dict, finetuned_state_dict, None)
                self.tau = task_specific_state_dict

    def _safe_load(self, checkpoint):
        if isinstance(checkpoint, str):
            return torch_load(checkpoint)
        elif isinstance(checkpoint, torch.nn.Module):
            return checkpoint
        else:
            raise ValueError(f"Invalid type for checkpoint: {type(checkpoint)}")

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pt_model = self._safe_load(pretrained_checkpoint)

            updates = {}
            pt_model_state_dict = pt_model.state_dict()
            
            missing_keys = []
            for param_name in self.theta:
                # Skip task-specific module names
                # if any(task in key for task in pt_model.tasks.keys()):
                #     continue

                param_value = self.theta.get(param_name)
                if param_value is None:
                    missing_keys.append(param_name)
                    continue
                
                # Efficient in-place addition
                # param.add_(scaling_coef * vector_value)
                updates[param_name]  = pt_model_state_dict[param_name] + scaling_coef * param_value
                # updates[param_name] = pt_model_state_dict[param_name] + 1 * param_value
            
            # Log missing keys if any
            if missing_keys:
                print(f"Warning: the following keys are present in the pretrained state dict but not in the task vector: {missing_keys}")
            
            # Load updated state dict into the model
            pt_model.load_state_dict({**pt_model.state_dict(), **updates, **self.tau})
        
        return pt_model
