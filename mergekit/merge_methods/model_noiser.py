import torch
from typing import List
from mergekit.merge_methods.easy_define import merge_method
@merge_method(
    name="model_noiser",
    pretty_name="Model Noiser",
    reference_url="https://example.com/docs",  # optional
)
def model_noiser(
    tensors: List[torch.Tensor],  # List of tensors (in this case, just one model's parameters)
    noise_stddev: float = 0.1,  # Standard deviation of Gaussian noise
    device: str = "cpu",  # Device to perform the operation on
) -> torch.Tensor:
    # Check that there is exactly one tensor
    if len(tensors) != 1:
        raise ValueError("Only one tensor (model) is expected")
    
    model_tensor = tensors[0].to(device)
    
    # Generate Gaussian noise
    noise = torch.randn_like(model_tensor) * noise_stddev
    
    # Add noise to model parameters
    noised_model = model_tensor + noise
    
    return noised_model
