from typing import List

import torch

from mergekit.merge_methods.easy_define import merge_method


@merge_method(
    name="mult",
    pretty_name="Multiplicative Model Merger",
    reference_url="https://example.com/docs",  # optional
)
def multiplicative_merge(
    tensors: List[
        torch.Tensor
    ],  # List of tensors (in this case, just two models' parameters)
    scale: float = 0.1,  # Scaling factor to avoid runaway values
    device: str = "cpu",  # Device to perform the operation on
) -> torch.Tensor:
    # Check that there are exactly two tensors (models)
    if len(tensors) != 2:
        raise ValueError("Exactly two tensors (models) are expected")
      
    model_tensor1, model_tensor2 = tensors[0].to(device), tensors[1].to(device)

    # Ensure both models' tensors have the same size
    if model_tensor1.size() != model_tensor2.size():
        raise ValueError(f"Model tensors must have the same size: {model_tensor1.size()} vs {model_tensor2.size()}")

    # Element-wise multiplication of model tensors
    merged_tensor = model_tensor1 * model_tensor2 * scale  # apply scaling

    return merged_tensor
