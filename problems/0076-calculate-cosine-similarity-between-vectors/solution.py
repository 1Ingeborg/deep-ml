import torch
import torch.nn.functional as F

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    # Implement your code here
    # v1 = v1.float()
    # v2 = v2.float()
    v1 = v1.to(dtype = torch.float)
    v2 = v2.to(dtype = torch.float)
    res = F.cosine_similarity(v1, v2, -1)
    return res.item()

v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([2, 4, 6])
result = cosine_similarity(v1, v2)
print(round(result, 3))