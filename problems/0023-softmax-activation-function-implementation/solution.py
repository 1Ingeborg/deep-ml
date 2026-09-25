import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    a = torch.as_tensor(scores, dtype = float)
    a = a - a.max()
    exp = a.exp()
    return (exp / exp.sum()).tolist()