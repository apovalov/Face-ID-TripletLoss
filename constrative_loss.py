import numpy as np


def contrastive_loss(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the contrastive loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        x1 (np.ndarray): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (np.ndarray): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (np.ndarray): Ground truthlabels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The contrastive loss
    """
    # Compute the L2 distance between the embeddings
    distance = np.sqrt(np.sum(np.square(x1 - x2), axis=1))

    # Compute the loss for similar samples
    similar_loss = y * np.square(distance)

    # Compute the loss for dissimilar samples
    dissimilar_loss = (1 - y) * np.square(np.maximum(0, margin - distance))

    # Compute the total loss
    loss = np.mean(similar_loss + dissimilar_loss)

    return loss


import torch

def contrastive_loss(
    x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, margin: float = 5.0
) -> torch.Tensor:
    """
    Computes the contrastive loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        x1 (torch.Tensor): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (torch.Tensor): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (torch.Tensor): Ground truth labels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The contrastive loss
    """
    # Compute the L2 distance between the embeddings
    distance = torch.sqrt(torch.sum(torch.pow(x1 - x2, 2), dim=1))

    # Compute the loss for similar samples
    similar_loss = y * torch.pow(distance, 2)

    # Compute the loss for dissimilar samples
    dissimilar_loss = (1 - y) * torch.pow(torch.clamp(margin - distance, min=0), 2)

    # Compute the total loss
    loss = torch.mean(similar_loss + dissimilar_loss)

    return loss
