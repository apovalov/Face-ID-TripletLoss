import numpy as np


def triplet_loss(
    anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the triplet loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        anchor (np.ndarray): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (np.ndarray): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (np.ndarray): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The triplet loss
    """
    # Compute the L2 distance between the anchor and positive samples
    positive_distance = np.sqrt(np.sum(np.square(anchor - positive), axis=1))

    # Compute the L2 distance between the anchor and negative samples
    negative_distance = np.sqrt(np.sum(np.square(anchor - negative), axis=1))

    # Compute the loss for similar samples
    loss = np.maximum(0, positive_distance - negative_distance + margin).mean()

    return loss


import torch


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 5.0,
) -> torch.Tensor:
    """
    Computes the triplet loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        anchor (torch.Tensor): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (torch.Tensor): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (torch.Tensor): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The triplet loss
    """
    # Compute the L2 distance between the anchor and positive samples
    positive_distance = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), dim=1))

    # Compute the L2 distance between the anchor and negative samples
    negative_distance = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), dim=1))

    # Compute the loss for similar samples
    loss = torch.clamp(positive_distance - negative_distance + margin, min=0)

    loss = torch.mean(loss)

    return loss
