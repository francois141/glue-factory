from pathlib import Path

import torch

from gluefactory.geometry.homography import warp_points_torch


def compute_matches(
    keypoints_imA: torch.Tensor,
    keypoints_imB: torch.Tensor,
    H: torch.Tensor,
    matching_threshold: float = 3.0,
    best_match_only: bool = False,
) -> torch.Tensor:
    """
    Projects keypoints from image B to image A using a homography matrix H.
    Returns a list of matches for kp projected from B to A to all Points in A.
    if best match only - find best matching keypoint in img A for projected kp from B to A
    Args:
        keypoints_imA: keypoints in image A (B, N_A, 2)
        keypoints_imB: keypoints in image B (B, N_B, 2)
        H: homography matrix from image A to image B (B, 3, 3)
        matching_threshold: the threshold for a keypoint to be considered a match for another
    Returns:
        Tensor: (M, 3) where M is the number of matches (0 or 1 for best match). Each row (batch_idx, keypoint_idx_imA, keypoint_idx_imB)
    """
    # Warp detected Kp from Image B to Image A
    warped_points = warp_points_torch(keypoints_imB, H, inverse=True)  # (B, N_B, 2)
    # Compute distance for each warped point p_BA to all points P_A
    # keypoints_imA[:, :, None, :] -> (B, N_A, 1, 2)
    # warped_points[:, None, :, :] -> (B, 1, N_B, 2)
    # Broadcasting results in -> (B, N_A, N_B, 2)
    # After norm computation -> (B, N_A, N_B)
    dists = torch.linalg.norm(
        keypoints_imA[:, :, None, :] - warped_points[:, None, :, :], axis=-1
    )  # (B, N_A, N_BA)

    if best_match_only:
        # For each keypoint in B, find the closest keypoint in A
        min_dists, best_matches_A = torch.min(
            dists, dim=1
        )  # both (B, N_B) idx + value of best match
        valid_matches = min_dists < matching_threshold  # (B, N_B)

        # Get batch indices, and kp indices, of kp in B and corresponding kp indices in imA indices
        batch_idx, idx_B = torch.where(
            valid_matches
        )  # both: (M,) where M = number of valid matches (here its only 0 or 1)
        idx_A = best_matches_A[batch_idx, idx_B]  # (M,)

        matches = torch.stack([batch_idx, idx_A, idx_B], dim=1)
    else:
        batch_idx, idx_A, idx_B = torch.where(
            dists < matching_threshold
        )  # all: (M,), M=num-matches
        matches = torch.stack([batch_idx, idx_A, idx_B], dim=1)  # (M, 3)

    return matches  # (M, 3)


def sparse_nre_loss(
    descriptors1: torch.Tensor,
    descriptors2: torch.Tensor,
    matches: torch.Tensor,
    temperature: float = 0.1,
):
    """
    Compute the Sparse Neural Reprojection Error (NRE) loss for batched input.
    For each keypoint in A (descr 1), get softmax out of best matching keypoint projected from B (descr 2) to A.
    For this to work, the number of keypoints and tus matches must be the same for all samples in the batch.

    Args:
        descriptors1 (torch.Tensor): Descriptors from image 1 (B, N1, D).
        descriptors2 (torch.Tensor): Descriptors from image 2 (B, N2, D).
        matches (torch.Tensor): shape (M, 3) where each row contains
                               (batch_idx, idx_kp_desc1, idx_kp_desc2).
        temperature (float): Temperature scaling factor for the softmax.

    Returns:
        torch.Tensor: Computed Sparse NRE loss (scalar), here we return scalar loss as per sample calculation would be
                      way less efficient (need to build loss vector for later after return)
    """
    # if there is no match return 0
    if matches.shape[0] == 0:
        return torch.tensor(0.0, device=descriptors1.device, requires_grad=True)

    # Extract batch indices and keypoint indices
    batch_idx = matches[:, 0]  # (M,)
    kp_idx1 = matches[:, 1]  # (M,)
    kp_idx2 = matches[:, 2]  # (M,)

    # Batch wise similarity calculation (need softmax prob over all descr of img 2 for kp in img 1)
    # Extract matched descriptors from image 1 -> For each kp in Im1 for that a match exists, get descriptor
    desc1_matched = descriptors1[batch_idx, kp_idx1]  # (M, D)
    M = matches.shape[0]

    # For each match, we need descriptors for whole image of respective sample -> (M, N2, D)
    desc2_expanded = descriptors2[batch_idx]  # (M, N2, D)

    # Expand desc1_matched to match: (M, D) -> (M, 1, D)
    desc1_expanded = desc1_matched.unsqueeze(1)  # (M, 1, D)

    # Compute similarities: (M, 1, D) @ (M, D, N2) -> (M, 1, N2)
    similarities = torch.bmm(
        desc1_expanded, desc2_expanded.transpose(-2, -1)
    )  # (M, 1, N2)
    similarities = similarities.squeeze(1)  # (M, N2)

    # Subtract 1 and apply temperature scaling, then apply softmax along N2 dimension
    softmax_probs = torch.softmax((similarities - 1) / temperature, dim=1)  # (M, N2)

    # Get probabilities for correct matches using advanced indexing
    correct_match_probs = softmax_probs[torch.arange(M), kp_idx2]  # (M,)

    # Compute negative log likelihood
    loss = -torch.log(correct_match_probs + 1e-8).mean()
    return loss
