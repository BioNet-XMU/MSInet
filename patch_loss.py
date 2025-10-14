import torch
import torch.nn.functional as F
import torch.nn as nn


def compute_pairwise_similarities_batched_new(center_pixels, batch_size, dim=-1):
    num_patches = center_pixels.size(0)
    feature_dim = center_pixels.size(1)
    pairwise_similarities = torch.zeros((num_patches, num_patches), device=center_pixels.device)
    for i in range(0, num_patches, batch_size):
        end_i = min(i + batch_size, num_patches)
        for j in range(0, num_patches, batch_size):
            end_j = min(j + batch_size, num_patches)
            batch_center_i = center_pixels[i:end_i]
            batch_center_j = center_pixels[j:end_j]
            batch_similarities = F.cosine_similarity(
                batch_center_i.unsqueeze(1),
                batch_center_j.unsqueeze(0),
                dim=dim
            )
            pairwise_similarities[i:end_i, j:end_j] = batch_similarities

    return pairwise_similarities


def contrastive_patch_loss(tensor, window_size, weight_sim=1, threshold=0.55, batch_size=2048):
    C, H, W = tensor.shape
    padding = window_size // 2
    tensor = tensor.unsqueeze(0)
    unfold = nn.Unfold(kernel_size=window_size, stride=2, padding=padding)
    """
    Lower stride --- more overlapping patches, captures finer local details and provides richer contrastive 
    pairs but slower and memory-heavy.
    Higher stride --- fewer or non-overlapping patches, faster but may miss fine details and give 
    weaker supervision.
    """
    patches = unfold(tensor)
    num_patches = patches.shape[-1]
    print(num_patches, 'is number of patches')
    patches = patches.view(1, C, window_size * window_size, -1)
    patches = patches.permute(0, 3, 2, 1)
    center_idx = window_size // 2
    center_pixel_from_patches = patches[:, :, center_idx, :].squeeze(0)
    pairwise_similarities = compute_pairwise_similarities_batched_new(
        center_pixel_from_patches, batch_size=batch_size, dim=-1)
    above_threshold_mask = pairwise_similarities > threshold
    similar_pixels_loss = torch.mean(
        (1 - pairwise_similarities[above_threshold_mask]) * weight_sim
    )
    positive_pairs = pairwise_similarities > threshold
    negative_pairs = pairwise_similarities < threshold
    positive_loss = torch.mean((1 - pairwise_similarities[positive_pairs]) * weight_sim)
    negative_loss = torch.mean(F.relu(pairwise_similarities[negative_pairs]))
    total_loss = positive_loss + negative_loss

    return total_loss

