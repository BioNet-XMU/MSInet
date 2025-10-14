from skimage import segmentation, color
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import umap
import cv2
cmap = matplotlib.colors.ListedColormap(np.random.rand(256,3))

def superpixels(in_image, h, w , c):
    image = in_image
    #image = MinMaxScaler().fit_transform(image)
    print(image.shape)
    print(image.shape[-1])
    print(image.shape)
    image = image.reshape(h, w, c)
    print(image.shape, "after reshape")

    #segments_slic = segmentation.slic(image, n_segments=500)
    segments_slic = segmentation.felzenszwalb(image, scale=3, min_size=9, sigma=0.8)
    segments_slic_shape = segments_slic.reshape(-1)
    print(segments_slic_shape.shape, 'sp-shape')
    #np.save('sp_umap_tumort_up.npy', segments_slic_shape)
    print(f'superpixel number of segments: {len(np.unique(segments_slic))}')
    plt.imshow(segments_slic.reshape(h, w).astype(np.uint8))
    plt.show()

    return segments_slic_shape

def create_superpixel_indices(segmentation_map):
    superpixel_indices = {}
    for index, label in enumerate(segmentation_map):
        if label not in superpixel_indices:
            superpixel_indices[label] = []
        superpixel_indices[label].append(index)
    return superpixel_indices

def norm_array(image):
    normalized_array = np.zeros_like(image)
    for channel in range(image.shape[1]):
        channel_min = image[:, channel].min()
        channel_max = image[:, channel].max()
        normalized_array[:, channel] = (image[:, channel] - channel_min) / (channel_max - channel_min)
    return normalized_array

def reduce_dimensions_umap(data, n_components=3, metric='cosine', random_state=42):
    reducer = umap.UMAP(
        n_components=n_components,
        metric=metric,
        random_state=random_state)
    reduced = reducer.fit_transform(data)
    return reduced

# Superpixel refinement
def superpixel_refinement_1(seg_map, l_inds):
    if isinstance(seg_map, torch.Tensor):
        seg_map = seg_map.detach().cpu().numpy()
    #seg_map = seg_map.detach().numpy()
    for i in range(len(l_inds)):
        labels_per_sp = seg_map[l_inds[i]]
        u_labels_per_sp = np.unique(labels_per_sp)
        hist = np.zeros(len(u_labels_per_sp))
        for j in range(len(hist)):
            hist[j] = len(np.where(labels_per_sp == u_labels_per_sp[j])[0])
        seg_map[l_inds[i]] = u_labels_per_sp[np.argmax(hist)]
    target0 = torch.from_numpy(seg_map)
    return target0

def show_image(img):
    try:
        #importing Colab-specific display
        from google.colab.patches import cv2_imshow
        cv2_imshow(img)
    except ModuleNotFoundError:
        # If not in Colab, use regular OpenCV
        cv2.imshow("Segmentation Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    plt.show()

def visualize_patches(tensor, window_size):
    C, H, W = tensor.shape
    padding = window_size // 2
    tensor = tensor.unsqueeze(0)
    unfold = nn.Unfold(kernel_size=window_size, stride=1, padding=padding)
    patches = unfold(tensor)
    patches = patches.view(1, C, window_size * window_size, -1)
    patches = patches.permute(0, 3, 2, 1)
    num_patches = patches.shape[1]
    start_index = 541
    end_index = min(num_patches, 550)
    center_idx = (window_size // 2)
    plt.figure(figsize=(12, 12))
    for i in range(start_index, end_index):
        patch = patches[0, i].view(window_size, window_size, C)
        patch_normalized = (patch - patch.min()) / (patch.max() - patch.min())
        plt.subplot(4, 3, i - start_index + 1)
        if C >= 3:
            plt.imshow(patch_normalized[:, :, :3].detach().numpy())
        elif C == 1:
            plt.imshow(patch_normalized.squeeze(), cmap='gray')
        else:
            plt.imshow(patch_normalized.numpy())
        plt.gca().add_patch(plt.Rectangle((center_idx - 0.5, center_idx - 0.5), 1, 1,
                                          edgecolor='red', facecolor='none', linewidth=2))

        plt.title(f"Patch {i + 1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()



