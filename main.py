import time
import numpy as np
import torch
from torch.autograd import Variable
from model import MyNet
from utilis import superpixels, create_superpixel_indices, norm_array
from utilis import  reduce_dimensions_umap
from utilis import superpixel_refinement_1
from config import args
from training import train_model

use_cuda = torch.cuda.is_available()

def main():
    # === Step 1: Path to the data
    data_path = input("Enter path to your .npy data file: ").strip()
    load_image = np.load(data_path)
    print(f"Loaded data shape: {load_image.shape}")

    # === Step 2: Apply UMAP
    n_features = load_image.shape[1]
    print(f"Number of features: {n_features}")
    if n_features > 3:
        print("Data has more than 3 features — applying UMAP to reduce ...")
        load_image = reduce_dimensions_umap(load_image)
        print(f"UMAP reduced shape: {load_image.shape}")
    else:
        print("Data already has 3 features, skip UMAP.")
    
    # === Step 3: insert data shape
    shape_input = input("Enter image shape (height width channels): ").strip()
    shape = tuple(int(x) for x in shape_input.split())
    if len(shape) != 3:
        raise ValueError("Please enter exactly three integers for height, width, and channels.")

    data_name = data_path.split('/')[-1].replace('.npy', '')
    load_image = norm_array(load_image)
    im = load_image.reshape(shape)

    print(f"Loaded data from: {data_path}")
    print(f"Reshaped to: {im.shape}")
    print(f"Value range: [{im.min():.3f}, {im.max():.3f}]")
    
    data = torch.from_numpy(np.array([im.transpose(2, 0, 1).astype('float32') / 255.]))
    if use_cuda:
        data = data.cuda()
    data = Variable(data)
    
    # ===Step 4: Generate superpixels
    sp_map = superpixels(load_image, *im.shape)
    sp_indices_map = create_superpixel_indices(sp_map)
    labels = [np.array(indices) for indices in sp_indices_map.values()]

    # === Model and training
    model = MyNet(data.size(1))
    if use_cuda:
        model.cuda()
    model.train()

    train_model(model, data, im, labels, data_name)


if __name__ == "__main__":
    main()






