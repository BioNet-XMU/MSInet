import torch
import torch.optim as optim
import numpy as np
import cv2
from matplotlib import pyplot as plt
from config import args
from patch_loss import contrastive_patch_loss
from utilis import superpixel_refinement_1
use_cuda = torch.cuda.is_available()

def train_model(model, data, im, labels, data_name):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    label_colours = np.random.randint(255, size=(100, 3))
    num_epochs = args.maxIter
    loss_values = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        if use_cuda:
            data_batch = data.cuda()
        else:
            data_batch = data

        optimizer.zero_grad()
        output1 = model(data_batch)[0]
        print(output1.shape, 'shape of model output')
        output = output1.permute(1, 2, 0).contiguous().view(-1, args.outClust)
        print(output.shape, 'after reshape output')
        clusify = torch.argmax(output, dim=1)
        seg_map = clusify.data.cpu().numpy()
        nLabels = len(np.unique(seg_map))

        if args.visualize:
            seg_rgb = np.array([label_colours[c % 100] for c in seg_map])
            seg_rgb = seg_rgb.reshape(im.shape).astype(np.uint8)
            np.save(f'output/MSInet_seg_{data_name}_rgb.npy', seg_rgb)
            cv2.imshow("output", seg_rgb)
            cv2.imwrite(f'output/MSInet_seg_{data_name}_rgb_{epoch}.png', seg_rgb)
            cv2.waitKey(50)
            patch_sim = contrastive_patch_loss(output1, 5)
            rf_target = superpixel_refinement_1(seg_map, labels)
            if use_cuda:
                rf_target = rf_target.cuda()
            loss = loss_fn(output, rf_target)*0.5 + patch_sim*4
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            loss_values.append(loss_value)
            print(epoch, '/', len(data), ':', nLabels, loss.item())
            if nLabels <= args.minLabels:
                print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
                break
    # Plot loss
    plt.plot(loss_values)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f'output/MSInet_seg_{data_name}_loss.png')
    plt.show()
    # Save output
    if args.visualize:
        output = model(data)[0]
        output = output.permute(1, 2, 0).contiguous().view(-1, args.outClust)
        _, seg_map = torch.max(output, 1)
        seg_map = seg_map.data.cpu().numpy()
        seg_map_out = np.array([label_colours[c % 100] for c in seg_map])
        seg_map_out = seg_map_out.reshape(im.shape).astype(np.uint8)
        cv2.imwrite(f"output/MSInet{data_name}_.png", seg_map_out)
        np.savetxt(f"output/MSInet{data_name}_", seg_map)
        np.save(f"output/MSInet{data_name}_.npy", seg_map_out)
        print("Final output saved.")
