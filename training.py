import torch
import torch.optim as optim
import numpy as np
import cv2
from matplotlib import pyplot as plt
from config import args
from patch_loss import contrastive_patch_loss
from utilis import superpixel_refinement_1
from utilis import show_image
use_cuda = torch.cuda.is_available()


def train_model(model, data, im, labels, data_name):

    height = im.shape[0]
    width = im.shape[1]
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    label_colours = np.random.randint(255, size=(100, 3))
    num_epochs = args.maxIter
    loss_values = []

    loss_hpy = torch.nn.L1Loss(reduction='mean')  
    loss_hpz = torch.nn.L1Loss(reduction='mean') 
    HPy_target = torch.zeros(height - 1, width, args.inChannel)
    HPz_target = torch.zeros(width, height - 1, args.inChannel)
        
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        if use_cuda:
            data_batch = data.cuda()
        else:
            data_batch = data

        optimizer.zero_grad()
        output1 = model(data_batch)[0]
        #print(output1.shape, 'shape of model output')
        output = output1.permute(1, 2, 0).contiguous().view(-1, args.outClust)
        #print(output.shape, 'after reshape output')
        clusify = torch.argmax(output, dim=1)
        seg_map = clusify.data.cpu().numpy()
        nLabels = len(np.unique(seg_map))

        outputHP = output.reshape((height, width, args.inChannel))
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy, HPy_target)
        lhpz = loss_hpz(HPz, HPz_target)

        if args.visualize: #and epoch % 10 == 0:
            seg_rgb = np.array([label_colours[c % 100] for c in seg_map])
            seg_rgb = seg_rgb.reshape(im.shape).astype(np.uint8)
    
            # Save the output to file
            np.save(f'output/MSInet_seg_{data_name}_rgb.npy', seg_rgb)
            #cv2.imwrite(f'output/MSInet_seg_{data_name}_rgb_{epoch}.png', seg_rgb)
            #from google.colab.patches import cv2_imshow #uncomment it if running on colab
            #cv2_imshow(seg_rgb) #uncomment it if running on colab
            #cv2.imshow(seg_rgb)
            #plt.show()
            show_image(seg_rgb)
        patch_sim = contrastive_patch_loss(output1, 5)
        rf_target = superpixel_refinement_1(seg_map, labels)
        if use_cuda:
            rf_target = rf_target.cuda()
        
        loss = loss_fn(output, rf_target)*0.5 + patch_sim*2 + (lhpy+lhpz)*2
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        loss_values.append(loss_value)
        print(epoch, '/', len(data), ':', nLabels, loss.item())
        if nLabels <= args.minLabels:
            print("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break
    # Plot loss
    #plt.plot(loss_values)
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.savefig(f'output/MSInet_seg_{data_name}_loss.png')
    #plt.show()
    
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










