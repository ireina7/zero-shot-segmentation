import argparse
import torch
import torch.nn as nn
import numpy as np # type: ignore
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import matplotlib.pyplot as pyplot # type: ignore
from config import *
from utils import *

from dataset.dataset_voc import dataloader_voc
from model.vgg_voc import Our_Model


from typing import List, Set, Dict, Tuple, Optional, Union, Callable, Iterator




'''
Real work start!
@Auther: Ziqiang Y
'''
def main():
    """ Main zero shot segmentation function """
    args = get_arguments()
    device = args.device
    print_config(args)

    w, h = args.input_size.split(",")
    input_size = (int(w), int(h))
    if args.restore_from_where == "pretrained":
        model = Our_Model(split)
        i_iter = 0
    else:
        restore_from = get_model_path(args.snapshot_dir)
        model_restore_from = restore_from["model"]
        i_iter = restore_from["step"]

        model = Our_Model(split)
        saved_state_dict = torch.load(model_restore_from)
        model.load_state_dict(saved_state_dict)

    model.train()
    model.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_loader = dataloader_voc(split=split)
    data_len = len(train_loader)
    num_steps = data_len * args.num_epochs

    '''
    optimizer = optim.SGD(
        model.optim_parameters_1x(args),
        lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    '''
    optimizer = optim.Adam(
        model.optim_parameters_1x(args),
        lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    '''
    optimizer_10x = optim.SGD(
        model.optim_parameters_10x(args),
        lr=10 * args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    '''
    optimizer_10x = optim.Adam(
        model.optim_parameters_10x(args),
        lr=10 * args.learning_rate, weight_decay=args.weight_decay)
    optimizer_10x.zero_grad()

    seg_loss = nn.CrossEntropyLoss(ignore_index=255)
    #seg_loss = FocalLoss() # merely test if focal loss is useful...

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode="bilinear", align_corners=True)

    with open(RESULT_DIR, "a") as f:
        f.write(SNAPSHOT_PATH.split("/")[-1] + "\n")
        f.write("lambda : " + str(lambdaa) + "\n")


    for epoch in range(args.num_epochs):
        print("\n>> Epoch: ", epoch)
        train_iter = enumerate(train_loader)
        model.train()
        hist = np.zeros((15, 15))
        for i in range(data_len):
            print("\n> Epoch {}, loop {}".format(epoch, i))
            loss_pixel = 0
            loss_pixel_value = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, num_steps, args, times=1)

            optimizer_10x.zero_grad()
            adjust_learning_rate(optimizer_10x, i_iter, num_steps, args, times=10)

            # train strong
            try:
                _, batch = train_iter.__next__()
            except StopIteration:
                train_strong_iter = enumerate(train_loader)
                _, batch = train_iter.__next__()

            images, masks = batch["image"], batch["label"]
            #print("mask: ", masks[0].min())
            images = images.to(device)
            masks = masks.long().to(device)
            pred = model(images, "all")
            pred = interp(pred)


            # Calculate mIoU
            print(pred.shape, masks.shape)
            #pred_IoU = pred[0].permute(1, 2, 0)
            #pred_0 = pred[0].clone().cpu()
            #pred_IoU = torch.max(pred_0, 0)[0].byte()

            max_ = torch.argmax(pred, 1)
            pred_IoU = max_[0].clone().detach().cpu().numpy()

            #print(pred_IoU.shape)
            #pred_cpu = pred_IoU.data.cpu().numpy()
            pred_cpu = pred_IoU
            mask_cpu = masks[0].cpu().numpy()

            pred_cpu[mask_cpu == 255] = 255
            m = confusion_matrix(mask_cpu.flatten(), pred_cpu.flatten(), 15)
            #hist += m
            hist += m
            mIoUs = per_class_iu(hist)
            print("> mIoU: {}".format(per_class_iu(m)))
            print("> mIoUs: {}".format(mIoUs))
            print("> Average mIoUs: {}".format(sum(mIoUs) / len(mIoUs)))

            '''
            m = confusion_matrix(np.array([1, 1, 1, 255]), np.array([1, 2, 0, 0]), 3)
            mIoUs = per_class_iu(m)
            print("test> mIoU: {}".format(mIoUs))
            '''

            #vis = to_color_img(pred.clone().detach())
            #print(vis.shape)
            #pyplot.imshow(vis)
            #pyplot.imshow(masks[0])
            #pyplot.show()
            loss_pixel = seg_loss(pred, masks)
            loss = loss_pixel# + loss_qfsl

            max_ = torch.argmax(pred, 1)
            #print(max_[0])
            if i % 10 == 0:
                ans = max_[0].clone().detach().cpu().numpy()
                #x = np.where(ans == 0, 255, ans)
                x = ans
                y = masks.cpu()[0]
                x[y == 255] = 255
                show_sample(batch)
                #x = ans
                pyplot.imshow(x, cmap = 'tab20', vmin = 0, vmax = 21)
                plt.colorbar()
                pyplot.show()
                '''
                pyplot.imshow(masks[0].cpu())
                pyplot.show()
                '''

            print("{} {}, loss: {}".format(max_[0, 200, 200].data, masks[0, 200, 200].data, loss))
            loss.backward()
            optimizer.step()
            optimizer_10x.step()


    '''
    print("ok")

    for i in range(10):
        sample = next(train_iter)
        p = sample[1]['label'][0]
        print(p[p != 255])
        pyplot.imshow(sample[1]['label'][0])
        pyplot.show()
    '''

    #end main



if __name__ == "__main__":
    main()
