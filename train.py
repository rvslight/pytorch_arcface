from __future__ import print_function
import os
from data import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
from utils import Visualizer, view_model
import torch
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from test import *
import torch.optim.lr_scheduler as lr_scheduler
from common import design_list

from radam import RAdam

def save_model(model, save_path, name, iter_cnt,metric_name):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) +'_'+metric_name +'.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def showtopimg(label, output_cpu, prefix='train',index = 0):
    # max 5 array
    print(f'label_len: {len(label)}')
    print(f'label: {label[index]}')
    max_five = np.argsort(output_cpu[index], axis=-1)[-9:][::-1]
    design_five = [design_list[i] for i in max_five]

    label_item = design_list[label[index]]
    label_mat = cv2.imread("./data/Datasets/nail_search_data/" + label_item + "/" + label_item + "_0.png",
                           cv2.IMREAD_COLOR)
    label_mat = cv2.resize(label_mat, (400, 140))
    zero_mat = np.zeros(label_mat.shape,dtype=np.uint8)
    mat_show_list = []

    mat_show_list.append(label_mat)
    mat_show_list.append(zero_mat)
    for item in design_five:
        path = "./data/Datasets/nail_search_data/" + item + "/" + item + "_0.png"
        mat = cv2.imread(path, cv2.IMREAD_COLOR)
        if np.shape(mat) == ():
            print(path)
        mat = cv2.resize(mat, (400, 140))
        mat_show_list.append(mat)
    final_img = np.vstack(mat_show_list)
    final_img_shape = final_img.shape
    final_img = cv2.resize(final_img,tuple(int(i*0.5) for i in final_img_shape[:2][::-1]))
    cv2.imshow("labe_to_top_"+str(index)+prefix, final_img)
    cv2.waitKey(1000)

if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    val_dataset = Dataset(opt.train_root, opt.val_list, phase='test', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    valloader = data.DataLoader(val_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers)


    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    elif opt.loss == 'smooth_l1_loss':
        criterion = torch.nn.SmoothL1Loss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet18_softmax':
        model = resnet_face18_softmax(output_class=opt.num_classes,use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin': # embeded feature size is 512...
        metric_fc = AddMarginProduct(512, opt.num_classes, s=opt.s, m=opt.m)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    # model.load_state_dict(torch.load(opt.train_load_model_path))
    if opt.backbone != 'resnet18_softmax':
        metric_fc.to(device)
        metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True)

    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    # scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones,
                                     gamma=0.1, last_epoch=-1)
    start = time.time()
    for i in range(opt.max_epoch):

        for ii, data in enumerate(trainloader):

            model.train()
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            if opt.backbone != 'resnet18_softmax':
                if opt.metric is None:
                    output = metric_fc(feature)  # softmax output..
                else:
                    output = metric_fc(feature, label) #softmax output..
            else:
                output = feature # just passR
            loss = criterion(output, label) #softmax.. output.. cross entropy..
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()


        #val loader
        for ii, data in enumerate(valloader):
            model.eval()
            data_input, label, _ = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            if opt.backbone != 'resnet18_softmax':
                if opt.metric is None:
                    output = metric_fc(feature)  # softmax output..
                else:
                    output = metric_fc(feature, label) #softmax output..
            else:
                output = feature # just passR
            loss = criterion(output, label) #softmax.. output.. cross entropy..

            iters = i * len(valloader) + ii

            if iters % opt.print_val_freq == 0:
                output_cpu = output.data.cpu().numpy()
                output = np.argmax(output_cpu, axis=1)
                label = label.data.cpu().numpy()
                if len(label) >= 8:
                    showtopimg(label, output_cpu, index=0)
                    showtopimg(label, output_cpu, index=1)
                    showtopimg(label, output_cpu, index=2)
                    showtopimg(label, output_cpu, index=3)
                    showtopimg(label, output_cpu, index=4)
                    showtopimg(label, output_cpu, index=5)
                    showtopimg(label, output_cpu, index=6)
                    showtopimg(label, output_cpu, index=7)
                # print(f'max: {str(output[0])}')
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} val epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='val_loss')
                    visualizer.display_current_results(iters, acc, name='val_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i, opt.metric)

        scheduler.step()
        # model.eval()
        # acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        # if opt.display:
        #     visualizer.display_current_results(iters, acc, name='test_acc')

