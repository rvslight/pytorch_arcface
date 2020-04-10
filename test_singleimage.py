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

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


def showtopimg(label, output_cpu, prefix='train',index = 0, test_path = None):
    # max 5 array
    print(f'label: {label[index]}')
    max_five = np.argsort(output_cpu[index], axis=-1)[-9:][::-1]
    design_five = [design_list[i] for i in max_five]

    if test_path is not None:
        label_mat = cv2.imread(test_path,
                               cv2.IMREAD_COLOR)
        print(f'shape {opt.input_shape[::-1][:2]}')
        label_mat = cv2.resize(label_mat, (400,140))
    else:
        label_item = design_list[label[index]]
        label_mat = cv2.imread("./data/Datasets/nail_search_data/" + label_item + "/" + label_item + "_0.png",
                               cv2.IMREAD_COLOR)


    zero_mat = np.zeros(label_mat.shape,dtype=np.uint8)
    mat_show_list = []

    mat_show_list.append(label_mat)
    mat_show_list.append(zero_mat)
    for item in design_five:
        path = "./data/Datasets/nail_search_data/" + item + "/" + item + "_0.png"
        mat = cv2.imread(path, cv2.IMREAD_COLOR)
        mat_show_list.append(mat)
    final_img = np.vstack(mat_show_list)
    final_img_shape = final_img.shape
    final_img = cv2.resize(final_img,tuple(int(i*0.5) for i in final_img_shape[:2][::-1]))
    cv2.imshow("labe_to_top_"+str(index)+prefix, final_img)
    cv2.waitKey(1000)

if __name__ == '__main__':

    opt = Config()
    device = torch.device("cuda")

    test_dataset = Dataset(opt.train_root, opt.test_list, phase='test', input_shape=opt.input_shape)
    testloader = data.DataLoader(test_dataset,
                                  batch_size=opt.test_batch_size,
                                  shuffle=False,
                                  num_workers=opt.num_workers)



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
    model.to(device)
    model = DataParallel(model)
    # model.load_state_dict(torch.load(opt.train_load_model_path))
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)
    print(f'test_path: {opt.test_model_path}')
    model.load_state_dict(torch.load(opt.test_model_path))

    #val loader
    for ii, data in enumerate(testloader):
        model.eval()
        data_input, label, img_path = data
        data_input = data_input.to(device)
        label = label.to(device).long()
        feature = model(data_input)
        if opt.backbone != 'resnet18_softmax':
            if opt.metric is None:
                output = metric_fc(feature)  # softmax output..
            else:
                output = metric_fc(feature, label) #softmax output..
        else:
            output = feature
        output_cpu = output.data.cpu().numpy()
        output = np.argmax(output_cpu, axis=1)
        label = label.data.cpu().numpy()

        showtopimg(label, output_cpu, prefix='test',index=0, test_path=img_path[0])
        showtopimg(label, output_cpu, prefix='test',index=1, test_path=img_path[1])
        # showtopimg(label, output_cpu, prefix='test',index=2)
        # showtopimg(label, output_cpu, prefix='test',index=3)
        cv2.waitKey()
        # print(f'max: {str(output[0])}')
        # print(output)
        # print(label)




