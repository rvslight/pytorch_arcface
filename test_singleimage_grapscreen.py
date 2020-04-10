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

import tqdm
from collections import defaultdict
from PIL import Image
from torchvision import transforms as T
from mss import mss
from radam import RAdam



def get_center(vectors):
    avg = np.mean(vectors, axis=0)
    if avg.ndim == 1:
        avg = avg / np.linalg.norm(avg)
    elif avg.ndim == 2:
        assert avg.shape[1] == 512
        avg = avg / np.linalg.norm(avg, axis=1, keepdims=True)
    else:
        assert False, avg.shape
    return avg


def get_nearest_k(center, features, k, threshold):
    feature_with_dis = [(feature, np.dot(center, feature)) for feature in features]
    if len(feature_with_dis) > 10:
        distances = np.array([dis for _, dis in feature_with_dis])

    filtered = [feature for feature, dis in feature_with_dis if dis > 0.5]
    if len(filtered) < len(feature_with_dis):
        distances = np.array([feature for feature, dis in feature_with_dis if dis <= 0.5])
    if len(filtered) > k:
        return filtered
    feature_with_dis = [feature for feature, dis in sorted(feature_with_dis, key=lambda v: v[1], reverse=True)]
    return feature_with_dis[:k]


def get_image_center(features):
    if len(features) < 4:
        return get_center(features)

    for _ in range(2):
        center = get_center(features)
        features = get_nearest_k(center, features, int(len(features) * 3 / 4), 0.5)
        if len(features) < 4:
            break

    return get_center(features)

def getSortedFeatureCenterAndIdList(dataloader, totalstep):
    festures_dict = defaultdict(list)
    for i, data in tqdm.tqdm(enumerate(dataloader), total=totalstep):
        data_inputs, labels, _ = data
        data_inputs = data_inputs.to(device)
        labels = labels.to(device).long()
        # keys = [x.split('/')[4] for x in filepaths] # get folder name
        features = model(data_inputs)

        for label, feature in zip(labels.cpu().numpy(), features.cpu().numpy()):
            festures_dict[label].append(feature)

    ## get features_list_dict
    features_dict_center = {key: get_image_center(features) for key, features in festures_dict.items()}

    id_list_ori = list(sorted(festures_dict.keys()))
    features_dict_center_sorted_list = np.stack([features_dict_center[Id] for Id in id_list_ori],
                                                      axis=0)

    return features_dict_center_sorted_list, id_list_ori


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

    ## dataloader ready
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='test', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                batch_size=opt.train_batch_size,
                                shuffle=False,
                                num_workers=opt.num_workers)

    ## model inference..
    with torch.no_grad():
        model.eval()

        # train phase
        batch_size = opt.train_batch_size
        total_size = len(trainloader.dataset)
        total_step = math.ceil(total_size / batch_size)
        features_train_dict_center_sorted_list, id_train_list_ori = getSortedFeatureCenterAndIdList(trainloader,
                                                                                                    total_step)
        # get each class center embedied vector.. and get each train.. id. value.. ex.. saltus_sj_3421.

        # test phase
        while (True):
            mon = {'top': 300, 'left': 300, 'width': 200, 'height': 200}
            sct = mss()
            input_shape = (3, 128, 224) #(3, 256, 448)#(3, 96, 192) #(3, 128, 224)

            data_input = np.array(sct.grab(mon))
            data_input_grap_input = cv2.cvtColor(data_input, cv2.COLOR_RGBA2RGB)

            cv2.imshow("grap_input", data_input_grap_input)

            data_input = Image.fromarray(data_input_grap_input)
            ## simge image ##
            transforms = T.Compose([
                # T.Grayscale(),
                T.Resize(input_shape[1:]),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
            # data = data.convert('L')
            data_input = transforms(data_input).unsqueeze(0)
            #################

            data_input = data_input.to(device)
            feature = model(data_input)
            feature = list(np.squeeze(feature).detach().cpu().numpy())
            features_test_dict_center_sorted_list = [feature]
            id_test_list_ori = [0]


            # get each test item.. center.. and ids

            # calculate distance
            #### feature_ori => train.. feature.. get each.. feature's center.. label.. and feature_vectors..
            m_ori = np.matmul(features_test_dict_center_sorted_list,
                              features_train_dict_center_sorted_list.transpose())  # => get a cosine.. angles..

            for test_index, m in enumerate(m_ori):

                custom_img = True ## custoim test img.. so.. label mat.. is custom img not in test_id_list
                top_k_show_value = 10 # top 10 vale.
                max_item_show= 1

                top_indices = np.argsort(m)[::-1][:top_k_show_value] # get top 5 in column number
                top_indices = [id_train_list_ori[x] for x in top_indices] # get real top 5 labe in id_train_list

                ## showTestImg:

                if custom_img:
                    data_list_file = opt.test_list
                    with open(os.path.join(data_list_file), 'r') as fd:
                        imgs = fd.readlines()
                    custom_imgs = [os.path.join(opt.test_root, img.split()[0]) for img in imgs]

                    top_design_indice = [design_list[i] for i in top_indices]
                    label_mat = data_input_grap_input
                else:
                    # normal img
                    label_item = design_list[id_test_list_ori[test_index]]
                    label_mat = cv2.imread("./data/Datasets/nail_search_data/" + label_item + "/" + label_item + "_0.png",
                                                                  cv2.IMREAD_COLOR)

                label_mat = cv2.resize(label_mat, (400, 140))
                zero_mat = np.zeros(label_mat.shape, dtype=np.uint8)
                mat_show_list = []

                mat_show_list.append(label_mat)
                mat_show_list.append(zero_mat)
                for item in top_design_indice:
                    path = "./data/Datasets/nail_search_data/" + item + "/" + item + "_0.png"
                    mat = cv2.imread(path, cv2.IMREAD_COLOR)
                    mat_show_list.append(mat)
                final_img = np.vstack(mat_show_list)
                final_img_shape = final_img.shape
                final_img = cv2.resize(final_img, tuple(int(i * 0.6) for i in final_img_shape[:2][::-1]))
                cv2.imshow(str(test_index%max_item_show)+"_show", final_img)
                cv2.waitKey(1000)
                # if test_index%max_item_show == 0 and test_index != 0:
                #     cv2.waitKey()







