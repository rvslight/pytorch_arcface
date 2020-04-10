import cv2
import os
import numpy as np
# import tqdm
from itertools import permutations, combinations
from albumentations import (OneOf, Compose,Flip, ShiftScaleRotate, GridDistortion, ElasticTransform,RandomGamma, RandomContrast, RandomBrightness, RandomBrightnessContrast,Blur, MedianBlur, MotionBlur,CLAHE, IAASharpen, GaussNoise,HueSaturationValue, RGBShift, IAAAdditiveGaussianNoise)
import random
import shutil
import tqdm

max_num_in_one_class = 30
nail_search_root = "./data/Datasets/nail_search_data/"
SHOW_NAIL = False

def strong_aug(p=1.0):
    return Compose(
        [
        RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.0, p=1.0, brightness_by_max=True),
        ShiftScaleRotate(p=1.0, shift_limit=0.2, scale_limit=0.5, rotate_limit=180, border_mode=cv2.BORDER_REPLICATE),
        GridDistortion(p=0.5),
        ElasticTransform(p=0.5),
         ], p=p)

def transform(image):
    augmentation = strong_aug()
    data = {"image": image}
    augmented = augmentation(**data)
    image = augmented["image"]

    # image[np.where(image == 0)] = 255
    return image


def getHStackCropedImg(rgb_img, mask_img, resize=(80,140), bg_white=True, random_rotate=False):
    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # print(f'len: {len(contours)}')

    rgb_img_masked_img = cv2.copyTo(rgb_img, mask_img)

    if bg_white:
        ## black to white bg
        inv_mask_img = ~mask_img
        rgb_img_masked_img[np.where(inv_mask_img==255)]=255
        ## end of black bg white
    resized_rgb_img_array = []

    for contour in contours:
        assert len(contours) == 5 #check nail is 5

        ext_left = tuple(contour[contour[:, :, 0].argmin()][0])
        ext_right = tuple(contour[contour[:, :, 0].argmax()][0])
        ext_top = tuple(contour[contour[:, :, 1].argmin()][0])
        ext_bot = tuple(contour[contour[:, :, 1].argmax()][0])

        cropped_image = rgb_img_masked_img[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
        if random_rotate:
            padding = random.randint(20,70)
            cropped_image = cv2.copyMakeBorder(cropped_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                               value=(255, 255, 255))
            cropped_image = transform(cropped_image)
        else:
            padding = 20
            cropped_image = cv2.copyMakeBorder(cropped_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                               value=(255, 255, 255))
        resized_cropped_img = cv2.resize(cropped_image, resize, interpolation=cv2.INTER_CUBIC)

        # cv2.imshow("resized_cropped_img", resized_cropped_img)
        # cv2.waitKey()
        resized_rgb_img_array.append(resized_cropped_img)
    return resized_rgb_img_array


def getRidOfNoiseInMask(mask_img, noise_threshold = 400):
    mask_img = cv2.threshold(mask_img, 1, 255, 0)[1]

    contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    ## filled up mask
    for contour in contours:
        if cv2.contourArea(contour) < noise_threshold:
            cv2.drawContours(mask_img, [contour], 0, 0, -1)
            continue
        cv2.drawContours(mask_img, [contour], 0, 255, -1)

    return mask_img

def getPermutationOrderArray(max_count = 10):
    permut_list = list(permutations(range(0, 5)))
    print(f'size permute list : {len(permut_list)}')
    permut_list = permut_list[:max_count]

    return permut_list

if __name__ == '__main__':

    permut_order_list = getPermutationOrderArray(max_num_in_one_class)
    if os.path.exists(nail_search_root):
        shutil.rmtree(nail_search_root)
    os.mkdir(nail_search_root)

    root = "./nail_designs"
    path_list = os.listdir(root)
    for i,item in tqdm.tqdm(enumerate(path_list),total=len(path_list)):
        if '.DS_Store' in item:
            continue
        # item = 'zinipin_sj_4159.png'

        path = os.path.join(root , item)
        print(path)
        folder_path = item.split(".")[0]
        rgba_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        rgb_img = rgba_img[:, :, 0:3]

        mask_img = rgba_img[:, :, 3]

        mask_img = getRidOfNoiseInMask(mask_img)

        one_nail_resize_size = (80,140) # W, H

        for index, order_list in enumerate(permut_order_list):
            if index == 0:
                resize_rgb_img_array = getHStackCropedImg(rgb_img, mask_img, resize=one_nail_resize_size,
                                                          random_rotate=False)
            else:
                resize_rgb_img_array = getHStackCropedImg(rgb_img, mask_img, resize=one_nail_resize_size,
                                                          random_rotate=True)

            final_img = [resize_rgb_img_array[i] for i in order_list]

            final_img = np.hstack(final_img)
            # cv2.imshow("final_img", final_img)
            # print(final_img.shape)

            #black bg to white
            # mask = cv2.cvtColor(final_img,cv2.COLOR_BGR2GRAY)
            # mask = cv2.threshold(final_img,1,255,cv2.THRESH_BINARY_INV)[1]#[final_img==(0,0,0)]=(255,255,255)
            # final_img[np.where(final_img<10)]=255

            saved_file_name = folder_path+"_"+str(index)+".png"
            folder_root = nail_search_root+folder_path
            if SHOW_NAIL:
                cv2.imshow("show", final_img)
                cv2.waitKey()
            if not os.path.exists(folder_root):
                os.mkdir(folder_root)
            cv2.imwrite(folder_root+"/"+saved_file_name,final_img)

        # cv2.waitKey()
        assert final_img.shape[1] == one_nail_resize_size[0] * 5 # one nail'width is 80
        cv2.waitKey(10)