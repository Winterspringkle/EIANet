import numpy as np
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import torch.nn.functional as F

class ISBI_Loader(Dataset):
    def __init__(self, data_path,transform):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'imgs/*.png'))
        self.transform = transform

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    # isbi_train_dataset n*(imgs,masks,edges)
    #     8 个 index
    # 每一个index，都执行getitem操作
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]

        # 根据image_path生成label_path
        label_path = image_path.replace('imgs', 'masks')[:-8]+"_mask.png"
        edge_path = image_path.replace('imgs', 'edges')[:-8]+"_edge.png"

        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])

        # 处理标签，将像素值为255的改为1 0~255/255 (0,1)
        if label.max() > 1:
            label = label / 255
        if edge.max() > 1:
            edge = edge / 255
        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
            edge = self.augment(edge, flipCode)
    
        return image,label, edge

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)



class ISBI_Loadertest(Dataset):
    def __init__(self, data_path,transform):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'imgs/*.png'))
        self.transform = transform
     
    
    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('imgs', 'masks')[:-8] + "_mask.png"
        edge_path = image_path.replace('imgs', 'edges')[:-8] + "_edge.png"
        # 读取训练图片和标签图片
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)
        edge = cv2.imread(edge_path, 0)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])


        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        if edge.max() > 1:
            edge = edge / 255
        return image,label, edge

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


