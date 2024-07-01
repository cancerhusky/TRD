import numpy as np
from PIL import Image
from skimage import io, transform
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from os import listdir
from os.path import join

__all__ = (
    "PairFileDataset",\
    )

class PairFileDataset(Dataset):
    def __init__(self, root_dir, img_ext = '.jpg', transform=None):
        self.root_dir = root_dir
        self.img_ext = img_ext
        self.files = []
        self.transform = transform
        for i in os.listdir(root_dir): #对目录中的每个文件

            image_id,image_ext = os.path.splitext(i) #对变量赋值，id是名称，ext是扩展名
            if image_ext.lower() == '.txt': #如果扩展名转换为小写是txt
                image_path = join(root_dir,image_id+img_ext) #image_path赋值为目录/id.ext

                if os.path.exists(image_path): #如果这个txt文件存在了
                    self.files.append(image_id) #在files中加入这个txt文件
        
        print('num of files:',len(self.files))
        print('PairFileDataset-init:check!')

    
    def __len__(self):
        return len(self.files) #返回files中文件的数量

    def __getitem__(self, idx):
        image_id = self.files[idx] #读取图像名
        image_path = join(self.root_dir,image_id+self.img_ext) #读取图像路径
        label_path = join(self.root_dir,image_id+'.txt') #读取图像同名txt文件作为标签文件
        image = Image.open(image_path) #打开图像
        # image = io.imread(image_path)          
        label_file = open(label_path,'r') #读取标签文件

        bboxes = []
        labels = []
        for line in label_file: #逐行读取标签文件
            parts = line.split() #分成单个值
            if(len(parts) > 6): #
                bboxes.append([float(x.strip('[,]')) for x in parts[1:7]]) #将第1-6个值转换为float类型并填入bboxes列表
                labels.append(int(parts[0].strip('[,]'))) #将第0个值转换为int类型并填入labels列表中
        target = {}
        target["bboxes"] = np.array(bboxes) #将bboxes转换为数组，存入字典target
        target["labels"] = np.array(labels) #将labels转换为数组，存入字典target
        if self.transform: #如果transform不为none（本变量默认是none，也就是不对图像进行变换操作，如果调用函数时的参数self中的transform不为none，将会执行以下操作）
            image = self.transform(image) #对图像进行变换操作，例如缩放、裁剪、旋转等
        return image, target
    

