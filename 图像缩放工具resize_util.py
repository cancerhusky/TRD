import cv2
import bbox_tr
import os
import numpy as np

# def scale(image):

#     # 获取原始图像的宽度和高度
#     width = np.shape(image)[1]
#     height = np.shape(image)[0]

#     # 计算填充比例和填充后的新宽度和高度
#     resizeScale = min(416/width, 416/height)

#     return resizeScale
class base():
    def __init__(self):
        pass

    def resizeimg(self, image, target_size):

        # 获取原始图像的宽度和高度
        width = np.shape(image)[1]
        height = np.shape(image)[0]
        # 计算缩放比例
        resizeScale = min(target_size/width, target_size/height)
        # 计算缩放后图像宽度和高度
        new_width = int(resizeScale * width)
        new_height = int(resizeScale * height)

        # 创建一个填充后的图像，将原始图像绘制在其中心
        delta_w = target_size - new_width
        delta_h = target_size - new_height
        # 计算需要添加的边界宽度
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        # 为图像添加黑色边界
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # 缩放图像
        resized_image = cv2.resize(padded_image, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return resized_image

    def resizebbox(self, image, bboxes):
        
        bboxes4pt = []
        width = np.shape(image)[1]
        height = np.shape(image)[0]
        resizeScale = min(416/width, 416/height)
        # 创建一个填充后的图像，将原始图像绘制在其中心
        bboxes4pt.append(list(x * resizeScale for x in bbox_tr.bbox_tr_2_4pt_2(bboxes))) #对四点坐标的每个点应用缩放比率
        print(bboxes4pt)
        resized_bboxes = bbox_tr.bbox_4pt_2_tr(np.reshape(bboxes4pt, (8, 1))) #最后转换回tr坐标

        return resized_bboxes


# if __name__ == '__main__':

#     image = cv2.imread('TRD/example/image1.jpg') #打开图像
#     resize = base()
#     #################################################################################
#     resized_image = resize.resizeimg(image, 416) #将图像缩放为416*416
#     cv2.imshow('resized-image', resized_image)
#     #################################################################################
#     label_file = open('TRD/example/label/image1.txt','r') #读取标签文件
#     bboxes = []
#     labels = []
#     for line in label_file: #逐行读取标签文件
#         parts = line.split() #分成单个值
#         if(len(parts) > 6): 

#             #################################################################################
#             tembbox = [float(x.strip('[]')) for x in parts[1:7]] #读取该行标签文件的标识框
#             tembbox = resize.resizebbox(image, tembbox) #将标识框转为416分辨率下的标识框
#             #################################################################################

#             bboxes.append(tembbox) #将第1-6个值转换为float类型并填入bboxes列表
#             labels.append(int(parts[0])) #将第0个值转换为int类型并填入labels列表中

#             #################################################################################
#             tembbox = []
#             #################################################################################
#     print(bboxes,'\n', labels)
    
    