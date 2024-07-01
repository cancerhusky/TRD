import cv2
import bbox_tr
import os
import numpy as np
import bbox_tr as tr

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


def main():

    #读取主目录
    path = r"TRDt/DataSet/imgLabel1"
    reset_file_path = r"TRDt/DataSet/resetPt4_imgLabel1(string)"
    for file_name in os.listdir(path):
    
        label_path = os.path.join(path, file_name)
        label_file = open(label_path,'r')
        reset_name_part = file_name.split('.')
        # reset_file_path = os.path.join(path, 'reset_file')

        f_out = open(os.path.join(reset_file_path, file_name),'w')
        #逐行读取标签文件中的数据
        for line in label_file:
            # 整理单行数据格式
            parts = []
            bboxes = []
            labels = []
            parts = line.split(' ')
            parts = [i.strip("\n") for i in parts]
            parts = [i.strip('[]') for i in parts]
            parts = [i.strip(',') for i in parts]
            print("original label:")
            print(parts)
            if(len(parts) > 6): 
                #################################################################################
                #读取该行标签文件的标识框
                #将第2-7个值转换为float类型 "[1:7]"有1没7
                tembbox = [float(x.strip()) for x in parts[1:7]] 
                #################################################################################
                #由tr坐标转换为4点坐标
                pts4 = tr.bbox_tr_2_4pt_2(tembbox)
            #存入bboxes集合
            bboxes += pts4
            #将第0个值转换为int类型并填入labels列表中
            labels += bboxes
            labels.append('long')
            labels.append(int(parts[0]))
            # labels.append(8888888888888888888888888888888888888888888888888888888888)
        
            f_out.write(" ".join(map(str,labels)))
            f_out.write("\n")

        f_out.close

if __name__ == '__main__':
    main()