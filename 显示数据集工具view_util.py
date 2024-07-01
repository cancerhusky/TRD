import cv2
import os
import bbox_tr as tr
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

def main():
    #读取主目录
    path = r"DataSet/resizedImgSet1/"
    #读取目录中的文件
    for file_name in os.listdir(path):
        #判断文件后缀，选择目标为.jpg后缀的图像文件存入imagepath，目标为.txt后缀的文本文件存入labelpath
        if file_name.endswith('.jpg'):
            #指定图像路径
            image_path = os.path.join(path, file_name)
            print(image_path)
            #以图像名+txt后缀组合为标签文件，并指定文件路径
            label_path = os.path.join(path, file_name.split('.')[0] + '.txt')
            print(label_path)
        else:
            #同理如果先定位到txt文件，指定txt文件路径，并用文件名+jpg组合为图像文件并指定路径
            label_path = os.path.join(path, file_name)
            print(label_path)
            image_path = os.path.join(path, file_name.split('.')[0] + '.jpg')
            print(image_path)
            
        #读取图片文件
        image = cv2.imread(image_path)
        #读取标签文件
        try:
            label_file = open(label_path,'r')
        except Exception:
            print('文件不存在。')
            continue
        bboxes = []
        labels = []
        parts = []
        polyPoints = []
        #逐行读取标签文件中的数据
        for line in label_file:
            # 整理单行数据格式
            parts = line.split(' ')
            # parts = [i.strip() for i in parts]
            # parts = [i.strip('[]') for i in parts]
            # print(parts)
            if(len(parts) > 6): 
                #################################################################################
                #读取该行标签文件的标识框
                #将第1-6个值转换为float类型
                tembbox = [float(x.strip()) for x in parts[1:7]] 
                #################################################################################
                #由tr坐标转换为4点坐标
                pts4 = tr.bbox_tr_2_4pt_2(tembbox)
                #存入bboxes集合
                bboxes.append(pts4)
                #将第0个值转换为int类型并填入labels列表中
                labels.append(int(parts[0]))
            else:
                break
            conerted_bboxes = np.array(bboxes, np.float16)
            polyPoints = [np.reshape(bbox, (-1, 2)) for bbox in conerted_bboxes]
        print("bboxes contains: ")
        print(polyPoints)
        print("labels contains: ")
        print(labels)
        
        # 从标识框集合中提取标识框，此时是四点坐标表示
        # mTargetRate = 0
        # for count, line in enumerate(bboxes, start=1):
            
        #     print(line)
            # #整理四点坐标格式
            # line = np.array(line, np.int32)
            # polyPoints = line.reshape((-1, 2))
            # print(polyPoints)
            # cv2.polylines(image, [polyPoints], isClosed=True, color=(255, 0, 0), thickness=1)
        

        # for count, line in enumerate(bboxes, start=1):
            
        #     #整理四点坐标格式
        #     line = np.array(line, np.int32)
        #     polyPoints = line.reshape((-1, 2))
        # 计算每个四边形的面积
        polygons = [Polygon(bbox) for bbox in polyPoints]
        # areas = [polygon.area for polygon in polygons]
        mArea = [polygon.area/(416*416) for polygon in polygons]
        mRate = np.mean(mArea)

        # # 计算合并后的总面积以考虑重叠部分
        # merged_polygon = unary_union(polygons)
        # # 计算最终的总面积比例
        # final_area = merged_polygon.area
        # final_rate = final_area/(416*416)
        # print("目标所占图像比例：",final_rate)
        # 准备写入文本文件的内容
        output_lines = [
            f"{file_name.split('.')[0]} {mRate}\n",
        ]

        # 将内容写入txt文件
        with open("mRateOfSingleBox.txt", "a") as file:
            file.writelines(output_lines)

        print("Results written to output.txt")

        # #显示当前图像
        # cv2.namedWindow(file_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(file_name, image)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     print('Break.')
        #     break
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    #计算当前图像的标注内容比例
    # area = getPolygonArea(polyPoints)
    # targetRate = area/(416*416)
    # print("目标项占图像比例平均值："+targetRate)




if __name__ == '__main__':
    main()


        # target = {}
        # target["bboxes"] = np.array(bboxes) #将bboxes转换为数组，存入字典target
        # target["labels"] = np.array(labels) #将labels转换为数组，存入字典target
        # return image, target
