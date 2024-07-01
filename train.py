
import os
import sys

import multiprocessing  

import argparse

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim


from TRD import TRD
from TRD import TRDLoss
from bbox_tr import plot_bbox_tr as plot_bbox
# import resize_util as rs

from PairFileDataset import PairFileDataset


def my_collate_fn(batch):

    images = [item[0] for item in batch]
    temtarget = [item[1] for item in batch]

    # #################################################################################
    # tembbox = []
    # for i in range(len(images)):
    #     resize = rs.base()
    #     images[i] = resize.resizeimg(images[i], 416) #将图像缩放为416*416
    #     tembbox.append(resize.resizebbox(images[i], temtarget['bboxes'])) #将目标框缩放为416像素下的坐标
    #     temtarget['bboxes'] = tembbox
    #################################################################################

    images = torch.stack(images,0)
    targets_np = temtarget
    targets = []
    for target_np in targets_np:
        target = {key: torch.tensor(target_np[key]) for key in target_np}
        targets.append(target)
    return images, targets_np

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train_net(net,
              save_path,
              dataset_path,              
              image_ext='.jpg',
              start_epoch=0,
              epochs=50,
              batch_size=10,             
              lr=0.01,
              momentum=0.9):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print (device)
    print('device:check!')

    net.to(device)
    criterion = TRDLoss(net.bboxw_range,net.image_size)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=momentum)
    optimizer = optim.Adam(net.parameters(), lr=lr,betas=(momentum,0.999), weight_decay=1e-8)

    print('optimizer:check!')

    transform = transforms.Compose([
        # transforms.Resize([net.image_size,net.image_size]),
        transforms.ToTensor()])
    
    print('transform:check!')

    trainset = PairFileDataset(dataset_path,image_ext,transform= transform) #打开规定后缀的训练集图像，匹配同名txt文件，存入target字典返回
    
    print('PairFileDataset:done!')
    
    image, target = trainset[3] #从trainset中获取第3个样本

    print('image:check!\ntarget:check!')

    bboxes = target['bboxes'] #读取字典中的bboxes的内容作为边界框

    cids = target['labels'] #读取字典中labels的内容，作为class ID

    image = image*255   # unnormalize 非标准化，将取值为0-1的标准化图像变为真实数值
    image = image.numpy()  #将image从tensor数据类型转变为numpy类型
    image = np.transpose(image, (1, 2, 0)) #将图像格式从【通道 高度 宽度】转变为【高度 宽度 通道】
    plot_bbox(image, bboxes, labels=cids,absolute_coordinates=True) #绘制标识框,绝对坐标，原来是False，我改成了True（我的标签坐标应该都是绝对坐标）
    plt.show() #显示画框后的图片

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=1,collate_fn = my_collate_fn)   #批量加载trainset中的数据，参数：batch_size：批大小；shuffle：是否打乱顺序；num_workers：使用多少个进程读取数据；collate_fn：在手机一个batch的数据时用到的函数，用于将一批数据转换成模型可以处理的形式。

    print('trainloader in process!\n')

    # dataiter = iter(trainloader)
    # images, targets = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    # images = images.to(device)
    # pred = net.detect(images,score_thresh=0.4)
    # image = images[0,:,:,:]
    # image = image*255    # unnormalize
    # image = image.to('cpu')
    # image = image.numpy()
    # image = np.transpose(image, (1, 2, 0))
    # plot_bbox(image, pred)
    # plt.show()    

    log_batchs = 1
    # 正例置信度
    score_loss_p_log = 0.0            
    # 负例置信度
    score_loss_n_log = 0.0         
    # 范围框数值部分
    bboxv_loss_log = 0.0 
    # 范围框符号部分
    bboxs_loss_log = 0.0 
    # 类别标签
    label_loss_log = 0.0 

    for epoch in range(start_epoch,start_epoch+epochs):  # loop over the dataset multiple times 循epoch训练网络
        
        print(epoch)
        #训练网络部分
        net.train()
        for i, (images, targets) in enumerate(trainloader, 0):
            # imshow(torchvision.utils.make_grid(images))

            # get the inputs
            images = images.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(images)

            loss = criterion(outputs, targets)

            score_loss_p_log = score_loss_p_log + criterion.score_loss_p_log
            score_loss_n_log = score_loss_n_log + criterion.score_loss_n_log
            bboxv_loss_log = bboxv_loss_log + criterion.bboxv_loss_log
            bboxs_loss_log = bboxs_loss_log + criterion.bboxs_loss_log
            label_loss_log = label_loss_log + criterion.label_loss_log

            loss.backward()
            optimizer.step()

            # print log
            if i % log_batchs == (log_batchs-1):    # print every log_batchs mini-batches 打出批日志
                log_msg = '[%d, %7d] sum_loss: %.5f score_p: %.5f score_n: %.5f bboxv: %.5f bboxs: %.5f label: %.5f' %(
                    epoch+1, i + 1, 
                    loss, 
                    score_loss_p_log/log_batchs, 
                    score_loss_n_log/log_batchs,
                    bboxv_loss_log/log_batchs,
                    bboxs_loss_log/log_batchs, 
                    label_loss_log/log_batchs) #日志内容：循环批次，正样本的分类损失，负样本的分类损失，bbox回归损失（使用smooth L1损失函数），bbox大小损失（使用smooth L1损失函数），标签损失（使用softmax损失函数），以上均为每个batch的损失值均值。
                print(log_msg)
                # log_file.writeline(log_msg)
                # 正例置信度
                score_loss_p_log = 0.0            
                # 负例置信度
                score_loss_n_log = 0.0         
                # 范围框数值部分
                bboxv_loss_log = 0.0 
                # 范围框符号部分
                bboxs_loss_log = 0.0 
                # 类别标签
                label_loss_log = 0.0 
        
        if (epoch+1) % 50 == 0: #如果已经循环50次
            model_path = os.path.join(save_path,'TRD_%d.pth'%(epoch+1)) 
            torch.save(net.state_dict(), model_path) #将训练好的模型保存在指定路径下，文件名是TRD_训练轮次.pth，即每训练50轮保存一次。
    
    model_path = os.path.join(save_path,'TRD_final.pth') #设定模型的路径指向TRD_final.pth
    torch.save(net.state_dict(), model_path) #state_dict()函数返回模型的所有参数，当完成设定的epoch后将模型的所有参数存入TRD_final.pth
    print('Finished Training')

# 定义了一个命令行控制参数，可以对下列参数通过命令行进行修改
# 模型保存路径(save_path) 数据集路径(data_path) 图像大小(image_size) 图像类型(image_ext) 起始元(start_epoch) 批大小(batch_size)等
def get_args():
    parser = argparse.ArgumentParser(description='Train the TRD on splited images and TRA lablels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 模型保存路径
    parser.add_argument('-s', '--save-path', metavar='S', type=str, default=r'lan4',
                        help='Model saving path', dest='save_path')     
    # 数据集路径 
    parser.add_argument('-d', '--dataset-path', metavar='D', type=str, default=r'DataSet/imgHub/',    #default=r'D:\cvImageSamples\lan4\splited_imgs',
                        help='Dataset path', dest='dataset_path')
    # 图片大小（边长像素） 默认608
    parser.add_argument('-iz', '--image-size', metavar='IZ', type=int, default=608,
                        help='Image size', dest='image_size')
    # 图片类型 默认.jpg
    parser.add_argument('-ie', '--image-ext', metavar='IE', type=str, default='.jpg',
                        help='Image extension name', dest='image_ext')
    # 起点epoch 默认0 从头开始
    parser.add_argument('-se', '--start-epoch', metavar='SE', type=int, default=0,
                        help='Start epoch', dest='start_epoch')
    # epoch 默认12
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=16,
                        help='Number of epochs', dest='epochs')
    # batch 默认1
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=1,
                        help='Batch size', dest='batch_size')
    # 类别数量 手动改为1类
    parser.add_argument('-c', '--num-classes', metavar='C', type=int, default=1,
                        help='number of classes', dest='num_classes')
    # 原代码默认学习率0.0001
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, default=0.0001,
                        help='Learning rate', dest='lr')
    # 动量（对梯度下降法的一种优化，原理上模拟了物理学中的动量） 默认0.9
    parser.add_argument('-m', '--momentum', metavar='M', type=float, default=0.9,
                        help='Momentum', dest='momentum')
    # 原代码由此载入预训练模型参数
    # parser.add_argument('-f', '--load', type=str, default=r"TRD/resnet50-19c8e357.pth",
    #                     help='Load model from a .pth file', dest='load')
    parser.add_argument('-f', '--load', type=str, default="",
                        help='Load model from a .pth file', dest='load')
    
    return parser.parse_args()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    args = get_args()
    
    # 
    bboxw_range = [(48,320),(24,160),(12,80)]
    net = TRD(bboxw_range,args.image_size,args.num_classes)  

    # if args.load:
    #     net.load_state_dict(
    #         torch.load(args.load)
    #     )
    
    if args.load:
        # 加载预训练的resnet参数
        pretrained_dict = torch.load(args.load)
        model_dict = net.state_dict()  
        #将pretrained_dict里不属于model_dict的键剔除掉 
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}         
        # 更新现有的model_dict 
        model_dict.update(pretrained_dict)         
        # 加载我们真正需要的state_dict 
        net.load_state_dict(model_dict)

    try:
        train_net(net=net,
                  save_path=args.save_path,
                  dataset_path=args.dataset_path,
                  image_ext=args.image_ext,                  
                  start_epoch=args.start_epoch,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  momentum=args.momentum)
    except Exception:

        import traceback
        traceback.print_exc()

        torch.save(net.state_dict(), 'INTERRUPTED.pth')

