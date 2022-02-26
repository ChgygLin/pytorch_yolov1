from cv2 import KeyPoint
import torch
import torch.nn as nn

import numpy as np

from backbone import resnet18
from utils import SPP, Conv

import tools


'''
    nn.Module是什么东西?
'''
class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5) -> None:
        super(myYOLO, self).__init__()
        self.device = device                            # 'cuda'或者'cpu'
        self.input_size = input_size                          # 输入图像尺寸，如416,宽高均为input_size
        self.num_classes = num_classes                  # 目标类别数量
        self.trainable = trainable                      # 训练时设置为True，否则为False
        self.conf_thresh = conf_thresh                  # 最后的检测框进行筛选的阈值
        self.nms_thresh = nms_thresh                      # NMS中用到的阈值
        self.stride = 32                                # 最大降采样倍数
        self.grid_cell = self.create_grid(input_size)   # 用于得到最终的bbox参数


        # >>>>>>>>>>>>>>>>>>>> backbone网络 <<<<<<<<<<<<<<<<<<<<<<
        # 训练加载预训练权重， 预测时不用加载，直接使用trainable参数
        self.backbone = resnet18(pretrained=trainable)
        c5 = 512


        # >>>>>>>>>>>>>>>>>>>> neck网络 <<<<<<<<<<<<<<<<<<<<<<
        self.neck = nn.Sequential(SPP(), Conv(c5*4, c5, k=1))



        # >>>>>>>>>>>>>>>>>>>> detection head网络 <<<<<<<<<<<<<<<<<<<<<<
        self.head = nn.Sequential(
            Conv(c5, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1)
        )


        # >>>>>>>>>>>>>>>>>>>> prediction <<<<<<<<<<<<<<<<<<<<<<
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, kernel_size=1)


    def create_grid(self, input_size):
        """
        创建此矩阵是为了优化掉for循环，避免对每一个(grid_x, grid_y)做遍历，求真实的cx,cy,w,h

        初始化网络时，线程一个[1, HxW, 2]的矩阵G，1表示batch_size
        cxcy = (txtytwth[:,:,:2] + G ) x stride
        wh = e^(txtytwth[:,:,:2])

        生成一个tensor,grid_xy，每个位置的元素是网格的坐标
        """

        '''
            [X, Y] = meshgrid(vec_x, vec_y), vec_x, evc_y均为行向量

            X: 将vec_x复制len(vec_y)-1行得到， 即总共len(vec_y)行
            Y: 先将vec_y转置，再复制len(vec_x)-1行，即总共len(vec_x)行

            [X, Y] = meshgrid(1:3, 10:14)
            X = 1   2   3
                1   2   3
                1   2   3
            Y = 10  10  10
                11  11  11
                12  12  12
                13  13  13
                14  14  14
        '''

        '''
            以x(5, 7), y(5, 7)为例

            torch.stack([x, y], dim=0)  --->    torch.Size([2, 5, 7])
            torch.stack([x, y], dim=1)  --->    torch.Size([5, 2, 7])
            torch.stack([x, y], dim=2)  --->    torch.Size([5, 7, 2])
            torch.stack([x, y], dim=-1)  --->    torch.Size([5, 7, 2])
        '''

        # 输入图像的宽和高
        w, h = input_size, input_size
        # 特征图的宽和高
        ws = w // self.stride
        hs = h // self.stride
        # 使用torch.meshgrid函数来获得矩阵G的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # 将xy两部分坐标拼在一起，得到矩阵G
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        # 最终G矩阵的维度是[1, HxW, 2]
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)

        return grid_xy

    def set_grid(self, input_size):
        # To do:
        # 用于重置grid_xy
		
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)


    def decode_boxes(self, pred):
        """
        # 输入: 网络输出的[tx, ty, tw, th]
        # 输出: bbox的[xmin, ymin, xmax, ymax]
        """

        # zeros_like: 生成和括号内变量维度维度一致的全是零的内容
        output = torch.zeros_like(pred)

        # 获取bbox的中心点坐标
        pred[:,:,:2] = ( torch.sigmoid(pred[:,:,:2]) + self.grid_cell ) * self.stride
        # 宽高
        pred[:,:,2:] = torch.exp(pred[:,:,2:])

        # 由中心点坐标和宽高获得左上角与右下角的坐标
        output[:,:,0] = pred[:,:,0] - pred[:,:,2] / 2       # 中心点x坐标 - w半宽   xmin
        output[:,:,1] = pred[:,:,1] - pred[:,:,3] / 2       # 中心点y坐标 - h半高   ymin
        output[:,:,2] = pred[:,:,0] + pred[:,:,2] / 2       # 中心点x坐标 + w半宽   xmax
        output[:,:,3] = pred[:,:,1] + pred[:,:,3] / 2       # 中心点y坐标 + h半高   ymax

        return output
    
    def nms(self, dets, scores):
        # 这是一个最基本的基于python语言的nms操作
        # 这一代码来源于Faster RCNN项目

        # 左上角、右下角坐标
        x1 = dets[:, 0] # xmin
        y1 = dets[:, 1] # ymin
        x2 = dets[:, 2] # xmax
        y2 = dets[:, 3] # ymax

        areas = (x2 - x1) * (y2 - y1)       # bbox的面积
        order = scores.argsort()[::-1]      # 按照降序对bbox的得分进行排序

        # TODO: 具体细节待后续分析
        keep = []                           # 用于保存经过筛的最终bbox结果
        while order.size > 0:
            i = order[0]                    # 得分最高的bbox
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        
        return keep


    def postprocess(self, bboxes, scores):
        # bbox_pred: (N, 4), bsize = 1
        # prob_pred: (N, num_classes), bsize = 1

        """ 后处理
        1. 过滤掉得分很低的边界框
        2. 滤掉掉针对同一目标的冗余检测，即NMS处理
        """

        # scores: [H*W, num_classes], axis=1表示此处的num_classes维度
        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # 阈值筛选，滤掉得分低的检测框
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # 对每一类去进行NMS操作
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1
        
        # 获得最终的检测结果
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds
    
    def forward(self, x, target=None):
        # 训练部分： 网络得到obj、cls和tx,ty,tw,th三个分支预测，并计算loss
        # 推理部分： 输出经过后处理得到的bbox、cls和每个bbox的预测得分。

        #  x: [32, 3, 416, 416]     --->    [32, 512, 13, 13]
        c5 = self.backbone(x)

        # neck  [32, 512, 13, 13]   --->    [32, 2048, 13, 13]
        p5 = self.neck(c5)

        # detection head    [32, 2048, 13, 13]  --->    [32, 512, 13, 13]
        p5 = self.head(p5)

        # pred              [32, 512, 13, 13]   --->    [32, 25, 13, 13]
        pred = self.pred(p5)
        # view相当于numpy中的reshape，重新定义矩阵的形状, p5.size(0)为batch size, 将C挪到最后维度方便使用数组下标访问
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C], 方便后续的loss计算和后处理
        pred = pred.view(p5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)

        # 从预测的pred中处理objectness、clas、txtytwth三部分预测

        # objectness预测： [B, H*W, 1]      [32, 13*13, 1]
        conf_pred = pred[:, :, :1]

        # class预测： [B, H*W, num_cls]     [32, 13*13, 20]
        cls_pred = pred[:, :, 1:1+self.num_classes]

        # bbox预测: [B, H*W, 4], 4个维度分别为tx, ty, tw, th    [32, 13*13, 4]
        txtytwth_pred = pred[:, :, 1+self.num_classes:]

        # 训练时，网络将返回三部分的loss
        if self.trainable:
            conf_loss, cls_loss, bbox_loss, total_loss = tools.loss(
                                                        pred_conf=conf_pred,
                                                        pred_cls=cls_pred,
                                                        pred_txtytwth=txtytwth_pred,
                                                        label=target
                                                    )

            return conf_loss, cls_loss, bbox_loss, total_loss
        
        # 预测阶段
        else:

            with torch.no_grad():
                # batch size = 1
                # 测试时，默认batch是1，因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W, 1] -> [H*W, 1]

                conf_pred = torch.sigmoid(conf_pred)[0]

                # TODO: ???此处的bboxes不是最终的像素值
                # clamp: 将每个元素加紧到区间
                # 事实上此处 / input_size后， 结果肯定在[0, 1]中
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)

                # [B, H*W, 1] -> [H*W, 1]
                # scores = class置信度 * objectness置信度
                # 此处的 dim=1 表示cls_pred的最后维度，即num_cls
                scores = (torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred)

                # 将结果放在cpu上，便于后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds






