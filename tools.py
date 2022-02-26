import numpy as np
import torch
import torch.nn as nn


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, logits, targets):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        pos_id = (targets==1.0).float()         #   得到obj损失的系数       [32, 13*13]
        neg_id = (targets==0.0).float()         #   得到noobj损失的系数
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs)**2

        # TODO： 5.0和1.0是怎么确定的？, 倾向于学习更小的正样本损失？
        loss = 5.0*pos_loss + 1.0*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss
        else:
            return loss


def generate_txtytwth(gt_label, w, h, s):
    # s[:-1] 等价于 s[0:len(s)-1] 或 s[:len(s)-1]， 即除了最后一个元素的切片
    # gt_label的最后一个元素为class类别
    xmin, ymin, xmax, ymax = gt_label[:-1]

    # 可见label传入的xmin、ymin、xmax、ymax均归一化后的值
    # 计算边界框的中心点
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w
    box_h = (ymax - ymin) * h

    if  box_w < 1e-4 or box_h < 1e-4:
        print("Not a valid data !!!")
        return False
    
    # 计算中心点所在的网格坐标
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)

    # 计算中心点偏移量和宽高的标签
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)          # 由于log化，不需要box_w, box_h对w、h归一化处理
    th = np.log(box_h)

    # 计算边界框位置参数的损失权重, 是个损失函数系数经验公式？
    # 损失函数采用MSE,显然框小loss小，框大loss大，为了平衡这一点，给予大框更小的惩罚权重
    weight = 2.0 - (box_w / w) * (box_h / h)
    
    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=[]):
    # 必要的参数
    batch_size = len(label_lists)
    w, h = input_size, input_size
    
    # /:结果为浮点  //: “地板除”，即下取整  %： 取余数
    ws = w // stride
    hs = h // stride

    s = stride
    # 注意hs在前，ws后，即grid_y前，grid_x后， [32, 13, 13, 7]
    gt_tensor = np.zeros([batch_size, hs, ws, 1+1+4+1])     # 1+1+4+1 对应 grid_x,grid_y,tx,ty,tw,th,weight

    """
        制作训练标签
        gt_label:   长度为5的list, （已经经过归一化处理），分别为xmin, ymin, xmax, ymax, class
    """
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_class = int(gt_label[-1])
            result = generate_txtytwth(gt_label, w, h, s)
            # 归一化框的宽高<1e-4时会返回False
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result
                # TODO: 若不同目标的中心点落在同一个grid_y,grid_x时，后面的会覆盖掉前面的label，这个问题叫做“语义歧义问题”！
                # shape[2] shape[1]网格维度，加入判断，防止溢出，尽管在generate_txtytwth中能保证条件肯定满足
                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0         # 表示该网格有物体
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight

    # reshape后的维度[batch_size, hs*ws, 1+1+4+1]
    gt_tensor = gt_tensor.reshape(batch_size, -1, 1+1+4+1)

    return torch.from_numpy(gt_tensor).float()

"""
        pred_conf:  [32, 13*13, 1]
        pred_cls:   [32, 13*13, 20]
        pred_txty:  [32, 13*13, 4]

        label:      [32, 13*13, 7]      制作方法见gt_creator    1+1+4+1
"""
def loss(pred_conf, pred_cls, pred_txtytwth, label):
    # 损失函数
    conf_loss_function = MSEWithLogitsLoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')

    # 预测
    pred_conf = pred_conf[:, :, 0]                  # [32, 169]
    pred_cls = pred_cls.permute(0, 2, 1)            # nn.CrossEntropyLoss要求C在中间 [32, 13*13, 20] -> [32, 20, 13*13]
    pred_txty = pred_txtytwth[:, :, :2]             # [32, 13*13, 2]
    pred_twth = pred_txtytwth[:, :, 2:]             # [32, 13*13, 2]

    # 标签
    gt_obj = label[:, :, 0]
    gt_cls = label[:, :, 1].long()
    gt_txty = label[:, :, 2:4]
    gt_twth = label[:, :, 4:6]
    gt_box_scale_weight = label[:, :, 6]

    
    batch_size = pred_conf.size(0)
    # 置信度损失
    conf_loss = conf_loss_function(pred_conf, gt_obj)

    # 类别损失
    """
        pred_cls:   [32, 20, 13*13]
        gt_cls:     [32, 13*13]

        nn.CrossEntropyLoss要求pred_conf为(minibatch,C,d1 ,d2,...,dK)
        pytorch会将标签值gt_cls做one_hot处理,之后再进行交叉熵的公式运算,无须自己做转换

        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss
    """
    cls_loss = torch.sum(cls_loss_function(pred_cls, gt_cls)*gt_obj) / batch_size       # TODO: 只计算了正样本损失？

    # 边界框的位置损失
    txty_loss = torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight * gt_obj) / batch_size
    twth_loss = torch.sum(torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight * gt_obj) / batch_size
    bbox_loss = txty_loss + twth_loss

    # 总的损失
    total_loss = conf_loss + cls_loss + bbox_loss

    return conf_loss, cls_loss, bbox_loss, total_loss  






















































