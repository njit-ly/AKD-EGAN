import torch
import torch.nn as nn

class MetaNet(nn.Module):
    def __init__(self):
    # def __init__(self, num_classes=3072, slice=4):
        super(MetaNet, self).__init__()
        self.meta_layer = nn.Linear(32*32*3, 1)  # 输入通道为64，输出为1

        self.init_weights()

    def forward_meta(self, features):
        x = features.view(features.size(0), -1)  # [batch_size, 32*32*3]
        x = self.meta_layer(x)      # [batch_size, 1]
        output = torch.mean(x)      # 输出一个标量，将batch中所有样本的结果平均
        return output.view(1, 1)    # 调整输出维度为 [1, 1]


    def init_weights(self):
        """
        初始化 meta_layer 的权重
        """
        # 使用 Xavier 均匀分布初始化权重
        nn.init.xavier_uniform_(self.meta_layer.weight, gain=1.0)

        # 初始化 bias 为 0
        if self.meta_layer.bias is not None:
            nn.init.constant_(self.meta_layer.bias, 0)