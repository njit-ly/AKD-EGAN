import torch
import numpy as np
import torch.nn.functional as F
from utils.nsga import NSGA_2

from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

# Prioritized Path Board
class PrioritizedBoard():
    '''
    prioritized_board: 优先级板，存储候选架构及其相关信息。
    choice_num: 操作数的选择范围，即每个位置上有多少个可能的操作（默认为6）。
    is_gap: 精度差距，用于决定何时更新优先级板。
    fid_gap: 精度差距，用于决定何时更新优先级板。
    '''
    def __init__(self, cfg, CHOICE_NUM=10, is_gap=0.1, fid_gap=2):
        self.cfg = cfg
        self.prioritized_board = []
        self.choice_num = CHOICE_NUM
        self.is_gap = is_gap
        self.fid_gap = fid_gap

    # select teacher from prioritized board
    def select_teacher(self, gen_net, MetaN, random_cand):
        meta_value, cand_idx, teacher_cand = -1000000000, -1, None
        for now_idx, item in enumerate(self.prioritized_board):
            inputx = item[4]    # 索引4存储的训练数据
            output, _ = gen_net(inputx, random_cand)
            output = F.softmax(output, dim=1)   # 学生架构的输出
            # 计算当前候选架构的权重
            # logger.info(f'feature.shape: {(output - item[5]).shape}')
            weight = MetaN.module.forward_meta(output - item[5])    # 索引5存储的特征
            # logger.info(f'weight.shape: {weight.shape}')
            if weight > meta_value:
                meta_value = weight
                cand_idx = now_idx
                teacher_cand = self.prioritized_board[cand_idx][3]
        assert teacher_cand is not None
        meta_value = torch.sigmoid(-weight)
        # 返回匹配度和教师模型
        return meta_value, teacher_cand


    def board_size(self):
        return len(self.prioritized_board)

    # 判断当前架构是否应该更新优先级板
    def isUpdate(self, current_epoch, IS, FID):
        if current_epoch <= self.cfg.PRIORITIZED_BOARD_STA_EPOCH:
            return False

        if len(self.prioritized_board) < self.cfg.PRIORITIZED_BOARD_POOL_SIZE:
            return True

        if IS > self.prioritized_board[-1][1] + self.is_gap:
            return True

        if FID < self.prioritized_board[-1][2] - self.fid_gap:
            return True

        if IS > self.prioritized_board[-1][1] and FID < self.prioritized_board[-1][2]:
            return True

        return False

    # 根据当前架构的精度、FLOPs 等信息，判断是否更新优先级板。
    # 如果满足更新条件，将当前架构及其相关信息加入优先级板，并根据精度进行排序。
    # 如果板上架构数量超出池大小，移除最差架构。
    def update_prioritized_board(self, inputs, teacher_output, outputs, current_epoch, IS, FID, cand):
        if self.isUpdate(current_epoch, IS, FID):
            is_value = IS  # 将 IS 作为第一个目标
            fid_value = FID  # 将 FID 作为第二个目标

            training_data = deepcopy(inputs.detach())

            if len(self.prioritized_board) == 0:
                features = deepcopy(outputs.detach())
            else:
                features = deepcopy(
                    teacher_output.detach())

            self.prioritized_board.append(
                (is_value,
                 is_value,
                 fid_value,
                 cand,
                 training_data,
                 F.softmax(
                     features,
                     dim=1)))

            # 从 prioritized_board 中提取目标 IS 和 FID 的值，用于 NSGA-II 排序
            IS_list = [item[1] for item in self.prioritized_board]  # 第一个目标：IS
            FID_list = [-item[2] for item in self.prioritized_board]  # 第二个目标：FID

            # 调用 NSGA-II 排序算法
            sorted_indices = NSGA_2(IS_list, FID_list, self.cfg.PRIORITIZED_BOARD_POOL_SIZE)

            # 根据排序结果重新排列 prioritized_board
            self.prioritized_board = [self.prioritized_board[i] for i in sorted_indices]

        if len(self.prioritized_board) > self.cfg.PRIORITIZED_BOARD_POOL_SIZE:
            del self.prioritized_board[-1]