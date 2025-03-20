from __future__ import absolute_import, division, print_function    # 兼容 Python 2 和 Python 3
import search_cfg
import archs
import datasets
from trainer.trainer_generator import GenTrainer
from trainer.trainer_utils import LinearLrDecay
from utils.utils import set_log_dir, save_checkpoint, create_logger, count_parameters_in_MB
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from utils.flop_benchmark import print_FLOPs
from archs.super_network import Generator, Discriminator
from archs.fully_super_network import simple_Discriminator
from algorithms.search_algs import GanAlgorithm
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm   # 进度条
# import copy
from copy import deepcopy
from pytorch_gan_metrics import get_inception_score_and_fid

torch.backends.cudnn.enabled = True     # 启用 CUDA 随机数生成器的 cuDNN 后端，用于 GPU 加速。
torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的基准模式。


def main():
    args = search_cfg.parse_args()      # 解析命令行参数
    torch.cuda.manual_seed(args.random_seed)    # 为当前GPU设置随机种子
    # set visible GPU ids
    if len(args.gpu_ids) > 0:           # 如果指定了GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids   # 根据指定的GPU ID设置可见GPU
    # the first GPU in visible GPUs is dedicated for evaluation (running Inception model)
    str_ids = args.gpu_ids.split(',')   # 以逗号为分隔符，将字符串分割为子字符串
    args.gpu_ids = []                   # 初始化一个空列表来存储GPU ID
    for id in range(len(str_ids)):
        if id >= 0:                     # 检查ID是否为非负数
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 1:           # 如果指定了多个GPU
        args.gpu_ids = args.gpu_ids[1:] # 只使用第二个及以后的GPU
    else:
        args.gpu_ids = args.gpu_ids     # 否则只使用单个GPU

    # genotype G
    gan_alg = GanAlgorithm(args)        # 初始化GAN算法

    # import network from genotype
    basemodel_gen = Generator(args)     # 使用指定的参数创建 Generator 类的实例，这里为超网
    gen_net = torch.nn.DataParallel(
        basemodel_gen, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])   # 将生成器网络放入 DataParallel 中并移动到 GPU 上。
    basemodel_dis = simple_Discriminator()  # 使用指定的参数创建 Discriminator 类的实例，这里为超网
    dis_net = torch.nn.DataParallel(
        basemodel_dis, device_ids=args.gpu_ids).cuda(args.gpu_ids[0])   # 将判别器网络放入 DataParallel 中并移动到 GPU 上。

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError(
                    '{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init) # 将权重初始化应用于生成器网络。
    dis_net.apply(weights_init) # 将权重初始化应用于判别器网络。
    # set up data_loader
    dataset = datasets.ImageDataset(args)   # 使用指定的参数创建 ImageDataset 类的实例
    train_loader = dataset.train            # 从数据集中获取训练数据加载器
    # epoch number for dis_net
    args.max_epoch_D = args.max_epoch_G * args.n_critic # 根设置判别器的最大训练周期数
    if args.max_iter_G:  # 向上取整
        args.max_epoch_D = np.ceil(
            args.max_iter_G * args.n_critic / len(train_loader))    # 根据生成器的迭代次数计算判别器的最大训练周期数。
    max_iter_D = args.max_epoch_D * len(train_loader)
    # set TensorFlow environment for evaluation (calculate IS and FID)
    # _init_inception()
    # 检查或下载用于计算 Inception Score 的 Inception 模型
    inception_path = check_or_download_inception('./tmp/imagenet/')
    create_inception_graph(inception_path)  # 为计算 Inception Score 创建 Inception 图。

    # 检查数据集，设置FID 统计文件路径
    if args.dataset.lower() == 'cifar10':
        fid_stat = './fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = './fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)
    # 使用Adam优化算法设置优化器。filter(lambda p: p.requires_grad, gen_net.parameters()) 用于获取所有需要梯度更新的参数。
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    # 使用线性衰减（LinearLrDecay）设置学习率衰减器
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter_D)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter_D)

    # 初始化训练的起始周期（epoch）和最佳的FID值。
    start_epoch = 0
    best_fid = 1e4

    # 设置日志目录并创建记录器，记录器将日志信息保存到文件中。logger.info(args) 打印并记录命令行参数信息。
    args.path_helper = set_log_dir('exps', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    # 创建一个字典 writer_dict，其中包含用于TensorBoard记录的SummaryWriter对象，以及训练和验证的全局步数。
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    # model size
    logger.info('Param size of G = %fMB', count_parameters_in_MB(gen_net))
    logger.info('Param size of D = %fMB', count_parameters_in_MB(dis_net))
    # genotype_fixG = gan_alg.search(remove=False)
    # 加载事先保存的生成器的最佳基因型（genotype）。
    genotype_fixG = np.load(os.path.join('exps', 'best_G.npy'))
    # genotype_fixG = gan_alg.sample_zero()

    # read supernetG
    # 加载预训练的生成器网络的参数，并设置超网的固定结构。
    ckpt = torch.load(os.path.join('exps', args.model_name))
    gen_net.load_state_dict(ckpt['weight_G'])
    gan_alg.Normal_G_fixed = deepcopy(ckpt['normal_G_fixed'])
    gan_alg.Up_G_fixed = deepcopy(ckpt['up_G_fixed'])
    # up_temp = (np.load(os.path.join('exps', 'Up_G_fixed.npy'))).tolist()
    # normal_temp = (np.load(os.path.join('exps', 'Normal_G_fixed.npy'))).tolist()
    # gan_alg.Normal_G_fixed = deepcopy(normal_temp)
    # gan_alg.Up_G_fixed = deepcopy(up_temp)
    # 创建生成器训练器，用于训练生成器网络。
    trainer_gen = GenTrainer(args, gen_net, dis_net, gen_optimizer,
                             dis_optimizer, train_loader, gan_alg, None,
                             genotype_fixG)
    best_genotypes = None
    # search genarator
    # 生成初始的生成器架构种群
    ll = []
    for i in range(args.num_individual):
        ll.append([])
        while (True):
            a1 = gan_alg.search()
            if trainer_gen.judege_model_size(a1, limit=args.max_model_size):
                break
        ll[i] = a1
    population = np.stack(ll)

    # 使用进化算法进行生成器结构的搜索，记录并保存生成器的性能指标。
    record_is = []
    record_fid = []

    for ii in tqdm(range(args.Total_evolutionary_algebra), desc='search genearator using evo alg'):
        population, pop_selected, a, b, is_record, fid_record = trainer_gen.my_search_evolv2(population, fid_stat, ii)
        record_is.append(is_record.tolist())
        record_fid.append(fid_record.tolist())

    # 保存搜索到的最佳生成器结构，并将生成器的性能指标记录到文件中。
    for index, geno in enumerate(pop_selected):
        file_path = os.path.join(args.path_helper['ckpt_path'],
                                 "best_gen_{}.npy".format(str(index)))
        np.save(file_path, geno)
    file = open("IS_record.txt", 'w')
    for fp in record_is:
        file.write(str(fp))
        file.write('\n')
    file.close()

    file = open("FID_record.txt", 'w')
    for fp in record_fid:
        file.write(str(fp))
        file.write('\n')
    file.close()


if __name__ == '__main__':
    main()
