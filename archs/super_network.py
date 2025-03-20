from torch import nn
from archs.basic_blocks_search import Cell, DisCell, OptimizedDisBlock


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width

        # 根据数据集不同设置基础的 latent dim
        if args.dataset == 'cifar10':
            # for lower resolution (32 * 32) dataset CIFAR-10
            self.base_latent_dim = args.latent_dim // 3
        else:
            # for higher resolution (48 * 48) dataset STL-10
            self.base_latent_dim = args.latent_dim // 2
        # 三个线性层用于映射输入的噪声向量到不同分辨率的特征图
        self.l1 = nn.Linear(self.base_latent_dim,
                            (self.bottom_width ** 2) * args.gf_dim)
        self.l2 = nn.Linear(self.base_latent_dim, ((self.bottom_width * 2) ** 2) * args.gf_dim)
        # 根据数据集不同，可能需要额外的线性层
        if args.dataset == 'cifar10':
            self.l3 = nn.Linear(self.base_latent_dim, ((self.bottom_width * 4) ** 2) * args.gf_dim)
        # 三个cell用于构建生成器网络结构
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=0)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'bilinear', num_skip_in=1)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=2)
        # 最终输出通过卷积层转化为RGB图像
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim), nn.ReLU(), nn.Conv2d(
                args.gf_dim, 3, 3, 1, 1), nn.Tanh()
        )

    def forward(self, z, genotypes):
        # 第一个线性层映射输入噪声向量到特征图
        h = self.l1(z[:, :self.base_latent_dim]) \
            .view(-1, self.ch, self.bottom_width, self.bottom_width)
        # 第二个线性层映射到更高分辨率的特征图
        n1 = self.l2(z[:, self.base_latent_dim:self.base_latent_dim * 2]) \
            .view(-1, self.ch, self.bottom_width * 2, self.bottom_width * 2)
        # 如果是CIFAR-10数据集，可能需要额外的线性层映射
        if self.args.dataset == 'cifar10':
            n2 = self.l3(z[:, self.base_latent_dim * 2:]) \
                .view(-1, self.ch, self.bottom_width * 4, self.bottom_width * 4)
        # 通过三个细胞构建生成器网络结构
        h1_skip_out, h1 = self.cell1(h, genotype=genotypes[0])
        h2_skip_out, h2 = self.cell2(h1 + n1, (h1_skip_out,), genotype=genotypes[1])
        _, h3 = self.cell3(h2 + n2, (h1_skip_out, h2_skip_out), genotype=genotypes[2])
        # 最终输出通过卷积层转化为RGB图像
        output = self.to_rgb(h3)

        return output, [h1, h2, h3]

    def get_subnet_parameters(self, genotypes):
        """
        根据给定的 genotypes 返回子网的参数
        """
        subnet_params = []

        # 添加线性层的参数
        subnet_params.extend(list(self.l1.parameters()))
        subnet_params.extend(list(self.l2.parameters()))
        if self.args.dataset == 'cifar10':
            subnet_params.extend(list(self.l3.parameters()))

        # 根据 genotypes 选择 cell 的路径并添加 cell 的参数
        subnet_params.extend(self.cell1.get_selected_parameters(genotype=genotypes[0]))
        subnet_params.extend(self.cell2.get_selected_parameters(genotype=genotypes[1]))
        subnet_params.extend(self.cell3.get_selected_parameters(genotype=genotypes[2]))

        # 添加最终 to_rgb 的参数
        subnet_params.extend(list(self.to_rgb.parameters()))

        return subnet_params

    def update_student_weights_only(self, selected_params, grad_kd, optimizer):
        param_dict = {p: p.grad for p in selected_params}
        for i in range(len(grad_kd)):
            param = selected_params[i]
            param_dict[param] = grad_kd[i]

        # 清除之前的梯度
        optimizer.zero_grad()

        # 仅更新选中的参数
        for param in self.parameters():
            if param in param_dict:
                param.grad = param_dict[param]

        optimizer.step()


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisCell(args, self.ch, self.ch, activation=activation)
        self.block3 = DisCell(args, self.ch, self.ch, activation=activation)
        self.block4 = DisCell(args, self.ch, self.ch, activation=activation)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x, genotypes):
        h = x
        h = self.block1(h)
        h = self.block2(h, genotypes[0])
        h = self.block3(h, genotypes[1])
        h = self.block4(h, genotypes[2])
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output
