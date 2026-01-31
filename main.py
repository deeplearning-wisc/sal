"""
SAL (Softmax Aggregation Layer) - 分布外检测实现
主要功能：使用基于梯度的方法检测混合数据中的OOD样本
"""
import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from Log import MyLog
import random

# functorch: PyTorch的函数式编程工具，用于高效计算每个样本的梯度
from functorch import make_functional_with_buffers, vmap, grad
from wrn import WideResNet


# 余弦退火学习率调度函数
def cosine_annealing(step, total_steps, lr_max, lr_min):
    """
    计算余弦退火学习率
    Args:
        step: 当前步数
        total_steps: 总步数
        lr_max: 最大学习率
        lr_min: 最小学习率
    Returns:
        当前步的学习率
    """
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

class SAL:
    """
    SAL (Softmax Aggregation Layer) 类
    核心功能：使用基于梯度的SVD方法检测分布外（OOD）样本
    """

    def __init__(self,
                 args,
                 logger,
                 p_def=0.03,
                 num_rounds=5,
                 seed=12345):
        """
        初始化SAL检测器
        Args:
            args: 命令行参数
            logger: 日志记录器
            p_def: 预期的OOD数据比例（默认3%）
            num_rounds: 过滤轮数（默认5轮）
            seed: 随机种子
        """
        super(SAL, self).__init__()

        # 过滤轮数
        self.num_rounds = num_rounds
        # 预期OOD比例
        self.p_def = p_def
        self.args = args

        # 如果不在混合数据上计算均值，则初始化curmean用于存储干净数据的均值
        if not self.args.avg_on_wild:
            self.curmean = None

        # 初始化Wide ResNet-40-2网络（depth=40, widen_factor=2, dropout=0.3）
        self.net = WideResNet(40, self.args.num_class, 2, dropRate=0.3).cuda()
        self.net.train()  # 设置为训练模式（影响dropout和batchnorm）
        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        np.random.seed(seed)

    def filterByClassNoConditional(self, losses, y, gradients):
        """
        核心OOD检测函数：使用基于梯度的SVD方法按类别过滤OOD样本

        算法步骤：
        1. 按预测类别分组样本
        2. 对每个类别内的梯度进行中心化（减去均值）
        3. 对中心化后的梯度进行SVD分解
        4. 计算每个样本在主奇异向量上的投影长度作为得分
        5. 使用阈值过滤OOD样本

        Args:
            losses: 每个样本的损失值
            y: 每个样本的预测标签
            gradients: 每个样本的梯度 (n, num_classes, feature_dim)

        Returns:
            re_indices: 被判定为ID的样本索引
            re_scores: 每个样本的归一化得分
        """
        n = gradients.shape[0]

        allIndices = np.arange(n)
        labels = np.unique(y)  # 获取所有唯一的类别标签

        def filterSimple(g, p, m, curindices, class_index):
            """
            对单个类别内的样本进行SVD过滤
            Args:
                g: 当前类别的梯度
                p: OOD比例（未使用）
                m: 该类别的平均梯度
                curindices: 当前类别样本的索引
                class_index: 当前类别的索引
            """
            k = self.args.num_sing_vectors  # 使用的主奇异向量数量（通常为1）

            # 梯度中心化：减去该类别的平均梯度，只取对应类别的梯度分量
            gcentered = (g - m)[:, class_index, :]

            # 奇异值分解（SVD）
            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)

            # 提取前k个主奇异向量作为投影矩阵
            projection = V_p[:k, :].T

            # 计算投影得分：在主子空间上的投影长度
            # 得分越大表示越符合ID数据的梯度分布
            scores = np.mean(np.matmul(gcentered, projection), -1, keepdims=True)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))

            # 保存每个样本的得分
            self.scores_draw[curindices] = scores

        print('the length of the scores_draw is:', len(self.scores_draw))
        self.p_def = self.args.ood_rate
        print("self.p_def: ", self.p_def)

        # 收集所有wild数据的梯度（用于后续分析）
        self.gradident_wild = None
        for i in labels:
            # 获取当前类别的样本索引
            curIndices = allIndices[y == i]

            # 提取当前类别的梯度
            curGradients = gradients[curIndices, :]
            # 计算当前类别在wild数据上的平均梯度
            curMean = np.mean(gradients[curIndices, :],
                              axis=0, keepdims=True)  # (1, d)

            # 保存wild数据的梯度（只保存对应类别的梯度分量）
            if self.gradident_wild is None:
                self.gradident_wild = curGradients[:, int(i), :]
            else:
                self.gradident_wild = np.concatenate([self.gradident_wild,
                    curGradients[:, int(i), :]], 0)

            # 根据配置选择使用干净数据均值还是wild数据均值
            if not self.args.avg_on_wild:
                # 使用干净ID数据上计算的均值（更鲁棒）
                filterSimple(curGradients,
                         self.p_def,
                         self.curmean[int(i)], curIndices, int(i))
            else:
                # 使用wild数据上计算的均值
                filterSimple(curGradients,
                         self.p_def,
                         curMean, curIndices, int(i))

        # 打印得分的分位数
        print(np.quantile(self.scores_draw, 1 - self.p_def))
        indices = np.arange(len(self.scores_draw))

        # 根据阈值过滤样本
        if np.quantile(self.scores_draw, 1 - self.p_def) > 0:
            if self.args.use_thres:
                # 使用在干净数据上计算的阈值进行归一化
                scores = self.scores_draw / se.thres

            # 保留得分小于等于1.0的样本（判定为ID）
            indices = indices[scores <= 1.0]
        else:
            # 使用最大值归一化
            scores = self.scores_draw / np.max(self.scores_draw)

        re_indices = indices
        re_scores = scores
        re_indices.sort()
        return re_indices, re_scores



    def GetThresNoConditional(self, losses, y, gradients):
        """
        在干净ID数据上计算阈值（用于后续过滤wild数据）

        与filterByClassNoConditional类似，但只计算得分并返回指定分位数作为阈值
        不执行过滤操作

        Args:
            losses: 每个样本的损失值
            y: 每个样本的预测标签
            gradients: 每个样本的梯度

        Returns:
            threshold: 得分的指定分位数值（如95%分位数）
        """
        n = gradients.shape[0]

        allIndices = np.arange(n)
        labels = np.unique(y)

        def filterSimple(g, p, m, curindices, class_index):
            """计算每个样本的SVD投影得分"""
            k = self.args.num_sing_vectors
            # 梯度中心化
            gcentered = (g - m)[:, class_index, :]

            # SVD分解
            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)

            # 投影到主子空间
            projection = V_p[:k, :].T

            # 计算投影得分（在主成分上的投影幅度）
            scores = np.mean(np.matmul(gcentered, projection), -1, keepdims=True)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))

            self.scores_draw[curindices] = scores

        self.p_def = self.args.ood_rate

        print('the length of the scores_draw is:', len(self.scores_draw))

        # 对每个类别分别计算得分
        for i in labels:
            curIndices = allIndices[y == i]
            curGradients = gradients[curIndices, :]
            curMean = np.mean(gradients[curIndices, :],
                              axis=0, keepdims=True)  # (1, d)

            if not self.args.avg_on_wild:
                # 使用预先计算的干净数据均值
                filterSimple(curGradients,
                         self.p_def,
                         self.curmean[int(i)], curIndices, int(i))
            else:
                # 使用当前数据的均值
                filterSimple(curGradients,
                         self.p_def,
                         curMean, curIndices, int(i))

        # 返回指定分位数作为阈值（例如95%分位数）
        return np.quantile(self.scores_draw, self.args.thres_id)


    def get_threshold(self, train_loader):
        """
        在干净ID训练数据上计算阈值

        Args:
            train_loader: 干净ID数据的DataLoader

        Returns:
            thres: 在干净数据上计算的阈值
        """
        for epoch in range(self.num_rounds):
            # 计算所有样本的梯度
            gradients, losses, targets = self.train_no_gradients(train_loader)
        # 基于梯度计算阈值
        thres = self.GetThresNoConditional(losses,
                                           targets,
                                           gradients)
        return thres


    def get_mask(self, X_aux_in, X_aux_out):
        """
        在混合wild数据上执行OOD检测，生成ID/OOD mask

        这是主要的检测流程函数，处理包含ID和OOD样本的混合数据集

        Args:
            X_aux_in: 辅助ID数据的DataLoader
            X_aux_out: 辅助OOD数据的DataLoader（用于构建混合数据集）

        Returns:
            mask: 检测结果mask，1表示ID样本，-1表示OOD样本
            mask_all: ground truth mask（用于评估）
        """

        self.logger.info("======> Defensing!")
        for epoch in range(self.num_rounds):

            self.logger.info("=====> round %d" % epoch)

            # 在混合数据上计算梯度
            gradients, losses, targets, mask_all = self.train_no_gradients_mixture(X_aux_in,
                                                                         X_aux_out)
            self.logger.info("=====> current training losses %.4f" % (np.sum(losses) / len(losses)))

            self.logger.info("=====> filterSimple")
            gradients = gradients.squeeze()
            self.scores_draw = np.zeros(len(targets))  # 初始化得分数组

            # 使用SVD方法过滤OOD样本
            indices, outlier_scores = self.filterByClassNoConditional(losses,
                                                         targets,
                                                         gradients)

            # 记录过滤掉的样本数量
            self.logger.info("=====> filtering %d samples" % (len(targets) - len(indices)))

            # 如果没有过滤掉任何样本或过滤掉了所有样本，提前结束
            if len(indices) == len(targets) or len(indices) == 0:
                break

            active_indices = indices

        # 生成最终的mask：-1表示OOD，1表示ID
        mask = np.asarray([-1 for i in range(len(targets))])  # 默认为OOD
        mask[active_indices] = 1  # 保留的样本标记为ID

        return mask, mask_all


    def train_no_gradients(self, X):
        """
        在单个数据集上计算所有样本的梯度（不进行训练更新）

        Args:
            X: DataLoader，包含数据和标签

        Returns:
            gradients_all: 所有样本的梯度 (n, num_classes, feature_dim)
            losses_all: 所有样本的损失值
            target_all: 所有样本的标签
        """
        index = 0
        for data, target in X:
            # 计算每个样本的梯度
            losses, gradients = self.per_sample_gradient(
                data.cuda(),
                target.cuda())

            # 第一个batch：初始化数组
            if index == 0:
                target_all = target.cpu().numpy()
                losses_all = losses.detach().cpu().numpy()
                gradients_all = gradients.detach().cpu().numpy()
            else:
                # 后续batch：拼接到现有数组
                target_all = np.concatenate([target_all,
                                             target.cpu().numpy()], 0)
                losses_all = np.concatenate([losses_all,
                                             losses.detach().cpu().numpy()], 0)
                gradients_all = np.concatenate([gradients_all,
                                                 gradients.detach().cpu().numpy()], 0)
            index += 1

        return gradients_all, losses_all, target_all



    def train_no_gradients_mixture(self, X_in, X_out):
        """
        在混合数据集（ID + OOD）上计算所有样本的梯度

        关键特点：
        - 将ID和OOD数据混合在一起
        - 使用模型预测的伪标签（argmax）而不是真实标签
        - 保存ground truth mask用于后续评估

        Args:
            X_in: ID数据的DataLoader
            X_out: OOD数据的DataLoader

        Returns:
            gradients_all: 所有样本的梯度
            losses_all: 所有样本的损失值
            target_all: 所有样本的伪标签（模型预测）
            mask_all: ground truth mask (1=OOD, 0=ID)
        """
        index = 0
        batch_iterator = iter(X_in)

        # 遍历OOD数据
        for data, target in X_out:
            # 获取对应的ID数据batch
            try:
                in_set = next(batch_iterator)
            except StopIteration:
                # 如果ID数据用完了，重新开始迭代
                batch_iterator = iter(X_in)
                in_set = next(batch_iterator)

            # 混合ID和OOD数据
            aux_set = torch.cat([data, in_set[0]], 0)

            # 保存所有数据（用于后续训练二分类器）
            if index == 0:
                self.kept_data_all = aux_set
            else:
                self.kept_data_all = torch.cat([self.kept_data_all, aux_set], 0)

            # 创建ground truth mask：1表示OOD，0表示ID
            mask = np.concatenate([np.ones(len(data)),
                                        np.zeros(len(in_set[0]))], 0)
            aux_set = aux_set.cuda()

            # 生成伪标签：使用模型预测的类别（argmax）
            target = self.net(aux_set).argmax(-1)

            # 计算每个样本的梯度
            losses, gradients = self.per_sample_gradient(
                aux_set,
                target)

            # 收集所有batch的结果
            if index == 0:
                target_all = target.detach().cpu()
                losses_all = losses.detach().cpu()
                gradients_all = gradients.detach().cpu()
                mask_all = mask
            else:
                target_all = torch.cat([target_all,
                                             target.detach().cpu()], 0)
                losses_all = torch.cat([losses_all,
                                             losses.detach().cpu()], 0)
                gradients_all = torch.cat([gradients_all,
                                                 gradients.detach().cpu()], 0)
                mask_all = np.concatenate([mask_all, mask], 0)

            index += 1
            # 每10个batch打印进度
            if index % 10 == 0:
                print(index * 128)

        # 转换为numpy数组
        gradients_all = gradients_all.numpy()
        losses_all = losses_all.numpy()
        target_all = target_all.numpy()
        self.kept_data_all = self.kept_data_all.numpy()

        return gradients_all, losses_all, target_all, mask_all

    def per_sample_gradient(self, x, y):
        """
        使用functorch计算每个样本的梯度（关键函数）

        这是SAL算法的核心：高效地计算每个样本对最后一层参数的梯度
        传统方法需要对每个样本单独backward，这里使用functorch的vmap并行化计算

        Args:
            x: 输入数据 (batch_size, ...)
            y: 标签 (batch_size,)

        Returns:
            losses: 每个样本的交叉熵损失 (batch_size,)
            gradients: 每个样本的梯度 (batch_size, num_classes, feature_dim=128)
                      这里返回倒数第二层（ft_per_sample_grads[-2]）的梯度
        """
        # 将网络转换为函数式表示（无状态模型）
        fmodel, params, buffers = make_functional_with_buffers(self.net)

        def compute_loss_stateless_model(params, buffers, sample, target):
            """
            计算单个样本的损失（无状态版本）
            Args:
                params: 模型参数
                buffers: 模型缓冲区（如BatchNorm的running stats）
                sample: 单个样本
                target: 单个标签
            """
            batch = sample.unsqueeze(0)  # 添加batch维度
            targets = target.unsqueeze(0)

            predictions = fmodel(params, buffers, batch)
            loss = self.criterion(predictions, targets.long())
            return loss

        # 第一步：使用vmap并行计算每个样本的损失
        ft_compute_grad = compute_loss_stateless_model
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        losses = ft_compute_sample_grad(params, buffers, x, y)

        # 第二步：使用vmap + grad并行计算每个样本的梯度
        # grad()对损失函数求导，vmap()对batch中的每个样本并行化
        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x, y)

        # 返回倒数第二层的梯度（最后一层是输出层，倒数第二层是特征层）
        # 对于Wide ResNet，这是128维的特征层
        return losses, ft_per_sample_grads[-2]


    def cal_avg_clean(self, X):
        """
        在干净ID数据上计算每个类别的平均梯度

        这个函数用于预计算干净数据的梯度统计信息，用作后续过滤的基准

        Args:
            X: 干净ID数据的DataLoader

        Returns:
            gradient_list: 每个类别的平均梯度列表
                          gradient_list[i] 是类别i的平均梯度 (num_classes, 128)
        """
        # 为每个类别初始化梯度列表
        gradient_list = [np.zeros((0, self.args.num_class, 128))
                         for _ in range(self.args.num_class)]
        labels = np.arange(self.args.num_class)

        # 用于保存所有ID数据的梯度（只保留对应类别的梯度分量）
        self.gradident_id = None

        # 对每个类别分别收集梯度
        for i in labels:
            for data, target in X:
                allIndices = np.arange(len(target))
                # 找到当前batch中属于类别i的样本
                curIndices = allIndices[target == i]

                # 计算这些样本的梯度
                _, gradients = self.per_sample_gradient(
                    data[curIndices].cuda(),
                   target[curIndices].cuda())

                # 收集该类别的所有梯度
                gradient_list[i] = np.concatenate([gradient_list[i],
                                                   gradients.detach().cpu().numpy()], 0)

                # 保存ID数据的梯度（只保留对应类别的梯度分量）
                if self.gradident_id is None:
                    self.gradident_id = gradients[:, int(i), :].detach().cpu().numpy()
                else:
                    self.gradident_id = np.concatenate([self.gradident_id,
                                         gradients[:, int(i), :].detach().cpu().numpy()],0)

        # 计算每个类别的平均梯度
        for i in labels:
            gradient_list[i] = np.mean(gradient_list[i],
                                         axis=0, keepdims=True)[0]

        return gradient_list





def test(net, test_loader):
    """
    测试网络在测试集上的准确率

    Args:
        net: 神经网络模型
        test_loader: 测试数据的DataLoader

    Returns:
        accuracy: 分类准确率
    """
    net.eval()  # 设置为评估模式（禁用dropout和BatchNorm的训练行为）
    loss_avg = 0.0
    correct = 0

    with torch.no_grad():  # 不计算梯度，节省内存
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # 前向传播
            output = net(data)

            # 计算损失
            loss = F.cross_entropy(output, target)

            # 计算准确率
            pred = output.data.max(1)[1]  # 获取预测类别
            correct += pred.eq(target.data).sum().item()  # 统计正确预测数量

            # 累积测试损失
            loss_avg += float(loss.data)

    net.train()  # 恢复训练模式
    return correct / len(test_loader.dataset)


if __name__ == '__main__':
    """
    主函数：SAL OOD检测的完整流程

    主要步骤：
    1. 参数解析和配置
    2. 数据集加载
    3. 初始化SAL检测器并加载预训练模型
    4. 在干净数据上计算阈值和均值
    5. 在混合数据上执行OOD检测
    6. 训练二分类器用于最终的OOD检测
    7. 评估性能（AUROC, AUPR, FPR@95）
    """
    # ==================== 参数定义 ====================
    parser = argparse.ArgumentParser()

    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'],
                       help='ID数据集（CIFAR-10或CIFAR-100）')
    parser.add_argument('--aux_out_dataset', type=str, default='svhn',
                        choices=['svhn', 'lsun_c', 'lsun_r',
                            'isun', 'dtd', 'places',
                            'tinyimages_300k', 'cifar100'],
                        help='辅助OOD数据集，用于构建混合数据')
    parser.add_argument('--test_out_dataset', type=str,
                        choices=['svhn', 'lsun_c', 'lsun_r',
                                    'isun', 'dtd', 'places', 'tinyimages_300k', 'cifar100'],
                        default='svhn', help='测试用OOD数据集')
    parser.add_argument('--pi', type=float, default=0.1,
                        help='混合数据中OOD样本的比例（如0.1表示10%）')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--prefetch', type=int, default=4,
                        help='数据预取线程数')

    # SAL算法相关参数
    parser.add_argument('--num_rounds', type=int, default=1,
                       help='过滤轮数')
    parser.add_argument('--woods_setting', type=int, default=1,
                       help='是否使用WOODS设置')
    parser.add_argument('--avg_on_wild', type=int, default=0,
                       help='是否在wild数据上计算均值（0=使用干净数据均值，更鲁棒）')
    parser.add_argument('--use_thres', type=int, default=1,
                       help='是否使用预计算的阈值')
    parser.add_argument('--num_class', type=int, default=100,
                       help='类别数量')
    parser.add_argument('--thres_id', type=float, default=0.95,
                       help='在干净数据上计算阈值的分位数（如0.95表示95%分位数）')
    parser.add_argument('--draw_dis', type=int, default=0,
                       help='是否绘制得分分布图')
    parser.add_argument('--plot', type=int, default=0,
                       help='是否绘图')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='权重衰减')
    parser.add_argument('--ood_rate', type=float, default=0.1,
                       help='预期的OOD比例')
    parser.add_argument('--seed', type=int, default=0,
                       help='随机种子')
    parser.add_argument('--num_sing_vectors', type=int, default=1,
                       help='使用的主奇异向量数量')

    # 二分类器训练相关参数
    parser.add_argument('--train_binary_classifier', type=int, default=1,
                       help='是否训练二分类器')
    parser.add_argument('--load_full_classifier', type=int, default=1,
                       help='是否加载预训练的完整分类器')
    parser.add_argument('--ft_epochs', type=int, default=100,
                       help='微调轮数')
    parser.add_argument('--ft_weight', type=int, default=10,
                       help='OOD损失的权重')
    parser.add_argument('--loss_add', type=int, default=1,
                       help='是否将分类损失和OOD损失相加（1=相加，0=只用OOD损失）')

    args = parser.parse_args()

    # ==================== 日志配置 ====================
    name = '{}_{}_{}_{}'.format(args.dataset, args.pi,
                                args.aux_out_dataset,
                                args.test_out_dataset)
    log_dir = 'log/{}'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = MyLog(os.path.join(log_dir, name + '.log'))
    logger.info(args)




    # ==================== 随机种子设置 ====================
    seed = args.seed
    device = 'cuda:0'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy 随机种子
    random.seed(seed)  # Python 随机模块种子

    # ==================== 数据集加载 ====================
    state = {k: v for k, v in args._get_kwargs()}
    from make_dataset import make_datasets_my

    # 加载所有需要的数据集
    # train_loader_in: 干净的ID训练数据
    # train_loader_in_large_bs: 大batch size的ID训练数据（用于计算统计信息）
    # train_loader_aux_in: 辅助ID数据（用于构建混合数据集）
    # train_loader_aux_out: 辅助OOD数据（用于构建混合数据集）
    # test_loader_in: ID测试数据
    # test_loader_out: OOD测试数据
    train_loader_in, train_loader_in_large_bs, train_loader_aux_in, \
    train_loader_aux_out, test_loader_in, \
    test_loader_out = make_datasets_my(
        args.dataset, args.aux_out_dataset,
        args.test_out_dataset, args.ood_rate, state)

    # 打印数据集大小信息
    print("\n len(train_loader_in.dataset) {} " \
          "len(train_loader_aux_in.dataset) {}, " \
          "len(train_loader_aux_out.dataset) {}, " \
          "len(test_loader_in.dataset) {}, " \
          "len(test_loader_ood.dataset) {}, ".format(
        len(train_loader_in.dataset),
        len(train_loader_aux_in.dataset),
        len(train_loader_aux_out.dataset),
        len(test_loader_in.dataset),
        len(test_loader_out.dataset)))



    # ==================== 阶段1：OOD检测 ====================
    if args.woods_setting:
        # 初始化SAL检测器
        se = SAL(args,
                   logger,
                   p_def=args.ood_rate,
                   num_rounds=args.num_rounds,
                   seed=args.seed)
        se.rng = np.random.default_rng(args.seed)

        # 加载预训练模型
        if args.load_full_classifier:
            if args.num_class == 10:
                se.net.load_state_dict(torch.load('./pretrained/cifar10_wrn_pretrained_epoch_99.pt'))
            else:
                se.net.load_state_dict(torch.load(
                    './pretrained/cifar100_wrn_pretrained_epoch_99.pt'))
        else:
            se.net.load_state_dict(torch.load(args.dataset + '.pt'))
        se.net.eval()  # 设置为评估模式

        # 步骤1：在干净ID数据上计算每个类别的平均梯度
        if not args.avg_on_wild:
            curmean = se.cal_avg_clean(train_loader_in_large_bs)
            se.curmean = curmean

        # 步骤2：在干净ID数据上计算阈值
        if args.use_thres:
            se.scores_draw = np.zeros(len(train_loader_in_large_bs.dataset))
            thres = se.get_threshold(train_loader_in)
            se.thres = thres
            print('thres is :', se.thres)

        # 步骤3：在混合数据（ID + OOD）上执行OOD检测
        mask, mask_gt = se.get_mask(train_loader_aux_in,
                                                  train_loader_aux_out)

        # 步骤4：评估OOD检测性能
        from sklearn.metrics import f1_score
        print('###########################')
        print('OOD detection precision: ', np.sum((mask==-1)[mask_gt == 1]) / (mask==-1).sum())
        print('OOD detection recall: ', np.sum((mask==-1)[mask_gt == 1]) / (mask_gt == 1).sum())
        print('f1 score is: ', f1_score(mask == -1, mask_gt))
        print('###########################')

    # 计算ID数据和Wild数据梯度之间的平均距离（用于分析）
    dist = torch.cdist(torch.from_numpy(se.gradident_id).unsqueeze(0),
                       torch.from_numpy(se.gradident_wild).unsqueeze(0),
                       compute_mode='donot_use_mm_for_euclid_dist')
    print('Average gradient distance between ID and Wild:', dist.mean())

    # ==================== 可视化：绘制得分分布 ====================
    import matplotlib.pyplot as plt

    if args.draw_dis:
        # 绘制ID样本和OOD样本的得分分布对比图
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        import pandas as pd
        import seaborn as sns

        # 分别提取ID和OOD样本的得分
        id_pd = pd.Series(se.scores_draw[mask_gt == False])  # ID样本得分
        ood_pd = pd.Series(se.scores_draw[mask_gt == True])  # OOD样本得分

        # 绘制核密度估计图
        p1 = sns.kdeplot(id_pd, shade=True, color="r", label='ID')
        p1 = sns.kdeplot(ood_pd, shade=True, color="b", label='OOD')
        plt.legend()
        plt.savefig('score_dis_cifar.jpg', dpi=250)
        print('Score distribution plot saved to score_dis_cifar.jpg')


    # ==================== 阶段2：训练二分类OOD检测器 ====================
    if args.train_binary_classifier:
        # 步骤1：提取检测出的OOD样本
        loaded_data = se.kept_data_all[mask == -1]  # 取出被标记为OOD的样本

        # 步骤2：初始化二分类器
        # 使用Wide ResNet作为特征提取器
        binary_classifier = WideResNet(40, args.num_class, 2, dropRate=0.3).cuda()
        # Logistic回归层：从128维特征映射到1维二分类得分
        logistic_regression = torch.nn.Sequential(
            torch.nn.Linear(128, 1))

        logistic_regression = logistic_regression.cuda().train()
        binary_classifier.train()

        # 步骤3：配置优化器
        # 同时优化特征提取器和logistic回归层的参数
        binary_cls_optimizer = torch.optim.SGD(
            list(binary_classifier.parameters()) + list(logistic_regression.parameters()),
            momentum=0.9,
            nesterov=True,
            lr=0.001,
            weight_decay=args.weight_decay)

        # 步骤4：加载预训练模型
        if args.load_full_classifier:
            if args.num_class == 10:
                binary_classifier.load_state_dict(torch.load(
                    './pretrained/cifar10_wrn_pretrained_epoch_99.pt'))
            else:
                binary_classifier.load_state_dict(torch.load(
                    './pretrained/cifar100_wrn_pretrained_epoch_99.pt'))
        else:
            binary_classifier.load_state_dict(torch.load(args.dataset + '.pt'))

        # 准备OOD数据
        ood_data_length = len(loaded_data)
        permutation_idx = torch.randperm(ood_data_length)  # 随机打乱OOD数据
        batch_begin = 0
        criterion = torch.nn.CrossEntropyLoss()

        # 步骤5：配置学习率调度器（余弦退火）
        binary_scheduler = torch.optim.lr_scheduler.LambdaLR(
            binary_cls_optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.ft_epochs * len(train_loader_in),
                1,  # lr_lambda计算的是乘法因子
                1e-6 / 0.001))

        # 步骤6：训练循环
        loaded_data = torch.from_numpy(loaded_data).cuda()
        for epoch in range(args.ft_epochs):
            for in_set in train_loader_in:  # 遍历ID数据
                binary_cls_optimizer.zero_grad()

                # 如果OOD数据不够了，重新打乱
                if ood_data_length - batch_begin < args.batch_size:
                    batch_begin = 0
                    permutation_idx = torch.randperm(ood_data_length)

                # 混合ID和OOD数据
                mixed_data = torch.cat([in_set[0].cuda(),
                                       loaded_data[permutation_idx][batch_begin:batch_begin+args.batch_size]], 0)
                out, out_logits = binary_classifier.forward_my1(mixed_data)

                batch_begin += args.batch_size

                # 损失1：ID数据的分类损失
                loss = F.cross_entropy(out_logits[:len(in_set[1])], in_set[1].cuda())

                # 损失2：二分类损失（区分ID和OOD）
                output1 = logistic_regression(out)
                # 创建二分类标签：1表示ID，0表示OOD
                binary_labels = torch.ones(len(in_set[1]) + args.batch_size).cuda()
                binary_labels[len(in_set[1]):] = 0  # OOD样本标记为0

                # 计算二分类交叉熵损失
                energy_reg_loss = F.binary_cross_entropy_with_logits(output1.view(-1), binary_labels.float())

                # 组合两个损失
                if args.loss_add:
                    loss += args.ft_weight * energy_reg_loss  # 加权相加
                else:
                    loss = args.ft_weight * energy_reg_loss  # 只用OOD损失

                # 反向传播和优化
                loss.backward()
                binary_cls_optimizer.step()
                binary_scheduler.step()

            # 每个epoch结束后测试准确率
            print('Epoch: ', epoch, 'Acc:', test(binary_classifier, test_loader_in))
            print('Loss: ', loss)

        # 步骤7：评估模式
        binary_classifier.eval()
        logistic_regression.eval()

        def test_ood_function(test_loader_in):
            """
            在测试集上计算OOD得分
            Returns:
                得分数组，得分越高表示越可能是ID
            """
            index = 0
            with torch.no_grad():
                for in_set in test_loader_in:
                    out, _ = binary_classifier.forward_my1(in_set[0].cuda())

                    if index == 0:
                        logistic_all = logistic_regression(out).view(-1)
                    else:
                        logistic_all = torch.cat([logistic_all, logistic_regression(out).view(-1)], -1)
                    index += 1
            return logistic_all.cpu().detach().numpy()

        # 步骤8：在测试集上计算OOD得分
        energy_id = test_ood_function(test_loader_in)    # ID测试数据的得分
        energy_ood = test_ood_function(test_loader_out)  # OOD测试数据的得分

        # 步骤9：计算并打印评估指标（AUROC, AUPR, FPR@95）
        from utils.metric_utils import get_measures, print_measures
        measures = get_measures(energy_id, energy_ood, plot=False)
        print_measures(measures[0], measures[1], measures[2], 'energy')







