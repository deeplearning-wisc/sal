import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from Log import MyLog
import random

from functorch import make_functional_with_buffers, vmap, grad
from wrn import WideResNet



def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

class SAL:


    def __init__(self,
                 args,
                 logger,
                 p_def=0.03,
                 num_rounds=5,
                 seed=12345):
        super(SAL, self).__init__()


        self.num_rounds = num_rounds

        self.p_def = p_def
        self.args = args
        if not self.args.avg_on_wild:
            self.curmean = None


        self.net = WideResNet(40, self.args.num_class, 2, dropRate=0.3).cuda()
        self.net.train()
        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss()
        np.random.seed(seed)

    def filterByClassNoConditional(self, losses, y, gradients):
        n = gradients.shape[0]

        allIndices = np.arange(n)
        labels = np.unique(y)

        def filterSimple(g, p, m, curindices, class_index):
            k = self.args.num_sing_vectors
            gcentered = (g - m)[:, class_index, :] #/ np.sqrt(N_filt)

            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)

            projection = V_p[:k, :].T

            # %Scores are the magnitude of the projection onto the top principal component
            scores = np.mean(np.matmul(gcentered, projection), -1, keepdims=True)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))

            self.scores_draw[curindices] = scores

        print('the length of the scores_draw is:', len(self.scores_draw))
        self.p_def = self.args.ood_rate
        print("self.p_def: ", self.p_def)


        self.gradident_wild = None
        for i in labels:

            curIndices = allIndices[y == i]

            # print(curIndices.shape)
            curGradients = gradients[curIndices, :]
            curMean = np.mean(gradients[curIndices, :],
                              axis=0, keepdims=True)  # (1, d)
            if self.gradident_wild is None:
                self.gradident_wild = curGradients[:, int(i), :]
            else:
                self.gradident_wild = np.concatenate([self.gradident_wild,
                    curGradients[:, int(i), :]], 0)
            if not self.args.avg_on_wild:
                filterSimple(curGradients,
                         self.p_def,
                         self.curmean[int(i)], curIndices, int(i))
            else:
                filterSimple(curGradients,
                         self.p_def,
                         curMean, curIndices, int(i))

        print(np.quantile(self.scores_draw, 1 - self.p_def))
        indices = np.arange(len(self.scores_draw))

        if np.quantile(self.scores_draw, 1 - self.p_def) > 0:
            if self.args.use_thres:
                scores = self.scores_draw / se.thres#np.quantile(self.scores_draw, 1 - self.p_def)

            indices = indices[scores <= 1.0]
        else:
            scores = self.scores_draw / np.max(self.scores_draw)
        re_indices = indices
        re_scores = scores
        re_indices.sort()
        return re_indices, re_scores



    def GetThresNoConditional(self, losses, y, gradients):
        n = gradients.shape[0]

        allIndices = np.arange(n)
        labels = np.unique(y)

        def filterSimple(g, p, m, curindices, class_index):
            k = self.args.num_sing_vectors
            gcentered = (g - m)[:, class_index, :] #/ np.sqrt(N_filt)

            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)


            projection = V_p[:k, :].T

            # %Scores are the magnitude of the projection onto the top principal component
            scores = np.mean(np.matmul(gcentered, projection), -1, keepdims=True)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))

            self.scores_draw[curindices] = scores

        self.p_def = self.args.ood_rate

        print('the length of the scores_draw is:', len(self.scores_draw))

        for i in labels:

            curIndices = allIndices[y == i]
            curGradients = gradients[curIndices, :]
            curMean = np.mean(gradients[curIndices, :],
                              axis=0, keepdims=True)  # (1, d)

            if not self.args.avg_on_wild:
                filterSimple(curGradients,
                         self.p_def,
                         self.curmean[int(i)], curIndices, int(i))
            else:
                filterSimple(curGradients,
                         self.p_def,
                         curMean, curIndices, int(i))
        return np.quantile(self.scores_draw, self.args.thres_id)


    def get_threshold(self, train_loader):
        '''
        Xdata and Ydata are numpy array.
        and do not split the train/val/test data
        '''
        for epoch in range(self.num_rounds):
            gradients, losses, targets = self.train_no_gradients(train_loader)
        thres = self.GetThresNoConditional(losses,
                                           targets,
                                           gradients)
        return thres


    def get_mask(self, X_aux_in, X_aux_out):
        '''
        Xdata and Ydata are numpy array.
        and do not split the train/val/test data
        '''


        self.logger.info("======> Defensing!")
        for epoch in range(self.num_rounds):

            self.logger.info("=====> round %d" % epoch)

            gradients, losses, targets, mask_all = self.train_no_gradients_mixture(X_aux_in,
                                                                         X_aux_out)
            self.logger.info("=====> current training losses %.4f" % (np.sum(losses) / len(losses)))

            self.logger.info("=====> filterSimple")
            gradients = gradients.squeeze()
            self.scores_draw = np.zeros(len(targets))

            indices, outlier_scores = self.filterByClassNoConditional(losses,
                                                         targets,
                                                         gradients)


            self.logger.info("=====> filtering %d samples" % (len(targets) - len(indices)))

            if len(indices) == len(targets) or len(indices) == 0:
                break

            active_indices = indices


        mask = np.asarray([-1 for i in range(len(targets))])# OOD
        mask[active_indices] = 1

        return mask, mask_all


    def train_no_gradients(self, X):
        index = 0
        for data, target in X:
            losses, gradients = self.per_sample_gradient(
                data.cuda(),
                target.cuda())
            if index == 0:
                target_all = target.cpu().numpy()
                losses_all = losses.detach().cpu().numpy()
                gradients_all = gradients.detach().cpu().numpy()
            else:
                target_all = np.concatenate([target_all,
                                             target.cpu().numpy()], 0)
                losses_all = np.concatenate([losses_all,
                                             losses.detach().cpu().numpy()], 0)
                gradients_all = np.concatenate([gradients_all,
                                                 gradients.detach().cpu().numpy()], 0)
            index += 1
        # breakpoint()
        return gradients_all, losses_all, target_all



    def train_no_gradients_mixture(self, X_in, X_out):
        index = 0
        batch_iterator = iter(X_in)
        for data, target in X_out:
            try:
                in_set = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(X_in)
                in_set = next(batch_iterator)

            aux_set = torch.cat([data, in_set[0]], 0)
            if index == 0:
                self.kept_data_all = aux_set
            else:
                self.kept_data_all = torch.cat([self.kept_data_all, aux_set], 0)

            mask = np.concatenate([np.ones(len(data)),
                                        np.zeros(len(in_set[0]))], 0)
            aux_set = aux_set.cuda()

            target = self.net(aux_set).argmax(-1)

            losses, gradients = self.per_sample_gradient(
                aux_set,
                target)

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
            if index % 10 == 0:
                print(index * 128)

        gradients_all = gradients_all.numpy()
        losses_all = losses_all.numpy()
        target_all = target_all.numpy()
        self.kept_data_all = self.kept_data_all.numpy()

        return gradients_all, losses_all, target_all, mask_all

    def per_sample_gradient(self, x, y):
        fmodel, params, buffers = make_functional_with_buffers(self.net)


        def compute_loss_stateless_model(params, buffers, sample, target):
            batch = sample.unsqueeze(0)
            targets = target.unsqueeze(0)

            predictions = fmodel(params, buffers, batch)
            loss = self.criterion(predictions, targets.long())
            return loss

        ft_compute_grad = compute_loss_stateless_model
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        # breakpoint()
        losses = ft_compute_sample_grad(params, buffers, x, y)



        ft_compute_grad = grad(compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x, y)
        # breakpoint()
        return losses, ft_per_sample_grads[-2]


    def cal_avg_clean(self, X):

        gradient_list = [np.zeros((0, self.args.num_class, 128))
                         for _ in range(self.args.num_class)]
        labels = np.arange(self.args.num_class)

        self.gradident_id = None

        for i in labels:
            for data, target in X:
                allIndices = np.arange(len(target))
                curIndices = allIndices[target == i]

                _, gradients = self.per_sample_gradient(
                    data[curIndices].cuda(),
                   target[curIndices].cuda())

                gradient_list[i] = np.concatenate([gradient_list[i],
                                                   gradients.detach().cpu().numpy()], 0)
                if self.gradident_id is None:
                    self.gradident_id = gradients[:, int(i), :].detach().cpu().numpy()
                else:
                    self.gradident_id = np.concatenate([self.gradident_id,
                                         gradients[:, int(i), :].detach().cpu().numpy()],0)
        for i in labels:
            gradient_list[i] = np.mean(gradient_list[i],
                                         axis=0, keepdims=True)[0]

        return gradient_list





def test(net, test_loader):
    net.eval()
    loss_avg = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)

            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    net.train()
    return correct / len(test_loader.dataset)


if __name__ == '__main__':
    '''
    define argument parser
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    # dataset related
    parser.add_argument('--aux_out_dataset', type=str, default='svhn',
                        choices=['svhn', 'lsun_c', 'lsun_r',
                            'isun', 'dtd', 'places',
                            'tinyimages_300k', 'cifar100'],
                        help='Auxiliary out of distribution dataset')
    parser.add_argument('--test_out_dataset', type=str,
                        choices=['svhn', 'lsun_c', 'lsun_r',
                                    'isun', 'dtd', 'places', 'tinyimages_300k', 'cifar100'],
                        default='svhn', help='Test out of distribution dataset')
    parser.add_argument('--pi', type=float, default=0.1,
                        help='pi in ssnd framework, proportion of ood data in auxiliary dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prefetch', type=int, default=4,
                        help='Pre-fetching threads.')

    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--woods_setting', type=int, default=1)
    parser.add_argument('--avg_on_wild', type=int, default=0)
    parser.add_argument('--use_thres', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=100)
    parser.add_argument('--thres_id', type=float, default=0.95)
    parser.add_argument('--draw_dis', type=int, default=0)
    parser.add_argument('--plot', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--ood_rate', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_binary_classifier', type=int, default=1)
    parser.add_argument('--load_full_classifier', type=int, default=1)
    parser.add_argument('--ft_epochs', type=int, default=100)
    parser.add_argument('--ft_weight', type=int, default=10)
    parser.add_argument('--loss_add', type=int, default=1)
    parser.add_argument('--num_sing_vectors', type=int, default=1)


    args = parser.parse_args()

    name = '{}_{}_{}_{}'.format(args.dataset, args.pi,
                                args.aux_out_dataset,
                                args.test_out_dataset)
    log_dir = 'log/{}'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = MyLog(os.path.join(log_dir, name + '.log'))
    logger.info(args)




    seed = args.seed
    device = 'cuda:0'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module

    state = {k: v for k, v in args._get_kwargs()}
    from make_dataset import make_datasets_my

    train_loader_in, train_loader_in_large_bs, train_loader_aux_in, \
    train_loader_aux_out, test_loader_in, \
    test_loader_out = make_datasets_my(
        args.dataset, args.aux_out_dataset,
        args.test_out_dataset, args.ood_rate, state)


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



    if args.woods_setting:
        se = SAL(args,
                   logger,
                   p_def=args.ood_rate,
                   num_rounds=args.num_rounds,
                   seed=args.seed)
        se.rng = np.random.default_rng(args.seed)
        if args.load_full_classifier:
            if args.num_class == 10:
                se.net.load_state_dict(torch.load('./pretrained/cifar10_wrn_pretrained_epoch_99.pt'))
            else:
                se.net.load_state_dict(torch.load(
                    './pretrained/cifar100_wrn_pretrained_epoch_99.pt'))
        else:
            se.net.load_state_dict(torch.load(args.dataset + '.pt'))
        se.net.eval()


        if not args.avg_on_wild:
            curmean = se.cal_avg_clean(train_loader_in_large_bs)
            se.curmean = curmean

        if args.use_thres:
            se.scores_draw = np.zeros(len(train_loader_in_large_bs.dataset))
            thres = se.get_threshold(train_loader_in)
            se.thres = thres
            print('thres is :', se.thres)


        mask, mask_gt = se.get_mask(train_loader_aux_in,
                                                  train_loader_aux_out)

        # breakpoint()
        # breakpoint()
        from sklearn.metrics import f1_score
        print('###########################')
        print('OOD detection precision: ', np.sum((mask==-1)[mask_gt == 1]) / (mask==-1).sum())
        print('OOD detection recall: ', np.sum((mask==-1)[mask_gt == 1]) / (mask_gt == 1).sum())
        print('f1 score is: ', f1_score(mask == -1, mask_gt))
        print('###########################')

    dist = torch.cdist(torch.from_numpy(se.gradident_id).unsqueeze(0), torch.from_numpy(se.gradident_wild).unsqueeze(0),
                       compute_mode='donot_use_mm_for_euclid_dist')
    print(dist.mean())



    import matplotlib.pyplot as plt

    if args.draw_dis:
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        import pandas as pd
        import seaborn as sns
        id_pd = pd.Series(se.scores_draw[mask_gt == False])
        ood_pd = pd.Series(se.scores_draw[mask_gt == True])

        p1 = sns.kdeplot(id_pd, shade=True, color="r", label='ID')
        p1 = sns.kdeplot(ood_pd, shade=True, color="b", label='OOD')
        plt.legend()
        plt.savefig('score_dis_cifar.jpg', dpi=250)


    if args.train_binary_classifier:
        loaded_data = se.kept_data_all[mask == -1]

        binary_classifier = WideResNet(40, args.num_class, 2, dropRate=0.3).cuda()
        logistic_regression = torch.nn.Sequential(
            torch.nn.Linear(128, 1))

        logistic_regression = logistic_regression.cuda().train()

        binary_classifier.train()

        binary_cls_optimizer = torch.optim.SGD(list(binary_classifier.parameters()) + list(logistic_regression.parameters()),
                                               momentum=0.9,
                                               nesterov=True,
                                         lr=0.001, weight_decay=args.weight_decay)
        if args.load_full_classifier:
            if args.num_class == 10:
                binary_classifier.load_state_dict(torch.load(
                    './pretrained/cifar10_wrn_pretrained_epoch_99.pt'))
            else:
                binary_classifier.load_state_dict(torch.load(
                    './pretrained/cifar100_wrn_pretrained_epoch_99.pt'))

        else:
            binary_classifier.load_state_dict(torch.load(args.dataset + '.pt'))
        ood_data_length = len(loaded_data)
        # print(ood_data_length)
        permutation_idx = torch.randperm(ood_data_length)
        batch_begin = 0
        criterion = torch.nn.CrossEntropyLoss()

        binary_scheduler = torch.optim.lr_scheduler.LambdaLR(
            binary_cls_optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.ft_epochs * len(train_loader_in),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / 0.001))


        loaded_data = torch.from_numpy(loaded_data).cuda()
        for epoch in range(args.ft_epochs):
            for in_set in train_loader_in:
                binary_cls_optimizer.zero_grad()
                if ood_data_length - batch_begin < args.batch_size:
                    batch_begin = 0
                    permutation_idx = torch.randperm(ood_data_length)
                # breakpoint()
                out, out_logits = binary_classifier.forward_my1(torch.cat([in_set[0].cuda(),
                                                       loaded_data[permutation_idx][batch_begin:batch_begin+args.batch_size]], 0))

                batch_begin += args.batch_size

                loss = F.cross_entropy(out_logits[:len(in_set[1])], in_set[1].cuda())


                output1 = logistic_regression(out)
                # breakpoint()
                binary_labels = torch.ones(len(in_set[1]) + args.batch_size).cuda()
                binary_labels[len(in_set[1]):] = 0


                energy_reg_loss = F.binary_cross_entropy_with_logits(output1.view(-1), binary_labels.float())

                if args.loss_add:
                    loss += args.ft_weight * energy_reg_loss
                else:
                    loss = args.ft_weight * energy_reg_loss

                loss.backward()
                binary_cls_optimizer.step()
                binary_scheduler.step()
            print('Epoch: ', epoch, 'Acc:', test(binary_classifier, test_loader_in))
            print('Loss: ', loss)


        binary_classifier.eval()
        logistic_regression.eval()


        def test_ood_function(test_loader_in):
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


        energy_id = test_ood_function(test_loader_in)
        energy_ood = test_ood_function(test_loader_out)

        from utils.metric_utils import get_measures, print_measures
        measures = get_measures(energy_id, energy_ood, plot=False)
        print_measures(measures[0], measures[1], measures[2], 'energy')







