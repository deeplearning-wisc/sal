import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from Log import MyLog
import random

from functorch import make_functional_with_buffers, vmap, grad


class MLP(nn.Module):
    def __init__(self, input_dim=2, class_num=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32,32)
        self.fc3 = nn.Linear(32, class_num)

    def forward(self, x):
        # breakpoint()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAL:


    def __init__(self,
                 args,
                 opt_method,
                 logger,
                 batch_size=32,
                 eta=0.01,
                 p_def=0.03,
                 num_epochs=3,
                 num_rounds=5,
                 epsilon=0.05,
                 seed=12345):
        super(SAL, self).__init__()

        assert opt_method in ['SGD', 'Adagrad'], 'check the opt method'

        self.opt_method = opt_method
        self.num_rounds = num_rounds
        self.epsilon = epsilon
        self.p_def = p_def
        self.args = args
        if not self.args.avg_on_wild:
            self.curmean = None

        if self.args.draw_dis:
            self.scores_draw = np.zeros(self.args.N_wild)
        if self.args.plot_gradient:
            self.gradient_vector = np.zeros((5500, 32))
            self.top_singular_vector = None#np.zeros((self.args.N_wild, 32))

        self.batch_size = batch_size
        self.eta = eta
        self.num_epochs = num_epochs
        self.net = MLP(class_num=self.args.num_class)
        self.net.train()

        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=self.eta, weight_decay=self.args.weight_decay)

        self.logger = logger
        self.criterion = torch.nn.CrossEntropyLoss()

        np.random.seed(seed)

    def attack_and_defense(self, Xdata, Ydata):
        '''
        Xdata and Ydata are numpy array.
        and do not split the train/val/test data
        '''

        k = 1
        N_train, d = Xdata.shape

        active_indices = np.arange(N_train)

        self.theta = np.random.uniform(size=(d, k))

        self.logger.info("======> Defensing!")
        for epoch in range(self.num_rounds):

            self.logger.info("=====> round %d" % epoch)
            xs = torch.from_numpy(Xdata[active_indices])
            ys = torch.from_numpy(Ydata[active_indices])

            gradients, losses = self.train_new(xs, ys)


            self.logger.info("=====> current training losses %.4f" % (torch.sum(losses) / len(active_indices)))

            self.logger.info("=====> filterSimple")
            gradients = gradients.squeeze()
            # breakpoint()
            indices, outlier_scores = self.filterByClass(losses.detach().numpy(),
                                                         ys.numpy(), gradients.detach().numpy())
            self.logger.info("=====> filtering %d samples" % (len(ys) - len(indices)))

            if len(indices) == len(active_indices) or len(indices) == 0:
                break

            active_indices = active_indices[indices]


        mask = np.asarray([-1 for i in range(N_train)])
        mask[active_indices] = 1
        return mask

    def filterByClass(self, losses, y, gradients):
        n = gradients.shape[0]

        allIndices = np.arange(n)
        labels = np.unique(y)

        def filterSimple(g, p, m, curindices):
            k = 1
            N_filt = g.shape[0]
            gcentered = (g - m) #/ np.sqrt(N_filt)

            # _, _, V_p = np.linalg.svd(gcentered)
            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)

            # print(_.shape)
            # projection = V_p[:, :k]
            projection = V_p[:k, :].T

            # %Scores are the magnitude of the projection onto the top principal component
            scores = np.matmul(gcentered, projection)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))
            if self.args.gradnorm:
                scores = np.linalg.norm(g, axis=-1)
            # breakpoint()
            if self.args.draw_dis:
                self.scores_draw[curindices] = scores
            # debug
            print(scores.shape)
            print(np.quantile(scores, 1 - p))

            indices = np.arange(N_filt)

            if np.quantile(scores, 1 - p) > 0:
                scores = scores / np.quantile(scores, 1 - p)
                indices = indices[scores <= 1.0]
            else:
                scores = scores / np.max(scores)

            # print(indices.shape)
            return indices, scores

        assert 0 in labels and 1 in labels

        n_minus = sum(y == 0)
        n_plus = sum(y == 1)
        re_scores = np.zeros(n)

        self.p_def = (n_plus + n_minus) * self.epsilon / (self.num_rounds * min(n_minus, n_plus))

        self.p_def = self.args.ood_rate
        print("self.p_def: ", self.p_def)
        re_indices = []
        for i in labels:

            curIndices = allIndices[y == i]

            # print(curIndices.shape)
            curGradients = gradients[curIndices, :]
            curMean = np.mean(gradients[curIndices, :], axis=0, keepdims=True)  # (1, d)

            if not self.args.avg_on_wild:
                curFilteredIndices, curScores = filterSimple(curGradients,
                                                             self.p_def,
                                                             self.curmean[int(i)], curIndices)
            else:
                curFilteredIndices, curScores = filterSimple(curGradients,
                                                             self.p_def,
                                                             curMean, curIndices)
            reindex = allIndices[curIndices]
            if len(curFilteredIndices):
                re_indices.extend(reindex[curFilteredIndices].tolist())

            re_scores[curIndices] = curScores

        re_indices.sort()

        return re_indices, re_scores

    def filterNoClass(self, losses, y, gradients):
        n = gradients.shape[0]

        allIndices = np.arange(n)
        labels = np.unique(y)

        def filterSimple(g, p, m, curindices):
            k = 1
            N_filt = g.shape[0]
            gcentered = (g - m) #/ np.sqrt(N_filt)

            # _, _, V_p = np.linalg.svd(gcentered)
            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)

            # print(_.shape)
            # projection = V_p[:, :k]
            projection = V_p[:k, :].T

            # %Scores are the magnitude of the projection onto the top principal component
            scores = np.matmul(gcentered, projection)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))
            if self.args.draw_dis:
                self.scores_draw[curindices] = scores
            # debug
            print(scores.shape)
            print(np.quantile(scores, 1 - p))

            indices = np.arange(N_filt)
            # breakpoint()
            if np.quantile(scores, 1 - p) > 0:
                scores = scores / np.quantile(scores, 1 - p)
                indices = indices[scores <= 1.0]
            else:
                scores = scores / np.max(scores)

            # print(indices.shape)
            return indices, scores

        assert 0 in labels and 1 in labels

        n_minus = sum(y == 0)
        n_plus = sum(y == 1)
        re_scores = np.zeros(n)

        self.p_def = (n_plus + n_minus) * self.epsilon / (self.num_rounds * min(n_minus, n_plus))

        self.p_def = self.args.ood_rate
        print("self.p_def: ", self.p_def)
        re_indices = []
        # for i in labels:

        curIndices = allIndices

        # print(curIndices.shape)
        curGradients = gradients[curIndices, :]
        curMean = np.mean(gradients[curIndices, :], axis=0, keepdims=True)  # (1, d)
        # breakpoint()
        if not self.args.avg_on_wild:
            curFilteredIndices, curScores = filterSimple(curGradients,
                                                         self.p_def,
                                                         self.curmean[0], curIndices)
        else:
            curFilteredIndices, curScores = filterSimple(curGradients,
                                                         self.p_def,
                                                         curMean, curIndices)
        reindex = allIndices[curIndices]
        if len(curFilteredIndices):
            re_indices.extend(reindex[curFilteredIndices].tolist())

        re_scores[curIndices] = curScores

        re_indices.sort()

        return re_indices, re_scores


    def filterByClassNoConditional(self, losses, y, gradients):
        n = gradients.shape[0]

        allIndices = np.arange(n)
        labels = np.unique(y)

        def filterSimple(g, p, m, curindices, class_index):
            k = 1
            N_filt = g.shape[0]
            # breakpoint()
            gcentered = (g - m)[:, class_index, :] #/ np.sqrt(N_filt)

            # _, _, V_p = np.linalg.svd(gcentered)
            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)

            # print(_.shape)
            # projection = V_p[:, :k]
            projection = V_p[:k, :].T

            # %Scores are the magnitude of the projection onto the top principal component
            scores = np.matmul(gcentered, projection)
            if self.args.plot_gradient and curindices[0] == 0:
                self.gradient_vector = gcentered
                self.top_singular_vector = V_p[:k, :]
                # breakpoint()
                # breakpoint()
            scores = np.sqrt(np.sum(np.square(scores), axis=1))
            if self.args.gradnorm:
                scores = np.linalg.norm(g, axis=-1)
            # breakpoint()
            if self.args.draw_dis:
                self.scores_draw[curindices] = scores


        assert 0 in labels and 1 in labels

        n_minus = sum(y == 0)
        n_plus = sum(y == 1)
        re_scores = np.zeros(n)

        self.p_def = (n_plus + n_minus) * \
                     self.epsilon / (self.num_rounds * min(n_minus, n_plus))

        self.p_def = self.args.ood_rate
        print("self.p_def: ", self.p_def)
        re_indices = []
        self.gradident_wild = None
        # self.index_save = None
        for i in labels:
            # breakpoint()
            curIndices = allIndices[y == i]

            # print(curIndices.shape)
            curGradients = gradients[curIndices, :]
            curMean = np.mean(gradients[curIndices, :],
                              axis=0, keepdims=True)  # (1, d)
            if self.gradident_wild is None:
                self.gradident_wild = curGradients[:, int(i), :]
                # self.index_save = np.nonzero(y == i)[0]

            else:

                self.gradident_wild = np.concatenate([self.gradident_wild,
                    curGradients[:, int(i), :]], 0)
                # self.index_save = np.concatenate([self.index_save,
                #                                   np.nonzero(y == i)[0]], 0)

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
        # breakpoint()
        if np.quantile(self.scores_draw, 1 - self.p_def) > 0:
            if self.args.use_thres:
                scores = self.scores_draw / se.thres#np.quantile(self.scores_draw, 1 - self.p_def)
            else:
                scores = self.scores_draw / np.quantile(self.scores_draw, 1 - self.p_def)
            indices = indices[scores <= 1.0]
        else:
            scores = self.scores_draw / np.max(self.scores_draw)
        re_indices = indices
        re_scores = scores
        re_indices.sort()
        # breakpoint()
        return re_indices, re_scores



    def GetThresNoConditional(self, losses, y, gradients):
        n = gradients.shape[0]

        allIndices = np.arange(n)
        labels = np.unique(y)

        def filterSimple(g, p, m, curindices, class_index):
            k = 1
            N_filt = g.shape[0]
            gcentered = (g - m)[:, class_index, :] #/ np.sqrt(N_filt)

            # _, _, V_p = np.linalg.svd(gcentered)
            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)

            # print(_.shape)
            # projection = V_p[:, :k]
            projection = V_p[:k, :].T

            # %Scores are the magnitude of the projection onto the top principal component
            scores = np.matmul(gcentered, projection)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))
            if self.args.gradnorm:
                scores = np.linalg.norm(g, axis=-1)
            # breakpoint()
            if self.args.draw_dis:
                self.scores_draw[curindices] = scores


        assert 0 in labels and 1 in labels

        n_minus = sum(y == 0)
        n_plus = sum(y == 1)
        re_scores = np.zeros(n)

        self.p_def = (n_plus + n_minus) * \
                     self.epsilon / (self.num_rounds * min(n_minus, n_plus))

        self.p_def = self.args.ood_rate

        re_indices = []
        for i in labels:

            curIndices = allIndices[y == i]

            # print(curIndices.shape)
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


    def get_threshold(self, Xdata, Ydata):
        '''
        Xdata and Ydata are numpy array.
        and do not split the train/val/test data
        '''

        k = 1
        N_train, d = Xdata.shape

        active_indices = np.arange(N_train)

        self.theta = np.random.uniform(size=(d, k))


        for epoch in range(self.num_rounds):


            xs = torch.from_numpy(Xdata[active_indices])
            ys = torch.from_numpy(Ydata[active_indices])

            gradients, losses = self.train_no_gradients(xs, ys)
            gradients = gradients.squeeze()
            # breakpoint()
            if self.args.filter_no_class:
                thres = self.filterNoClass(losses.detach().numpy(),
                                                             ys.numpy(), gradients.detach().numpy())
            else:
                if self.args.no_conditional:
                    thres = self.GetThresNoConditional(losses.detach().numpy(),
                                                                 ys.numpy(),
                                                                 gradients.detach().numpy())
                else:
                    thres = self.filterByClass(losses.detach().numpy(),
                                                                 ys.numpy(),
                                                                 gradients.detach().numpy())


        return thres


    def train_new(self, X_train, Y_train):
        N_train, d = X_train.shape
        ids = [i for i in range(N_train)]
        for epoch in range(self.num_epochs):
            np.random.shuffle(ids)
            for t in range(0, N_train, self.batch_size):
                t2 = min(t + self.batch_size, N_train)
                self.optimizer.zero_grad()
                Xb = X_train[ids[t:t2],].to(dtype=torch.float32)
                Yb = Y_train[ids[t:t2]].long()
                outb = self.net(Xb)
                # breakpoint()
                losses = self.criterion(outb, Yb)

                losses.backward()
                self.optimizer.step()
            losses = self.criterion(
                self.net(X_train.to(dtype=torch.float32)), Y_train.long())
            self.logger.info("Error (epoch %d): %.4f" % (epoch, losses))
        losses, gradients = self.per_sample_gradient(X_train.to(dtype=torch.float32), Y_train.float())
        # breakpoint()
        return gradients, losses


    def get_mask(self, Xdata, Ydata):
        '''
        Xdata and Ydata are numpy array.
        and do not split the train/val/test data
        '''

        k = 1
        N_train, d = Xdata.shape

        active_indices = np.arange(N_train)

        self.theta = np.random.uniform(size=(d, k))

        self.logger.info("======> Defensing!")
        for epoch in range(self.num_rounds):

            self.logger.info("=====> round %d" % epoch)
            xs = torch.from_numpy(Xdata[active_indices])
            ys = torch.from_numpy(Ydata[active_indices])

            gradients, losses = self.train_no_gradients(xs, ys)


            self.logger.info("=====> current training losses %.4f" % (torch.sum(losses) / len(active_indices)))

            self.logger.info("=====> filterSimple")
            gradients = gradients.squeeze()
            # breakpoint()
            if self.args.filter_no_class:
                indices, outlier_scores = self.filterNoClass(losses.detach().numpy(),
                                                             ys.numpy(), gradients.detach().numpy())
            else:
                if self.args.no_conditional:
                    indices, outlier_scores = self.filterByClassNoConditional(losses.detach().numpy(),
                                                                 ys.numpy(),
                                                                 gradients.detach().numpy())
                else:
                    indices, outlier_scores = self.filterByClass(losses.detach().numpy(),
                                                                 ys.numpy(),
                                                                 gradients.detach().numpy())

            self.logger.info("=====> filtering %d samples" % (len(ys) - len(indices)))

            if len(indices) == len(active_indices) or len(indices) == 0:
                break

            active_indices = active_indices[indices]


        mask = np.asarray([-1 for i in range(N_train)])
        mask[active_indices] = 1
        return mask


    def train_no_gradients(self, X_train, Y_train):

        losses, gradients = self.per_sample_gradient(X_train.to(dtype=torch.float32), Y_train.float())
        return gradients, losses


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


    def cal_avg_clean(self, X, Y):
        gradient_list = []
        labels = np.unique(Y)
        n = X.shape[0]
        self.gradident_id = None
        allIndices = np.arange(n)
        if self.args.filter_no_class:
            curIndices = allIndices
            _, gradients = self.per_sample_gradient(
                torch.from_numpy(X[curIndices]).float(),
                torch.from_numpy(Y[curIndices]))
            # breakpoint()
            gradient_list.append(np.mean(gradients.detach().numpy(),
                                         axis=0, keepdims=True)[0])
        else:
            for i in labels:

                curIndices = allIndices[Y == i]
                _, gradients = self.per_sample_gradient(
                    torch.from_numpy(X[curIndices]).float(),
                    torch.from_numpy(Y[curIndices]))

                if self.gradident_id is None:
                    self.gradident_id = gradients[:, int(i), :].detach().numpy()
                else:
                    self.gradident_id = np.concatenate([self.gradident_id,
                                         gradients[:, int(i), :].detach().numpy()],0)

                gradient_list.append(np.mean(gradients.detach().numpy(),
                                             axis=0, keepdims=True)[0])
        # breakpoint()
        return gradient_list



if __name__ == '__main__':
    '''
    define argument parser
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='toy', choices=['toy', 'age', 'lfw', 'food'])
    parser.add_argument('--batch_size', type=int, default=32)

    # deep model params
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--add_outliers', type=int, default=0)
    parser.add_argument('--woods_setting', type=int, default=1)
    parser.add_argument('--avg_on_wild', type=int, default=0)
    parser.add_argument('--circle_ood', type=int, default=1)
    parser.add_argument('--use_thres', type=int, default=1)
    parser.add_argument('--filter_no_class', type=int, default=0)
    parser.add_argument('--no_conditional', type=int, default=1)
    parser.add_argument('--num_class', type=int, default=3)

    parser.add_argument('--thres_id', type=float, default=0.95)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--draw_dis', type=int, default=1)
    parser.add_argument('--plot', type=int, default=1)
    parser.add_argument('--plot_gradient', type=int, default=0)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.4)
    parser.add_argument('--ood_rate', type=float, default=0.1)
    parser.add_argument('--gradnorm', type=int, default=0)
    parser.add_argument('--N_id', type=int, default=1000)
    parser.add_argument('--N_wild', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()


    #python toy_data.py  --N_id 1000  --circle_ood 1 --no_conditional 1 --use_thres 0  --num_epochs 3


    log_dir = 'log/{}'.format(args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = MyLog(os.path.join(log_dir, '_trial' + '.log'))
    logger.info(args)




    seed = args.seed
    device = 'cuda:0'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module




    # debug the defense model
    N, d = args.N_id, 2
    N_wild = args.N_wild

    variance = 0.25
    cov = np.diag([variance, variance])
    data_1 = np.random.multivariate_normal([0, 2 *  np.sqrt(3)], cov, N)
    data_2 = np.random.multivariate_normal([-2, 0], cov, N)
    data_3 = np.random.multivariate_normal([2, 0], cov, N)
    X = np.concatenate([data_1, data_2], 0)
    X = np.concatenate([X, data_3], 0)
    Y = np.concatenate([np.zeros(N), np.ones(N)], 0)
    Y = np.concatenate([Y, np.ones(N) * 2], 0)

    if args.add_outliers:
        X_outlier = np.random.multivariate_normal([10, 0], cov, 100)
        X = np.concatenate([X, X_outlier], 0)
        Y = np.concatenate([Y, np.zeros(100)], 0)


    if args.woods_setting:
        se = SAL(args,
                   args.optim,
                   logger,
                   epsilon=args.epsilon,
                   eta=args.lr,
                   batch_size=args.batch_size,
                   num_rounds=args.num_rounds,
                   num_epochs=args.num_epochs,
                   seed=args.seed)
        logger.info("=====> total samples: %d" % Y.shape[0])
        se.train_new(torch.from_numpy(X), torch.from_numpy(Y))
        if not args.avg_on_wild:
            curmean = se.cal_avg_clean(X, Y)
            se.curmean = curmean

        ood_num = int(N_wild * args.ood_rate)
        id_num = int((args.N_wild - ood_num) / args.num_class)
        if args.circle_ood:
            X_mixture = np.random.multivariate_normal([0, 2 / np.sqrt(3)],
                                                      np.diag([7, 7]),
                                                      ood_num * 100)
            norm = np.linalg.norm(X_mixture - np.array([0, 2 / np.sqrt(3)]), axis=-1)
            ind = np.argpartition(norm, -ood_num)[-ood_num:]
            X_mixture = X_mixture[ind]
        else:
            X_mixture = np.random.multivariate_normal([10, 2 / np.sqrt(3)],
                                                      cov,
                                                      ood_num)
        # breakpoint()
        X_mixture = np.concatenate([X_mixture,
                                    np.random.multivariate_normal([-2, 0], cov, id_num)],0)
        X_mixture = np.concatenate([X_mixture,
                                    np.random.multivariate_normal([2, 0], cov, id_num)], 0)
        X_mixture = np.concatenate([X_mixture,
                                    np.random.multivariate_normal([0, 2 *  np.sqrt(3)], cov, id_num)])


        # generate pseudo labels.
        Y_mixture = (F.softmax(
            se.net(torch.from_numpy(X_mixture).float())).argmax(-1)).float().view(-1)


        if args.use_thres:
            se.scores_draw = np.zeros(args.num_class * args.N_id)
            thres = se.get_threshold(X, Y)
            se.thres = thres
            print('thres is :', se.thres)
        if args.draw_dis:
            se.scores_draw = np.zeros(id_num * args.num_class + ood_num)
        mask = se.get_mask(X_mixture, Y_mixture.numpy())

        X = X_mixture
        Y = Y_mixture.numpy()
        # breakpoint()
        Y1 = np.concatenate([np.ones(1000), np.zeros(9000)], 0)
        assert len(mask) == len(Y), "check the filtering length!"
        from sklearn.metrics import f1_score
        print('###########################')
        print('OOD detection acc: ', np.sum((mask==-1)[:ood_num]) / (mask==-1).sum())
        print('OOD detection rate: ', np.sum((mask == -1)[:ood_num]) / ood_num)
        print('f1 score is: ', f1_score(mask == -1, np.concatenate([np.ones(ood_num), np.zeros(len(Y)-ood_num)])))
        print('###########################')




    n_plus = sum((Y == 1) & (mask == 1))
    n_minus = sum((Y == 0) & (mask == 1))
    n_third_class = sum((Y == 2) & (mask == 1))

    logger.info('=====> after defense, info:')
    logger.info('=====> first class: %d' % n_plus)
    logger.info('=====> second class: %d' % n_minus)
    logger.info('=====> third class: %d' % n_third_class)
    logger.info('=====> total pair: %d' % (n_plus + n_minus + n_third_class))
    print('value 1: ', np.sum((mask == -1)[ood_num:]) / (id_num * args.num_class))
    print('value 2: ', np.sum((mask == 1)[:ood_num]) / (ood_num))

    dist = torch.cdist(torch.from_numpy(se.gradident_id).unsqueeze(0),
    torch.from_numpy(se.gradident_wild)[
                         np.random.uniform(0, 9998, 3000)].unsqueeze(0),
    compute_mode = 'donot_use_mm_for_euclid_dist'   )
    print(dist.mean())
    # breakpoint()
    if args.plot:
        import matplotlib.pyplot as plt


        fig, ax = plt.subplots(figsize=(13, 6))
        plt.scatter(X[:, 0][(Y == 1) & (mask == 1) & (Y1 == 0) ],
                    X[:, 1][(Y == 1) & (mask == 1)& (Y1 == 0)],
                    c='#d6c7c0',label=r'$\mathcal{S}_{\rm wild}^{\rm in}$ class 1')

        plt.scatter(X[:, 0][(Y == 0) & (mask == 1)& (Y1 == 0)],
                    X[:, 1][(Y == 0) & (mask == 1)& (Y1 == 0)],
                    c='#e29c7a',label=r'$\mathcal{S}_{\rm wild}^{\rm in}$ class 2')

        plt.scatter(X[:, 0][(Y == 2) & (mask == 1)& (Y1 == 0)],
                    X[:, 1][(Y == 2) & (mask == 1)& (Y1 == 0)],
                    c='#61280e',label=r'$\mathcal{S}_{\rm wild}^{\rm in}$ class 3')

        plt.scatter(X[:, 0][(Y1 == 1)],
                    X[:, 1][ (Y1 == 1)],
                    c='#7373a2',label=r'$\mathcal{S}_{\rm wild}^{\rm out}$')

        plt.scatter(X[:, 0][mask == -1],
                    X[:, 1][mask == -1],
                    c='green', label=r'$\mathcal{S}_{T}$')
        plt.legend(fontsize=20)

        plt.setp(ax, xticks=[], yticks=[])
        plt.savefig('./visual_data.jpg', dpi=250)

        if args.draw_dis:
            plt.clf()
            import matplotlib
            matplotlib.rcParams.update({'font.size': 22})
            fig, ax = plt.subplots(figsize=(13,6))
            import pandas as pd
            import seaborn as sns
            id_pd = pd.Series(se.scores_draw[ood_num:])
            ood_pd = pd.Series(se.scores_draw[:ood_num])
            # ood_pd.rename('OOD')
            # breakpoint()
            p1 = sns.kdeplot(id_pd, shade=True, color="#e29c7a", label='ID')
            p1 = sns.kdeplot(ood_pd, shade=True, color="#7373a2", label='OOD')
            plt.legend()
            plt.savefig('visual_score.jpg', dpi=250)

