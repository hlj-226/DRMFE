import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

from fsl_loss import MMI
import evaluation
from util import next_batch
from fca_loss import MMD
from torch.nn import init
import sys

def to_mean_loss(features, laplacian):
    """
    Compute the loss term that compares features of adjacent nodes.
    """
    return torch.matmul(features.t().unsqueeze(1), laplacian.matmul(features).t().unsqueeze(2)).view(-1)
class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True,
                 d_model=32,
                 nhead=2,
                 dim_feedforward=128,
                 num_layers=2,
                 ):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)
        self.tran_en = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.extran_en = nn.TransformerEncoder(self.tran_en, num_layers=num_layers)


    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, feature Z^v.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, feature Z^v.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction samples.
        """
        latent=latent.unsqueeze(1)
        latent=self.extran_en(latent)
        latent=latent.squeeze(1)
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, feature Z^v.
              x_hat:  [num, feat_dim] float tensor, reconstruction samples.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

class GMRFLoss(nn.Module):
    """
    Implementation of the GMRF loss.
    """

    def __init__(self, beta=1):
        """
        Class initializer.
        """
        super().__init__()

        self.cached_adj = None
        self.beta = beta
        # self.lambda_reg = lambda_reg

    def forward(self, features, edge_index,lambda6):
        """
        Run forward propagation.
        """
        if self.cached_adj is None:
            self.cached_adj = edge_index

        num_nodes = features.size(0)
        hidden_dim = features.size(1)
        eye = torch.eye(hidden_dim, device=features.device)
        l1 = (eye + features.t().matmul(features) / self.beta).logdet()
        l2 = to_mean_loss(features, self.cached_adj).sum()
        MSE_loss = (l2 - l1 / lambda6) / num_nodes

        # L2 regularization
        l2_reg = torch.norm(features, p=2)
        return MSE_loss + 0.001 * l2_reg
class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent

class Drmfe():
    """Drmfe module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']

        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'], config['Autoencoder']['d_model'],config['Autoencoder']['nhead'],
                                        config['Autoencoder']['dim_feedforward'],config['Autoencoder']['num_layers'],)
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'], config['Autoencoder']['d_model'],config['Autoencoder']['nhead'],
                                        config['Autoencoder']['dim_feedforward'],config['Autoencoder']['num_layers'],)
        self.kld_loss = GMRFLoss(0.1)
        self.lambda6 = config['training']['lambda6']

        #  predictions.
        # To illustrate easily, we use "img" and "txt" to denote two different views.
        self.img2txt = Prediction(self._dims_view1)
        self.txt2img = Prediction(self._dims_view2)

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)
        self.img2txt.to(device)
        self.txt2img.to(device)

    def pretrain(self, config, x1_train, x2_train, optimizer, device):
        """预训练自编码器和跨视图预测器"""
        pretrain_epochs = config['pretraining']['epochs']
        batch_size = config['pretraining']['batch_size']

        # 确保输入数据是 PyTorch 张量并在正确的设备上
        x1_train = torch.from_numpy(x1_train).float().to(device)
        x2_train = torch.from_numpy(x2_train).float().to(device)

        for epoch in range(pretrain_epochs):
            X1, X2 = shuffle(x1_train, x2_train)
            loss_ae1, loss_ae2, loss_cross = 0, 0, 0

            for batch_x1, batch_x2, _, _, _ in next_batch(X1, X2, X1, X2, batch_size):
                # 自编码器预训练
                x1_recon, z1 = self.autoencoder1(batch_x1)
                x2_recon, z2 = self.autoencoder2(batch_x2)
                loss_ae1 = F.mse_loss(x1_recon, batch_x1)
                loss_ae2 = F.mse_loss(x2_recon, batch_x2)
                # 总损失
                loss = loss_ae1 + loss_ae2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Pretraining Epoch [{epoch + 1}/{pretrain_epochs}], "
                      f"AE1 Loss: {loss_ae1.item():.4f}, "
                      f"AE2 Loss: {loss_ae2.item():.4f}, ")

        return loss.item()  # 返回最后一个 batch 的总损失

    def train(self, config, logger, x1_train, x2_train, Y_list, mask, optimizer, device):
        """Training the model.
            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari
        """

        # Get complete data for training
        flag_1 = (torch.LongTensor([1, 1]).to(device) == mask).int()
        Y_list = torch.tensor(Y_list).int().to(device).squeeze(dim=0).unsqueeze(dim=1)
        Tmp_acc, Tmp_nmi, Tmp_ari = 0, 0, 0
        for epoch in range(config['training']['epoch'] + 1):
            X1, X2, X3, X4 = shuffle(x1_train, x2_train, flag_1[:, 0], flag_1[:, 1])
            loss_all, loss_rec1, loss_rec2, loss_fca, loss_fsl,loss_rec5, loss_rec6, loss_g1, loss_g2 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, x1_index, x2_index, batch_No in next_batch(X1, X2, X3, X4, config['training']['batch_size']):
                if len(batch_x1) < 2:
                    continue  # 跳过大小为1的批次
                def knn_graph(x, k):
                    # 计算pairwise距离
                    dist = torch.cdist(x, x)
                    # 获取最近的k个邻居（不包括自身）
                    _, indices = dist.topk(k + 1, largest=False)
                    indices = indices[:, 1:]  # 去掉自身
                    # 创建邻接矩阵
                    adj = torch.zeros(x.size(0), x.size(0), device=x.device)
                    adj.scatter_(1, indices, 1)
                    # 确保对称性
                    adj = adj + adj.t()
                    adj = (adj > 0).float()
                    # 添加自环
                    adj.fill_diagonal_(1)
                    return adj

                # 在训练循环中
                z_half1 = self.autoencoder1.encoder(batch_x1)
                z_half2 = self.autoencoder2.encoder(batch_x2)

                # 构建k近邻图
                A1 = knn_graph(z_half1, k=7)
                A2 = knn_graph(z_half2, k=7)
                # GMRFLoss
                loss_MSE1 = self.kld_loss(z_half1, A1, self.lambda6)
                loss_MSE2 = self.kld_loss(z_half2, A2, self.lambda6)
                loss_MSE = (loss_MSE1* + 0.01 * loss_MSE2) / 2


                index_both = x1_index + x2_index == 2                      # C in indicator matrix A of complete multi-view data
                index_peculiar1 = (x1_index + x1_index + x2_index == 2)    # I^1 in indicator matrix A of incomplete multi-view data
                index_peculiar2 = (x1_index + x2_index + x2_index == 2)    # I^2 in indicator matrix A of incomplete multi-view data
                z_1 = self.autoencoder1.encoder(batch_x1[x1_index == 1])   # [Z_C^1;Z_I^1]
                z_2 = self.autoencoder2.encoder(batch_x2[x2_index == 1])   # [Z_C^2;Z_I^2]

                recon1 = F.mse_loss(self.autoencoder1.decoder(z_1), batch_x1[x1_index == 1])
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_2), batch_x2[x2_index == 1])
                rec_loss = (recon1 + recon2)                               # reconstruction losses \sum L_REC^v

                z_view1_both = self.autoencoder1.encoder(batch_x1[index_both])
                z_view2_both = self.autoencoder2.encoder(batch_x2[index_both])

                if len(batch_x2[index_peculiar2]) % config['training']['batch_size'] == 1:
                    continue
                z_view2_peculiar = self.autoencoder2.encoder(batch_x2[index_peculiar2])
                if len(batch_x1[index_peculiar1]) % config['training']['batch_size'] == 1:
                    continue
                z_view1_peculiar = self.autoencoder1.encoder(batch_x1[index_peculiar1])

                img2txt1, _ = self.img2txt(z_half1)
                txt2img1, _ = self.txt2img(z_half2)
                recon5 = F.mse_loss(img2txt1, z_half2)
                recon6 = F.mse_loss(txt2img1, z_half1)
                kl_reg = config['training']['lambda5']*F.kl_div(F.log_softmax(img2txt1,dim=1),
                                       F.softmax(z_half2,dim=1),
                                       reduction='batchmean')
                PRE_loss = (recon5 + recon6) + kl_reg

                w1 = torch.var(z_view1_both)
                w2 = torch.var(z_view2_both)
                a1 = w1 / (w1 + w2)
                a2 = 1 - a1
                # the weight matrix is only used in MMI loss to explore the common cluster information
                # z_i = \sum a_iv w_iv z_i^v, here, w_iv = var(Z^v)/(\sum a_iv var(Z^v)) for MMI loss
                Z = torch.add(z_view1_both * a1, z_view2_both * a2)
                # mutual information losses \sum L_MMI^v (Z_C, Z_I^v)
                FSL_loss = MMI(z_view1_both, Z) + MMI(z_view2_both, Z)

                view1 = torch.cat([z_view1_both, z_view1_peculiar, z_view2_peculiar], dim=0)
                view2 = torch.cat([z_view2_both, z_view1_peculiar, z_view2_peculiar], dim=0)
                # z_i = \sum a_iv w_iv z_i^v, here, w_iv = 1/\sum a_iv for MMD loss
                view_both = torch.add(view1, view2).div(2)
                # mean discrepancy losses   \sum L_MMD^v (Z_C, Z_I^v)
                FCA_loss = MMD(view1, view_both, kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num']) + \
                           MMD(view2, view_both, kernel_mul=config['training']['kernel_mul'], kernel_num=config['training']['kernel_num'])

                # total loss
                loss = FSL_loss + FCA_loss * config['training']['lambda1'] + rec_loss * config['training']['lambda2']+ PRE_loss*config['training']['lambda4']+loss_MSE*config['training']['lambda3']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_rec5 += recon5.item()
                loss_rec6 += recon6.item()
                loss_g1 += loss_MSE1.item()
                loss_g2 += loss_MSE2.item()
                loss_fsl += FSL_loss.item()
                loss_fca += FCA_loss.item()

            if (epoch) % config['print_num'] == 0:
                output = "Epoch: {:.0f}/{:.0f} " \
                    .format(epoch, config['training']['epoch'])
                print(output)
                # evalution
                scores = self.evaluation(config, logger, mask, x1_train, x2_train, Y_list, device)
                # print(scores)

                if scores['kmeans']['ACC'] >= Tmp_acc:
                    Tmp_acc = scores['kmeans']['ACC']
                    Tmp_nmi = scores['kmeans']['NMI']
                    Tmp_ari = scores['kmeans']['ARI']
        return Tmp_acc, Tmp_nmi, Tmp_ari

    def evaluation(self, config, logger, mask, x1_train, x2_train, Y_list, device):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()
            
            flag = mask[:, 0] + mask[:, 1] == 2           # complete multi-view data
            view2_missing_idx_eval = mask[:, 0] == 0      # incomplete multi-view data
            view1_missing_idx_eval = mask[:, 1] == 0      # incomplete multi-view data

            common_view1 = x1_train[flag]
            common_view1 = self.autoencoder1.encoder(common_view1)
            common_view2 = x2_train[flag]
            common_view2 = self.autoencoder2.encoder(common_view2)
            y_common = Y_list[flag]

            view1_exist = x1_train[view1_missing_idx_eval]
            view1_exist = self.autoencoder1.encoder(view1_exist)
            y_view1_exist = Y_list[view1_missing_idx_eval]
            view2_exist = x2_train[view2_missing_idx_eval]
            view2_exist = self.autoencoder2.encoder(view2_exist)
            y_view2_exist = Y_list[view2_missing_idx_eval]
            common = torch.add(common_view1, common_view2).div(2)

            latent_fusion = torch.cat([common, view1_exist, view2_exist], dim=0).cpu().detach().numpy()
            Y_list = torch.cat([y_common, y_view1_exist, y_view2_exist], dim=0).cpu().detach().numpy()

            scores, _ = evaluation.clustering([latent_fusion], Y_list[:, 0])
            self.autoencoder1.train(), self.autoencoder2.train()
            
        return scores
