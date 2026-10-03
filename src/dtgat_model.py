"""DTGAT model definition and early-stopping helper."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dtgat_layers import AttentionLayer


class DTGAT(nn.Module):


    def __init__(self, input_dim, hidden1_dim=16, hidden2_dim=16, hidden3_dim=8,
                 hidden4_dim=8, output_dim=16, num_head1=3, num_head2=3, num_head3=3,
                 alpha=0.2, type='MLP', reduction='mean'):
        super().__init__()
        self.logit_tau = nn.Parameter(torch.tensor(5.0))

        if reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
            self.hidden3_dim = hidden3_dim
        elif reduction == 'concate':
            self.hidden1_dim = num_head1 * hidden1_dim
            self.hidden2_dim = num_head2 * hidden2_dim
            self.hidden3_dim = num_head3 * hidden3_dim

        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.num_head3 = num_head3
        self.alpha = alpha
        self.reduction = reduction
        self.type = type

        self.ConvLayer1 = [AttentionLayer(input_dim, hidden1_dim, alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i), attention)

        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim, hidden2_dim, alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i), attention)

        self.ConvLayer3 = [AttentionLayer(self.hidden2_dim, hidden3_dim, alpha) for _ in range(num_head3)]
        for i, attention in enumerate(self.ConvLayer3):
            self.add_module('ConvLayer3_AttentionHead{}'.format(i), attention)

        self.tf_linear1 = nn.Linear(hidden3_dim, hidden4_dim)
        self.target_linear1 = nn.Linear(hidden3_dim, hidden4_dim)
        self.tf_linear2 = nn.Linear(hidden4_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden4_dim, output_dim)

        if type == 'MLP':
            self.link_decoder = nn.Sequential(
                nn.Linear(2 * output_dim, 16),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(16, 1)
            )
        elif type == 'b_dot':
            self.bias_weight = nn.Parameter(torch.FloatTensor(output_dim, output_dim))
            nn.init.xavier_uniform_(self.bias_weight.data, gain=1.414)

        self.recon_network = nn.Sequential(
            nn.Linear(hidden3_dim, hidden2_dim),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.1),
            nn.Linear(hidden2_dim, hidden1_dim),
            nn.LeakyReLU(alpha),
            nn.Dropout(0.1),
            nn.Linear(hidden1_dim, input_dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()
        for attention in self.ConvLayer2:
            attention.reset_parameters()
        for attention in self.ConvLayer3:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

    def encode(self, x, adj):

        if self.reduction == 'concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1)
        elif self.reduction == 'mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0)
        x = F.elu(x)

        if self.reduction == 'concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer2], dim=1)
        elif self.reduction == 'mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]), dim=0)
        x = F.elu(x)

        out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer3]), dim=0)

        return out

    def link_predict(self, embed, edge_index):

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01, training=self.training)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01, training=self.training)

        target_embed = self.target_linear1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01, training=self.training)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01, training=self.training)

        train_tf = tf_embed[edge_index[:, 0].long()]
        train_target = target_embed[edge_index[:, 1].long()]

        if self.type == 'dot':
            pred = torch.sum(train_tf * train_target, dim=1).view(-1, 1)
        elif self.type == 'cosine':
            pred = F.cosine_similarity(train_tf, train_target, dim=1).view(-1, 1)
            pred = pred * self.logit_tau
        elif self.type == 'MLP':
            h = torch.cat([train_tf, train_target], dim=1)
            pred = self.link_decoder(h)
        elif self.type == 'b_dot':
            pred = torch.sum(torch.matmul(train_tf, self.bias_weight) * train_target, dim=1).view(-1, 1)

        return pred.squeeze()

    def expression_reconstruct(self, embed):
        return self.recon_network(embed)

    def forward(self, x, adj, edge_index):

        embed = self.encode(x, adj)

        link_pred = self.link_predict(embed, edge_index)

        recon_x = self.expression_reconstruct(embed)

        return link_pred, recon_x, embed


class EarlyStopping:
    def __init__(self, patience=80, delta=1e-5, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best = None
        self.cnt = 0
        self.stop = False
        self.best_state = None

    def __call__(self, val_metric, model):
        if self.best is None:
            self.best = val_metric
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            if self.mode == 'min':
                improved = val_metric < self.best - self.delta
            else:
                improved = val_metric > self.best + self.delta
            if improved:
                self.best = val_metric
                self.cnt = 0
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                self.cnt += 1
                if self.cnt >= self.patience:
                    self.stop = True

