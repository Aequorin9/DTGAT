"""Graph-attention layers used by DTGAT.

The attention-layer design in this module is adapted from the GENELink
implementation by the zpliulab contributors:
https://github.com/zpliulab/GENELink

Compared with the original dense implementation, this version operates only
on observed sparse edges and adds residual connections. GENELink is declared
by its authors to be available under the MIT License. See the repository-level
THIRD_PARTY_NOTICES.md for details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Sparse graph-attention layer with normalization and a residual path."""

    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True, add_residual=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.weight = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        # Retained for checkpoint compatibility with the development model.
        # It is not used in the current forward computation.
        self.weight_interact = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))

        self.add_residual = add_residual
        if add_residual and input_dim != output_dim:
            self.residual_linear = nn.Linear(input_dim, output_dim)
        else:
            self.residual_linear = None

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        h = torch.matmul(x, self.weight)
        h = F.leaky_relu(h, self.alpha)

        adj = adj.coalesce()
        edge_index = adj.indices()
        row = edge_index[0]
        col = edge_index[1]

        edge_scores = self._prepare_attentional_mechanism_input(h, row, col)
        attention = torch.sparse_coo_tensor(
            edge_index,
            edge_scores,
            size=adj.shape,
            device=h.device,
            dtype=h.dtype,
        ).coalesce()

        attention = torch.sparse.softmax(attention, dim=1).coalesce()
        attention_values = F.dropout(
            attention.values(), p=0.5, training=self.training
        )
        attention = torch.sparse_coo_tensor(
            attention.indices(),
            attention_values,
            size=adj.shape,
            device=h.device,
            dtype=h.dtype,
        ).coalesce()

        h_prime = torch.sparse.mm(attention, h)
        h_prime = F.normalize(h_prime, p=2, dim=1)

        if self.bias is not None:
            h_prime = h_prime + self.bias

        if self.add_residual:
            residual = self.residual_linear(x) if self.residual_linear is not None else x
            h_prime = h_prime + residual

        return h_prime

    def _prepare_attentional_mechanism_input(self, transformed_x, row, col):
        """Compute attention logits only for edges present in the sparse graph."""
        source_scores = torch.matmul(
            transformed_x, self.a[:self.output_dim, :]
        )
        target_scores = torch.matmul(
            transformed_x, self.a[self.output_dim:, :]
        )
        edge_scores = F.leaky_relu(
            source_scores[row] + target_scores[col],
            negative_slope=self.alpha,
        )
        return edge_scores.squeeze(-1)
