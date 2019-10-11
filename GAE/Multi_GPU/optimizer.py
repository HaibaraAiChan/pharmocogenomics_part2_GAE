import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import torch.nn as nn
from utils import correlation_coefficient_tensor, cosine_similarity


"""
function: cross_entropy
assuming pred and soft_targets are both Variables with shape (batchsize, num_of_classes), 
each row of pred is predicted logits and each row of soft_targets is a discrete distribution.
"""


def cross_entropy(preds, targets):
    logsoftmax = nn.LogSoftmax()
    tmp1 = logsoftmax(preds)
    tmp2 = - targets * logsoftmax(preds)
    tmp3 = torch.sum(- targets * logsoftmax(preds), 1)
    tmp4 = torch.mean(torch.sum(- targets * logsoftmax(preds), 1))
    return torch.mean(torch.sum(- targets * logsoftmax(preds), 1))


"""
matrix-similarity based loss functions
"""


def similarity_cross_entropy(preds, targets):
    logsoftmax = nn.LogSoftmax()
    r = correlation_coefficient_tensor(preds, targets)
    loss = torch.mean(torch.sum(- logsoftmax(preds) * r, 1))
    return loss


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):

    # cost = norm * cross_entropy(preds, labels)
    cost = norm * similarity_cross_entropy(preds, labels)
    # cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD
