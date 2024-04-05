import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self):
        super(SemiLoss, self).__init__()

    def linear_rampup(self, lambda_u, current, warm_up, rampup_length=16):
        current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
        return lambda_u * float(current)

    def forward(self, outputs_x, targets_x, outputs_u, targets_u, lambda_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.linear_rampup(lambda_u, epoch, warm_up)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class InfoNCELoss(nn.Module):
    def __init__(self, temperature, batch_size, world_size=1, flat=False, n_views=2):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.world_size = world_size
        self.flat = flat
        self.n_views = n_views

    def forward(self, features):
        labels = torch.cat([torch.arange(self.batch_size * self.world_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.n_views * self.batch_size * self.world_size, self.n_views * self.batch_size * self.world_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        if self.flat:
            logits = (negatives - torch.sum(positives, dim=1).view(labels.shape[0], -1)) / self.temperature
            labels = torch.zeros_like(logits).to(features.device)
            labels[:, :self.n_views - 1] = 1
        else:
            logits = torch.cat([positives, negatives], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)
        return logits, labels


class PLRLoss(nn.Module):
    def __init__(self, temperature=1, base_temperature=1, flat=False):
        super(PLRLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.flat = flat

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        Args:
            features: shape of [bsz, n_views, ...]
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz],
                mask_{i,j}=1 if sample j has the same class as sample i,
                mask_{i,j}=-1 if sample j has different classes with sample i,
                mask_{i,j}=0 if sample j is not used for contrast.
                Can be asymmetric.
        Returns:
            A loss scalar.
        """

        def _get_loss(_mask, _anchor_dot_contrast):
            # create negative_mask from mask where mask_{i,j}==-1
            negative_mask = torch.eq(_mask, -1).float().to(device)
            # create positive_mask from mask where mask_{i,j}==1
            positive_mask = torch.eq(_mask, 1).float().to(device)
            # create neglect_mask where mask_{i,j}==0
            neglect_mask = torch.eq(_mask, 0).float().to(device)
            non_neglect_mask = 1 - neglect_mask

            # compute logits for non-neglect cases
            _anchor_dot_contrast *= non_neglect_mask

            if self.flat:
                # follow FlatNCE in https://arxiv.org/abs/2107.01152
                # filter out no negative case samples, which lead to nan loss
                has_negative = torch.nonzero(negative_mask.sum(1)).squeeze(1)
                negative_mask = negative_mask[has_negative]
                logits = (_anchor_dot_contrast - torch.sum(_anchor_dot_contrast * positive_mask, dim=1,
                                                           keepdim=True)) / self.temperature
                logits = logits[has_negative]

                exp_logits = torch.exp(logits) * negative_mask
                v = torch.log(exp_logits.sum(1, keepdim=True))
                loss_implicit = torch.exp(v - v.detach())  # all equal to 1

                # loss_explicit = - torch.log(1 / (1 + torch.sum(exp_logits, dim=1, keepdim=True)))  # just for watching
                # loss = loss_implicit.mean() - 1 + loss_explicit.mean().detach()
                _loss = loss_implicit.mean()
            else:
                # compute logits for non-neglect cases
                _anchor_dot_contrast = torch.div(_anchor_dot_contrast, self.temperature)
                # for numerical stability
                logits_max, _ = torch.max(_anchor_dot_contrast, dim=1, keepdim=True)
                logits = _anchor_dot_contrast - logits_max.detach()

                # compute log_prob
                exp_logits = torch.exp(logits) * non_neglect_mask
                log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

                # compute mean of log-likelihood over positive
                mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)

                # loss
                _loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
                _loss = _loss.mean()
            return _loss

        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            mask = torch.where(mask == 0, -1, mask)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
            mask = torch.where(mask == 0, -1, mask)
        else:
            pass

        features = F.normalize(features, dim=-1)
        # noise-tolerant contrastive loss
        anchor_count = contrast_count = features.shape[1]
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        mask = mask * (1 - torch.eye(batch_size * anchor_count, dtype=torch.float32).to(device))

        anchor_feature = contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        loss = _get_loss(mask, anchor_dot_contrast)

        return loss
