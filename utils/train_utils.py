import os
import math
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

from utils.common_utils import AverageMeter, ProgressMeter


def adjust_lr(lr, cos, optimizer1, optimizer2, epoch, num_epochs, milestones=None):
    if cos:
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    else:
        if epoch >= num_epochs // 2:
            lr /= 10
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr


def resume(checkpoint_path, net1, net2, optimizer1, optimizer2, device):
    # checkpoint = torch.load(wandb.restore(CHECKPOINT_PATH), map_location=device)  # resume checkpoint from wandb
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Resume from checkpoint at epoch {}'.format(checkpoint["epoch"]))

    net1_state_dict = checkpoint["net1_state_dict"]
    net2_state_dict = checkpoint["net2_state_dict"]
    # remove the prefix added by torchinductor if exists (torch.compile() API in PyTorch 2.0.0+)
    # unwanted_prefix = '_orig_mod.'
    # for k, v in list(net1_state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         net1_state_dict[k[len(unwanted_prefix):]] = net1_state_dict.pop(k)
    # for k, v in list(net2_state_dict.items()):
    #     if k.startswith(unwanted_prefix):
    #         net2_state_dict[k[len(unwanted_prefix):]] = net2_state_dict.pop(k)
    net1.load_state_dict(net1_state_dict)
    net2.load_state_dict(net2_state_dict)
    optimizer1.load_state_dict(checkpoint["optimizer1_state_dict"])
    optimizer2.load_state_dict(checkpoint["optimizer2_state_dict"])

    all_loss = checkpoint["all_loss"]
    all_loss_proto = checkpoint["all_loss_proto"]
    meta_info = checkpoint["meta_info"]
    meta_info['device'] = device
    epoch = checkpoint["epoch"] + 1

    return net1, net2, optimizer1, optimizer2, all_loss, all_loss_proto, meta_info, epoch


def save(checkpoint_path, net1, net2, optimizer1, optimizer2, all_loss, all_loss_proto, meta_info, epoch):
    torch.save(
        {
            "net1_state_dict": net1.state_dict(),
            "net2_state_dict": net2.state_dict(),
            "optimizer1_state_dict": optimizer1.state_dict(),
            "optimizer2_state_dict": optimizer2.state_dict(),
            "all_loss": all_loss,
            "all_loss_proto": all_loss_proto,
            "meta_info": meta_info,
            "epoch": epoch,
        },
        checkpoint_path,
    )
    # wandb.save(CHECKPOINT_PATH)  # save checkpoint to wandb
    print('Checkpoint Saved')


@torch.no_grad()
def init_prototypes(net, eval_loader, device):
    net.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for _, (inputs, labels, _) in enumerate(eval_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            features = net(inputs, forward_pass='proj')
            all_features.append(features)
            all_labels.append(labels)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    net.init_prototypes(all_features, all_labels)


def gmm_selection(args, cur_net, model, all_loss, all_loss_proto, eval_loader, criterion, device, epoch):
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset), dtype=torch.float, device=device)
    pl = torch.zeros(len(eval_loader.dataset), dtype=torch.long, device=device)
    op = torch.zeros(len(eval_loader.dataset), args.num_classes, dtype=torch.float, device=device)
    pt = torch.zeros(len(eval_loader.dataset), args.num_classes, dtype=torch.float, device=device)
    ft = torch.zeros(len(eval_loader.dataset), 128, dtype=torch.float, device=device)
    losses_proto = torch.zeros(len(eval_loader.dataset), dtype=torch.float, device=device)
    # gt = torch.zeros(len(eval_loader.dataset), dtype=torch.long, device=device)
    # tg = torch.zeros(len(eval_loader.dataset), dtype=torch.long, device=device)
    # ground_targets = eval_loader.dataset.dataset.targets
    paths = []  # if args.dataset == 'clothing1m' or 'webvision', the index is img_path

    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            index = indices
            outputs, logits_proto, features = model(inputs, forward_pass='all')
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, targets)
            loss_proto = criterion(logits_proto, targets)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                pl[index[b]] = predicted[b]
                op[index[b]] = outputs[b]
                pt[index[b]] = logits_proto[b]
                ft[index[b]] = features[b]
                losses_proto[index[b]] = loss_proto[b]
                # gt[index[b]] = ground_targets[index[b]]
                # tg[index[b]] = targets[b]

    losses = (losses - losses.min()) / (losses.max() - losses.min())  # normalised losses for each image
    losses_proto = (losses_proto - losses_proto.min()) / (losses_proto.max() - losses_proto.min())
    all_loss.append(losses)
    all_loss_proto.append(losses_proto)

    input_loss = losses.reshape(-1, 1)
    input_loss_proto = losses_proto.reshape(-1, 1)

    # fit a two-component GMM(loss_proto-loss) to the loss
    input_loss = input_loss.cpu().numpy()
    input_loss_proto = input_loss_proto.cpu().numpy()
    gmm_input = np.column_stack((input_loss, input_loss_proto))
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4, covariance_type='full')
    gmm.fit(gmm_input)
    mean_square_dists = np.array([np.sum(np.square(gmm.means_[i])) for i in range(2)])
    argmin, argmax = mean_square_dists.argmin(), mean_square_dists.argmax()
    prob = gmm.predict_proba(gmm_input)
    prob = prob[:, argmin]
    pred_clean = (prob > args.p_threshold).nonzero()[0]
    pred_noisy = (prob <= args.p_threshold).nonzero()[0]

    # if not args.wo_wandb:
    #     # plot gmm
    #     is_clean = (gt == tg).cpu().numpy()
    #     is_noisy = ~is_clean
    #     means = gmm.means_[[argmin, argmax], :]
    #     covs = gmm.covariances_[[argmin, argmax], :]
    #     fig = plot_gmm(args, means, covs, input_loss, input_loss_proto, is_noisy, epoch, cur_net, '$loss^proto$')
    #     wandb.log({(cur_net + "gmm"): wandb.Image(fig)}, step=epoch)

    return prob, pred_clean, pred_noisy, all_loss, all_loss_proto, pl, op, pt, ft, paths


def plot_gmm(args, means, covs, losses, losses_proto, is_noisy, epoch, cur_net, label):
    # 定义均值向量和协方差矩阵
    mean1, cov1, mean2, cov2 = means[0], covs[0], means[1], covs[1]

    # 生成二维高斯分布
    x, y = np.mgrid[0:1:0.01, 0:1:0.01]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    rv1 = multivariate_normal(mean1, cov1)
    rv2 = multivariate_normal(mean2, cov2)
    z1 = rv1.pdf(pos)
    z2 = rv2.pdf(pos)

    # 绘制分布图
    fig = plt.figure()
    plt.contourf(x, y, z1, cmap='Blues', alpha=0.5)
    plt.contourf(x, y, z2, cmap='Reds', alpha=0.5)
    plt.scatter(losses[is_noisy][:256], losses_proto[is_noisy][:256], s=4, c='r', label='Noisy samples')
    plt.scatter(losses[~is_noisy][:256], losses_proto[~is_noisy][:256], s=4, c='b', label='Clean samples')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Normalized $l_{cls}$', fontsize=26)
    plt.ylabel('Normalized $l_{proto}$', fontsize=26)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc=0, fontsize=20)
    plt.title('{} GMM at epoch {}'.format(cur_net, epoch))
    if not os.path.exists(os.path.join('checkpoint', args.dataset, str(args.r))):
        os.makedirs(os.path.join('checkpoint', args.dataset, str(args.r)))
    plt.savefig(os.path.join('checkpoint', args.dataset, str(args.r), '{}_gmm_epoch_{}.pdf'.format(cur_net, epoch)),
                format='pdf', bbox_inches='tight', pad_inches=0.0, dpi=1000)
    plt.close()
    return fig


@torch.no_grad()
def build_mask_step(args, outputs, k, labels, device):
    outputs, labels = outputs.to(device), labels.to(device)

    tops = torch.zeros_like(outputs, device=device)
    if k == 0:
        topk = torch.topk(outputs, 1, dim=1)[1]
        # make the topk of the outputs to be 1, others to be 0
        tops = torch.scatter(tops, 1, topk, 1)
    else:
        topk = torch.topk(outputs, k, dim=1)[1]
        # make the topk of the outputs to be 1, others to be 0
        tops = torch.scatter(tops, 1, topk, 1)

        tops = torch.scatter(tops, 1, labels.unsqueeze(dim=1), 1)

    neg_samples = torch.ones(len(outputs), len(outputs), dtype=torch.float, device=device)

    # conflict matrix, where conflict[i][j]==0 means the i-th and j-th class do not have overlap topk,
    # can be used as negative pairs
    conflicts = torch.matmul(tops, tops.t())
    # negative pairs: (conflicts == 0) or (conflicts != 0 and neg_samples == 0)
    neg_samples = neg_samples * conflicts
    # make a mask metrix, where neg_samples==0, the mask is -1 (negative pairs), otherwise 0 (neglect pairs)
    mask = torch.where(neg_samples == 0, -1, 0)
    # make the diagonal of the mask to be 1 (positive pairs)
    mask = torch.where(torch.eye(len(outputs), device=device) == 1, 1, mask)
    return mask


def semi_train_step(args, net, net2, inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_u1, inputs_u2, inputs_u3,
                    inputs_u4, labels_x, w_x, criterion, batch_idx, num_iter, epoch, device):
    batch_size = inputs_x1.size(0)

    inputs_x1, inputs_x2, inputs_x3, inputs_x4 = (inputs_x1.to(device), inputs_x2.to(device),
                                                  inputs_x3.to(device), inputs_x4.to(device))
    inputs_u1, inputs_u2, inputs_u3, inputs_u4 = (inputs_u1.to(device), inputs_u2.to(device),
                                                  inputs_u3.to(device), inputs_u4.to(device))
    labels_x = labels_x.to(device)
    # Transform label to one-hot
    labels_x_soft = torch.zeros(batch_size, args.num_classes, device=device).scatter_(1, labels_x.view(-1, 1), 1)
    w_x = w_x.view(-1, 1).type(torch.FloatTensor).to(device)

    with torch.no_grad():
        # label co-guessing of unlabeled samples
        outputs_u11 = net(inputs_u1, forward_pass='cls')
        outputs_u12 = net(inputs_u2, forward_pass='cls')
        outputs_u21 = net2(inputs_u1, forward_pass='cls')
        outputs_u22 = net2(inputs_u2, forward_pass='cls')

        pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
              torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
        ptu = pu ** (1 / args.T)  # temperature sharpening

        targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
        targets_u = targets_u.detach()

        # label refinement of labeled samples
        outputs_x11 = net(inputs_x1, forward_pass='cls')
        outputs_x12 = net(inputs_x2, forward_pass='cls')

        px = (torch.softmax(outputs_x11, dim=1) + torch.softmax(outputs_x12, dim=1)) / 2
        px = w_x * labels_x_soft + (1 - w_x) * px
        ptx = px ** (1 / args.T)  # temperature sharpening

        targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
        targets_x = targets_x.detach()

    # mixmatch
    l = np.random.beta(args.alpha, args.alpha)
    l = max(l, 1 - l)

    if args.aug in ['autoaug', 'randaug']:
        all_inputs = torch.cat([inputs_x3, inputs_x4, inputs_u3, inputs_u4], dim=0)
    else:
        all_inputs = torch.cat([inputs_x1, inputs_x2, inputs_u1, inputs_u2], dim=0)
    all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

    idx = torch.randperm(all_inputs.size(0))

    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    logits = net(mixed_input, forward_pass='cls')
    logits_x = logits[:batch_size * 2]
    logits_u = logits[batch_size * 2:]

    Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                             args.lambda_u, epoch + batch_idx / num_iter, args.warm_up)

    # regularization
    prior = torch.ones(args.num_classes) / args.num_classes
    prior = prior.to(device)
    pred_mean = torch.softmax(logits, dim=1).mean(0)
    penalty = torch.sum(prior * torch.log(prior / pred_mean))

    loss_semi = Lx + lamb * Lu + penalty
    return loss_semi


def plr_train_step(args, net, inputs_aug1, inputs_aug2, mask, criterion, device):
    inputs_aug1, inputs_aug2 = inputs_aug1.to(device), inputs_aug2.to(device)
    inputs = torch.cat([inputs_aug1, inputs_aug2], dim=0)
    features = net(inputs, forward_pass='proj')
    f1, f2 = torch.split(features, [inputs_aug1.shape[0], inputs_aug2.shape[0]], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    loss_crl = criterion(features, mask=mask)
    return loss_crl


@torch.no_grad()
def noise_correction(proj_outputs, cls_outputs, labels, indices, meta_info, device):
    proj_outputs, cls_outputs, labels, indices = (proj_outputs.to(device), cls_outputs.to(device),
                                                  labels.to(device), indices.to(device))
    # noise cleaning for clustering
    alpha = 0.5
    soft_labels = alpha * F.softmax(proj_outputs, dim=1) + (1 - alpha) * F.softmax(cls_outputs, dim=1)

    # assign a new pseudo label
    max_score, hard_label = soft_labels.max(1)
    correct_idx = max_score > meta_info['pseudo_th']
    labels[correct_idx] = hard_label[correct_idx]

    return labels


@torch.no_grad()
def pseudo_selection(proj_outputs, cls_outputs, labels, meta_info, epoch, num_epochs, device):
    proj_outputs, cls_outputs, labels = (proj_outputs.to(device), cls_outputs.to(device), labels.to(device))
    # alpha is cosinely decayed from 1 to 0
    alpha = 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
    avg_outputs = F.softmax((1 - alpha) * proj_outputs + alpha * cls_outputs, dim=1)

    max_probs, _ = torch.max(avg_outputs, dim=-1, keepdim=True)
    meta_info['time_p'] = meta_info['time_p'] * meta_info['semi_m'] + (1 - meta_info['semi_m']) * max_probs.mean()
    meta_info['p_model'] = meta_info['p_model'] * meta_info['semi_m'] + (1 - meta_info['semi_m']) * avg_outputs.mean(dim=0)

    max_probs, max_idx = avg_outputs.max(dim=-1)
    mod = meta_info['p_model'] / torch.max(meta_info['p_model'], dim=-1)[0]
    mask = max_probs.ge(meta_info['time_p'] * mod[max_idx])

    return mask


def uniform_warmup(args, epoch, net, optimizer, train_loader, ce_criterion, info_nce_loss, conf_penalty, scaler,
                   device):
    ce_losses = AverageMeter('CE Loss', ':.4e')
    crl_losses = AverageMeter('CRL Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [ce_losses, crl_losses], prefix="Epoch: [{}]".format(epoch))

    net.train()
    for batch_idx, batch in enumerate(train_loader):
        inputs, inputs_aug1, inputs_aug2, labels, _ = batch
        inputs, inputs_aug1, inputs_aug2, labels = (inputs.to(device), inputs_aug1.to(device),
                                                    inputs_aug2.to(device), labels.to(device))

        # Cross-entropy loss
        outputs = net(inputs, forward_pass='cls')
        loss_ce = ce_criterion(outputs, labels)
        # penalize confident prediction for asymmetric noise
        penalty = conf_penalty(outputs) if args.noise_mode == 'asym' else torch.tensor(0.0, device=device)
        loss_ce += penalty
        loss_ce.backward()

        # FlatNCE loss in https://arxiv.org/abs/2107.01152
        inputs_crl = torch.cat([inputs_aug1, inputs_aug2], dim=0)
        features = net(inputs_crl, forward_pass='proj')
        logits, labels = info_nce_loss(features)
        v = torch.logsumexp(logits, dim=1, keepdim=True)
        loss_crl = torch.exp(v - v.detach()).mean()
        loss_crl.backward()

        optimizer.step()
        optimizer.zero_grad()

        ce_losses.update(loss_ce.item())
        crl_losses.update(loss_crl.item())
        if batch_idx % 100 == 0:
            progress.display(batch_idx)

    if not args.wo_wandb:
        wandb.log({"ce loss": ce_losses.avg,
                   "crl loss": crl_losses.avg}, step=epoch)


def linear_rampup(current, warm_up, rampup_length=100):
    current = np.clip((current - warm_up) / rampup_length, 0.01, 0.1)
    return float(current)


def uniform_train(args, epoch, net, net2, optimizer, labeled_train_loader, unlabeled_train_loader, semi_criterion,
                  crl_loss, meta_info, scaler, device):
    semi_loss_meter = AverageMeter('Semi Loss', ':.4e')
    crl_loss_meter = AverageMeter('CRL Loss', ':.4e')  # always equal to 1
    progress = ProgressMeter(len(labeled_train_loader),
                             [semi_loss_meter, crl_loss_meter],
                             prefix="Epoch: [{}]".format(epoch))

    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_train_loader)
    num_iter = (len(labeled_train_loader.dataset) // args.batch_size) + 1
    print('len of labeled train dataset: ', len(labeled_train_loader.dataset))
    print('len of unlabeled train dataset: ', len(unlabeled_train_loader.dataset))

    for batch_idx, batch_x in enumerate(labeled_train_loader):
        try:
            batch_u = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_train_loader)
            batch_u = next(unlabeled_train_iter)

        inputs_x1, inputs_x2, inputs_x3, inputs_x4, labels_x, w_x, indices_x = batch_x
        inputs_u1, inputs_u2, inputs_u3, inputs_u4, labels_u, indices_u = batch_u

        # build contrastive mask for PLR loss
        indices = torch.cat((indices_x, indices_u), dim=0)
        cls_outputs = meta_info['cls_outputs'][indices, :]
        labels = torch.cat((labels_x, labels_u), dim=0)
        contrastive_mask = build_mask_step(args, cls_outputs, meta_info['topk'], labels, device)

        loss_semi = semi_train_step(args, net, net2, inputs_x1, inputs_x2, inputs_x3, inputs_x4, inputs_u1,
                                    inputs_u2, inputs_u3, inputs_u4, labels_x, w_x, semi_criterion,
                                    batch_idx, num_iter, epoch, device)
        loss_crl = plr_train_step(args, net, torch.cat((inputs_x3, inputs_u3), dim=0),
                                  torch.cat((inputs_x4, inputs_u4), dim=0), contrastive_mask,
                                  crl_loss, device)

        (loss_semi + args.lambda_c * loss_crl).backward()
        optimizer.step()
        optimizer.zero_grad()

        semi_loss_meter.update(loss_semi.item())
        crl_loss_meter.update(loss_crl.item())

        if batch_idx % 50 == 0:
            progress.display(batch_idx)

    # noise correction
    mask = pseudo_selection(meta_info['proj_outputs'], meta_info['cls_outputs'], meta_info['pred_label'],
                            meta_info, epoch, args.num_epochs, device)

    # update class prototypes
    if epoch > args.warm_up - 1:
        features = meta_info['features'].to(device)
        labels = meta_info['pred_label'].to(device)
        net.update_prototypes(features[mask], labels[mask])


@torch.no_grad()
def val(args, epoch, net1, net2, val_loader, device):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            inputs, targets = batch['image'].to(device), batch['target'].to(device)
            outputs1 = net1(inputs, forward_pass='cls')
            outputs2 = net2(inputs, forward_pass='cls')
            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print('\nEpoch:%d   Accuracy:%.2f\n' % (epoch, acc))
    if not args.wo_wandb:
        wandb.log({'test accuracy': acc}, step=epoch)
