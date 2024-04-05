import os
import copy
import random
import argparse
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
import wandb

from data.data_loader import get_loader
from models.model_loader import create_model
from losses.losses import SemiLoss, NegEntropy, InfoNCELoss, PLRLoss
from utils.train_utils_webvision import (adjust_lr, resume, save, init_prototypes, gmm_selection,
                                         uniform_warmup, uniform_train, val, test)
from utils.common_utils import iterateAllFile


parser = argparse.ArgumentParser(description='PyTorch PLReMix Training')
parser.add_argument('--dataset', default='webvision', type=str, choices=['webvision'])
parser.add_argument('--num_classes', default=50, type=int)
parser.add_argument('--data_path', default='~/Documents/WebVision', type=str, help='path to dataset')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--backbone', default='inception', type=str)
parser.add_argument('--pretrain', action='store_true', help='use pretrain model')

parser.add_argument('--batch_size', default=48, type=int, help='train batch size')
parser.add_argument('--lr', '--learning_rate', default=0.015, type=float, help='initial learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--cos', action='store_true', default=False, help='use cosine lr schedule')
parser.add_argument('--num_epochs', default=150, type=int)

parser.add_argument('--num_workers', default=16, type=int, help='num of workers to use')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--seed', default=123)

parser.add_argument('--alpha', default=0.5, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=0, type=float, help='weight for unsupervised loss')
parser.add_argument('--lambda_c', default=1, type=float, help='weight for contrastive loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature in semi loss')
parser.add_argument('--topk', default=3, type=int, help='kappa in PLR loss')

parser.add_argument('--aug', default='autoaug', type=str,
                    choices=['train', 'simclr', 'autoaug', 'randaug'],
                    help='use FixMatch following AugDesc-WS')
parser.add_argument('--crl', default='plr', type=str, choices=['plr', 'flat_plr'])
parser.add_argument('--mcrop', action='store_true', help='use multi-crop')

parser.add_argument('--wo_wandb', action='store_true', help='without using wandb to log')
parser.add_argument('--offline', action='store_true', help='use wandb in offline mode')
parser.add_argument('--resume_id', default='', type=str)
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.gpu))
torch.cuda.set_device(args.gpu)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if torch.__version__ >= '2.0.0':
    torch.set_float32_matmul_precision('high')

if args.dataset == 'webvision':
    args.num_classes = 50
    args.warm_up = 2
    args.backbone = 'inception'

cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
if not args.wo_wandb:
    wandb.init(project=args.dataset,
               name=cur_time if args.resume_id == '' else None,
               id=None if args.resume_id == '' else args.resume_id,
               resume=None if args.resume_id == '' else 'must',
               config=vars(args),
               mode='offline' if args.offline else 'online')
    print(vars(args))
    for root, f in iterateAllFile('.'):
        if 'wandb' not in root and 'archive' not in root and 'torchinductor' not in root:
            if f[-3:] == '.py':
                # print(root, f)
                wandb.save(f, base_path=root, policy="now")
    CHECKPOINT_PATH = "./checkpoint/{}.tar".format(wandb.run.id)
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')


def main():
    meta_info = {'r': args.r, 'noise_mode': args.noise_mode, 'dataset': args.dataset, 'transform': 'train',
                 'num_classes': args.num_classes, 'probability': None, 'pred_clean': None, 'pred_noisy': None,
                 'output': None, 'device': device, 'pseudo_th': None, 'multi_crop': args.mcrop,
                 'noise_file': './data/noise_file/{}/{:.2f}{}.json'.format(
                     args.dataset, args.r, '_asym' if args.noise_mode == 'asym' else '')}

    print('Building net')
    net1 = create_model(args, device, args.pretrain)
    net2 = create_model(args, device, args.pretrain)
    cudnn.benchmark = True

    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    semi_loss = SemiLoss()
    eval_loss = nn.CrossEntropyLoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss()
    info_nce_loss = InfoNCELoss(temperature=0.1,
                                batch_size=args.batch_size * 2,
                                flat=('flat' in args.crl),
                                n_views=8 if args.mcrop else 2)
    plr_loss = PLRLoss(flat=('flat' in args.crl))
    conf_penalty = NegEntropy()
    scaler = GradScaler()

    milestone1, milestone2 = 15, 30
    topk_list = [args.topk for _ in range(args.num_epochs + 1)]
    if args.topk > 1:
        topk_list[milestone1:] = [args.topk - 1 for _ in range(args.num_epochs + 1)]
    if args.topk > 2:
        topk_list[milestone2:] = [args.topk - 2 for _ in range(args.num_epochs + 1)]
    pseudo_th_list = [0.8 for _ in range(args.num_epochs + 1)]
    lr_milestones = [60, 120]  # first decay at 60, second at 120

    val_loader = get_loader(args, 'val', meta_info)
    meta_info1 = copy.deepcopy(meta_info)
    meta_info1['dataset'] = 'imagenet'
    imagenet_val_loader = get_loader(args, 'val', meta_info1)
    test_loader = get_loader(args, 'test', meta_info)

    all_loss = [[], []]  # save the history of losses from two networks
    all_loss_proto = [[], []]  # save the history of distances from two networks

    epoch = 0
    if not args.wo_wandb and wandb.run.resumed and os.path.exists(CHECKPOINT_PATH):  # resume from checkpoint
        net1, net2, optimizer1, optimizer2, all_loss, all_loss_proto, meta_info, epoch = (
            resume(CHECKPOINT_PATH, net1, net2, optimizer1, optimizer2, device))

    while epoch < args.num_epochs + 1:
        meta_info['epoch'] = epoch
        adjust_lr(args.lr, args.cos, optimizer1, optimizer2, epoch, args.num_epochs, lr_milestones)

        if epoch < args.warm_up:
            warmup_train_loader = get_loader(args, 'warmup', meta_info)

            print('\nWarmup Net1')
            meta_info['cur_net'] = 'net1'
            uniform_warmup(args, epoch, net1, optimizer1, warmup_train_loader,
                           ce_loss, info_nce_loss, conf_penalty, scaler, device)

            print('\nWarmup Net2')
            meta_info['cur_net'] = 'net2'
            uniform_warmup(args, epoch, net2, optimizer2, warmup_train_loader,
                           ce_loss, info_nce_loss, conf_penalty, scaler, device)

            if epoch == args.warm_up - 1:
                eval_loader = get_loader(args, 'eval_train', meta_info)
                init_prototypes(net1, eval_loader, device)
                init_prototypes(net2, eval_loader, device)

        else:
            print('\nGMM Select')
            eval_loader = get_loader(args, 'eval_train', meta_info)

            prob1, pred_clean1, pred_noisy1, all_loss[0], all_loss_proto[0], pl1, op1, pt1, ft1, paths1 = (
                gmm_selection(args, 'net1', net1, all_loss[0], all_loss_proto[0],
                              eval_loader, eval_loss, device, epoch))
            prob2, pred_clean2, pred_noisy2, all_loss[1], all_loss_proto[1], pl2, op2, pt2, ft2, paths2 = (
                gmm_selection(args, 'net2', net2, all_loss[1], all_loss_proto[1],
                              eval_loader, eval_loss, device, epoch))

            print('\nUniform Train Net1')
            meta_info.update(
                {'cur_net': 'net1', 'probability': prob2, 'pred_clean': pred_clean2, 'pred_noisy': pred_noisy2,
                 'pred_label': pl2, 'cls_outputs': op2, 'proj_outputs': pt2, 'features': ft2,
                 'pseudo_th': pseudo_th_list[epoch], 'topk': topk_list[epoch], 'paths': paths2})
            labeled_train_loader, unlabeled_train_loader = get_loader(args, 'train', meta_info)
            uniform_train(args, epoch, net1, net2, optimizer1, labeled_train_loader, unlabeled_train_loader,
                          semi_loss, plr_loss, meta_info, scaler, device)

            print('\nUniform Train Net2')
            meta_info.update(
                {'cur_net': 'net2', 'probability': prob1, 'pred_clean': pred_clean1, 'pred_noisy': pred_noisy1,
                 'pred_label': pl1, 'cls_outputs': op1, 'proj_outputs': pt1, 'features': ft1,
                 'pseudo_th': pseudo_th_list[epoch], 'topk': topk_list[epoch], 'paths': paths1})
            labeled_train_loader, unlabeled_train_loader = get_loader(args, 'train', meta_info)
            uniform_train(args, epoch, net2, net1, optimizer2, labeled_train_loader, unlabeled_train_loader,
                          semi_loss, plr_loss, meta_info, scaler, device)

        print('\nValidation')
        val(args, epoch, net1, net2, val_loader, device)
        val(args, epoch, net1, net2, imagenet_val_loader, device, imagenet=True)

        if not args.wo_wandb:
            save(CHECKPOINT_PATH, net1, net2, optimizer1, optimizer2, all_loss, all_loss_proto, meta_info, epoch)
        epoch += 1

    print('\nTest')
    test(args, epoch, net1, net2, test_loader, device)
    # test on imagenet val set
    test(args, epoch, net1, net2, imagenet_val_loader, device, imagenet=True)

    if not args.wo_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
