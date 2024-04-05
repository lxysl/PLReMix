import torch

from models.model import PLReMixModel


def create_model(args, device, pretrain=False):
    if args.backbone == 'resnet18':
        from models.resnet import resnet18
        backbone = resnet18()
    elif 'preact' in args.backbone:
        from models.resnet import preact_resnet18
        backbone = preact_resnet18()
        if args.dataset == 'tiny_imagenet':
            backbone['dim'] *= 4
    elif args.backbone == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50(pretrained=pretrain)
        print('Is model pretrained: ', pretrain)
    elif args.backbone == 'inception':
        from models.inception_resnet_v2 import inception_resnet_v2
        backbone = inception_resnet_v2()
    else:
        raise ValueError('Invalid backbone {}'.format(args.backbone))

    model = PLReMixModel(backbone, args.dataset, args.num_classes)

    if torch.__version__ >= '2.0.0':
        return torch.compile(model.to(device))
    else:
        return model.to(device)
