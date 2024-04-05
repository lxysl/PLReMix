import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.scan_augment import Augment, Cutout, GaussianBlur
from data.auto_augment import CIFARPolicy, ImageNetPolicy
from data.rand_augment import RandAugment


def get_train_dataset(args, transform, meta_info=None):
    # Base dataset
    if meta_info['dataset'] == 'cifar10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(root=args.data_path, train=True, transform=transform, download=True)
    elif meta_info['dataset'] == 'cifar100':
        from data.cifar import CIFAR100
        dataset = CIFAR100(root=args.data_path, train=True, transform=transform, download=True)
    elif meta_info['dataset'] == 'tiny_imagenet':
        from data.tiny_imagenet import TinyImageNet
        dataset = TinyImageNet(root=args.data_path, train=True, transform=transform)
    elif meta_info['dataset'] == 'clothing1m':
        from data.clothing1m import Clothing1M
        dataset = Clothing1M(root=args.data_path, transform=transform, meta_info=meta_info)
    elif meta_info['dataset'] == 'webvision':
        from data.webvision import WebVision
        dataset = WebVision(root=args.data_path, transform=transform, meta_info=meta_info)
    else:
        raise ValueError('Invalid train dataset {}'.format(meta_info['dataset']))

    if meta_info['dataset'] in ['cifar10', 'cifar100', 'tiny_imagenet']:
        # Wrap into other dataset (__getitem__ changes)
        # convert to noisy dataset
        from data.custom_dataset import NoisyDataset
        dataset = NoisyDataset(dataset, meta_info)

    return dataset


def get_val_dataset(args, transform=None, meta_info=None):
    # Base dataset
    if meta_info['dataset'] == 'cifar10':
        from data.cifar import CIFAR10
        dataset = CIFAR10(root=args.data_path, train=False, transform=transform, download=True)
    elif meta_info['dataset'] == 'cifar100':
        from data.cifar import CIFAR100
        dataset = CIFAR100(root=args.data_path, train=False, transform=transform, download=True)
    elif meta_info['dataset'] == 'tiny_imagenet':
        from data.tiny_imagenet import TinyImageNet
        dataset = TinyImageNet(root=args.data_path, train=False, transform=transform)
    elif meta_info['dataset'] == 'clothing1m':
        from data.clothing1m import Clothing1M
        dataset = Clothing1M(root=args.data_path, transform=transform, meta_info=meta_info)
    elif meta_info['dataset'] == 'webvision':
        from data.webvision import WebVision
        dataset = WebVision(root=args.data_path, transform=transform, meta_info=meta_info)
    elif meta_info['dataset'] == 'imagenet':
        from data.webvision import ImageNet
        dataset = ImageNet(root=args.data_path, transform=transform, meta_info=meta_info)
    else:
        raise ValueError('Invalid validation dataset {}'.format(meta_info['dataset']))
    return dataset


def get_test_dataset(args, transform=None, meta_info=None):
    # Base dataset
    if meta_info['dataset'] == 'clothing1m':
        from data.clothing1m import Clothing1M
        dataset = Clothing1M(root=args.data_path, transform=transform, meta_info=meta_info)
    elif meta_info['dataset'] == 'webvision':
        from data.webvision import WebVision
        dataset = WebVision(root=args.data_path, transform=transform, meta_info=meta_info)
    else:
        raise ValueError('Invalid test dataset {}'.format(meta_info['dataset']))
    return dataset


def get_train_dataloader(args, dataset, explicit_batch_size=None):
    batch_size = args.batch_size
    if explicit_batch_size is not None:
        batch_size = explicit_batch_size
    return DataLoader(dataset, num_workers=args.num_workers, batch_size=batch_size, pin_memory=True,
                      drop_last=True, shuffle=True)


def get_val_dataloader(args, dataset):
    return DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                      drop_last=False, shuffle=False)


def get_test_dataloader(args, dataset):
    return DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True,
                      drop_last=False, shuffle=False)


def get_train_transformations(meta_info):
    if meta_info['dataset'] in ['cifar10', 'cifar100']:
        if meta_info['dataset'] == 'cifar10':
            normalize = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
        elif meta_info['dataset'] == 'cifar100':
            normalize = {'mean': [0.5071, 0.4865, 0.4409], 'std': [0.2673, 0.2564, 0.2762]}
        else:
            raise ValueError('Invalid dataset {}'.format(meta_info['dataset']))

        if meta_info['transform'] == 'train':
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)])
        elif meta_info['transform'] == 'simclr':
            return transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(**normalize),
            ])
        elif meta_info['transform'] == 'autoaug':
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFARPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize),
            ])
        elif meta_info['transform'] == 'randaug':
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugment(4, 6),
                transforms.ToTensor(),
                transforms.Normalize(**normalize),
            ])
        else:
            raise ValueError('Invalid augmentation strategy {}'.format(meta_info['transform']))

    elif meta_info['dataset'] == 'tiny_imagenet':
        normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        if meta_info['transform'] == 'train':
            return transforms.Compose([
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)])
        elif meta_info['transform'] == 'simclr':
            return transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.35, 0.4, 0.07)]),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(**normalize),
            ])
        elif meta_info['transform'] == 'autoaug':
            return transforms.Compose([
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize),
            ])
        elif meta_info['transform'] == 'randaug':
            return transforms.Compose([
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                RandAugment(4, 6),
                transforms.ToTensor(),
                transforms.Normalize(**normalize),
            ])
        else:
            raise ValueError('Invalid augmentation strategy {}'.format(meta_info['transform']))

    elif meta_info['dataset'] == 'clothing1m':
        normalize = {'mean': [0.6959, 0.6537, 0.6371], 'std': [0.3113, 0.3192, 0.3214]}
        size_crops = [196, 120]
        min_scale_crops = [0.7, 0.4]
        max_scale_crops = [1., 0.7]
        nmb_crops = [2, 6]
        if meta_info['transform'] == 'train':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)])
        elif meta_info['transform'] in ['simclr', 'autoaug', 'randaug']:
            if meta_info['transform'] == 'simclr':
                trans = [transforms.RandomHorizontalFlip(),
                         transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.07)], p=0.8),
                         transforms.RandomGrayscale(p=0.2),
                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                         transforms.ToTensor(),
                         transforms.Normalize(**normalize), ]
            elif meta_info['transform'] == 'autoaug':
                trans = [transforms.RandomHorizontalFlip(),
                         ImageNetPolicy(),
                         transforms.ToTensor(),
                         transforms.Normalize(**normalize), ]
            elif meta_info['transform'] == 'randaug':
                trans = [transforms.RandomHorizontalFlip(),
                         RandAugment(4, 6),
                         transforms.ToTensor(),
                         transforms.Normalize(**normalize), ]
            else:
                raise ValueError('Invalid augmentation strategy {}'.format(meta_info['transform']))
            if meta_info['multi_crop']:
                multi_crop_trans = []
                for i in range(len(size_crops)):
                    random_resized_crop = transforms.RandomResizedCrop(
                        size_crops[i],
                        scale=(min_scale_crops[i], max_scale_crops[i]))
                    multi_crop_trans.extend([
                                                transforms.Compose([random_resized_crop] + trans)
                                            ] * nmb_crops[i])
                return multi_crop_trans
            else:
                return transforms.Compose([
                                              transforms.Resize(256),
                                              transforms.RandomCrop(224),
                                          ] + trans)
        else:
            raise ValueError('Invalid augmentation strategy {}'.format(meta_info['transform']))

    elif meta_info['dataset'] in ['webvision', 'imagenet']:
        normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        size_crops = [224, 128]
        min_scale_crops = [0.14, 0.05]
        max_scale_crops = [1., 0.14]
        nmb_crops = [2, 6]
        if meta_info['transform'] == 'train':
            return transforms.Compose([
                transforms.Resize(320),
                transforms.RandomCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**normalize)])
        elif meta_info['transform'] in ['simclr', 'autoaug', 'randaug']:
            if meta_info['transform'] == 'simclr':
                trans = [transforms.RandomHorizontalFlip(),
                         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                         transforms.RandomGrayscale(p=0.2),
                         transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                         transforms.ToTensor(),
                         transforms.Normalize(**normalize), ]
            elif meta_info['transform'] == 'autoaug':
                trans = [transforms.RandomHorizontalFlip(),
                         ImageNetPolicy(),
                         transforms.ToTensor(),
                         transforms.Normalize(**normalize), ]
            elif meta_info['transform'] == 'randaug':
                trans = [transforms.RandomHorizontalFlip(),
                         RandAugment(4, 6),
                         transforms.ToTensor(),
                         transforms.Normalize(**normalize), ]
            else:
                raise ValueError('Invalid augmentation strategy {}'.format(meta_info['transform']))
            if meta_info['multi_crop']:
                multi_crop_trans = []
                for i in range(len(size_crops)):
                    random_resized_crop = transforms.RandomResizedCrop(
                        size_crops[i],
                        scale=(min_scale_crops[i], max_scale_crops[i]))
                    multi_crop_trans.extend([
                                                transforms.Compose([random_resized_crop] + trans)
                                            ] * nmb_crops[i])
                return multi_crop_trans
            else:
                return transforms.Compose([
                                              transforms.Resize(320),
                                              transforms.RandomCrop(299),
                                          ] + trans)
        else:
            raise ValueError('Invalid augmentation strategy {}'.format(meta_info['transform']))
    else:
        raise ValueError('Invalid dataset {}'.format(meta_info['dataset']))


def get_val_transformations(meta_info):
    if meta_info['dataset'] in ['cifar10', 'cifar100', 'tiny_imagenet']:
        if meta_info['dataset'] == 'cifar10':
            normalize = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
        elif meta_info['dataset'] == 'cifar100':
            normalize = {'mean': [0.5071, 0.4865, 0.4409], 'std': [0.2673, 0.2564, 0.2762]}
        elif meta_info['dataset'] == 'tiny_imagenet':
            normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        else:
            raise ValueError('Invalid dataset {}'.format(meta_info['dataset']))
        return transforms.Compose([
            transforms.CenterCrop(32 if meta_info['dataset'] in ['cifar10', 'cifar100'] else 64),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)])
    elif meta_info['dataset'] == 'clothing1m':
        normalize = {'mean': [0.6959, 0.6537, 0.6371], 'std': [0.3113, 0.3192, 0.3214]}
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)])
    elif meta_info['dataset'] in ['webvision', 'imagenet']:
        normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
        return transforms.Compose([
            transforms.Resize(320),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(**normalize)])
    else:
        raise ValueError('Invalid dataset {}'.format(meta_info['dataset']))


def get_loader(args, mode, meta_info):
    if mode == 'val':
        meta_info['transform'] = 'val'
        val_transformations = get_val_transformations(meta_info)
        meta_info['mode'] = 'val'
        val_dataset = get_val_dataset(args, val_transformations, meta_info=meta_info)
        val_dataloader = get_val_dataloader(args, val_dataset)
        return val_dataloader

    elif mode == 'test':
        meta_info['transform'] = 'val'
        val_transformations = get_val_transformations(meta_info)
        meta_info['mode'] = 'test'
        test_dataset = get_test_dataset(args, val_transformations, meta_info=meta_info)
        test_dataloader = get_test_dataloader(args, test_dataset)
        return test_dataloader

    elif mode == 'train':
        meta_info['transform'] = 'train'
        train_transformations_normal = get_train_transformations(meta_info)
        meta_info['transform'] = args.aug
        train_transformations_aug = get_train_transformations(meta_info)
        train_transformations = {'standard': train_transformations_normal, 'augment': train_transformations_aug}
        meta_info['mode'] = 'labeled'
        labeled_dataset = get_train_dataset(args, train_transformations, meta_info=meta_info)
        labeled_dataloader = get_train_dataloader(args, labeled_dataset)
        meta_info['mode'] = 'unlabeled'
        unlabeled_dataset = get_train_dataset(args, train_transformations, meta_info=meta_info)
        unlabeled_dataloader = get_train_dataloader(args, unlabeled_dataset)
        return labeled_dataloader, unlabeled_dataloader

    elif mode == 'eval_train':
        meta_info['transform'] = 'val'
        eval_transformations = get_val_transformations(meta_info)
        meta_info['mode'] = 'eval'
        eval_dataset = get_train_dataset(args, eval_transformations, meta_info=meta_info)
        eval_dataloader = get_val_dataloader(args, eval_dataset)
        return eval_dataloader

    elif mode == 'warmup':
        # clothing1m uses strong aug when warmup
        meta_info['transform'] = args.aug if args.dataset == 'clothing1m' else 'train'
        warmup_transformations_normal = get_train_transformations(meta_info)
        meta_info['transform'] = args.aug
        warmup_transformations_aug = get_train_transformations(meta_info)
        warmup_transformations = {'standard': warmup_transformations_normal, 'augment': warmup_transformations_aug}
        meta_info['mode'] = 'all'
        warmup_dataset = get_train_dataset(args, warmup_transformations, meta_info=meta_info)
        # bs *= 2: in uniform_proto_train, we concatenate inputs_x and inputs_u at dim=0,
        # and in info_nce_loss, we keep its bs=2*args.bs for consistency
        warmup_dataloader = get_train_dataloader(args, warmup_dataset, explicit_batch_size=args.batch_size * 2)
        return warmup_dataloader

    else:
        raise NotImplementedError
