import os
import random
from PIL import Image
from torch.utils.data import Dataset


class ImageNet(Dataset):
    def __init__(self, root, transform, meta_info):
        self.root = os.path.join(root, 'imagenet/val/')
        if isinstance(transform, dict):
            self.transform = transform['standard']
            self.augmentation_transform = transform['augment']
        else:
            self.transform = transform

        labels_webvision = {}
        with open(os.path.join(root, 'info/synsets.txt')) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                label_info = line.split()
                labels_webvision[i] = label_info[0]

        self.val_data = []
        for c in range(meta_info['num_classes']):
            imgs = os.listdir(os.path.join(self.root, labels_webvision[c]))
            for img in imgs:
                self.val_data.append([c, os.path.join(self.root, labels_webvision[c], img)])

    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return {'image': img, 'target': target}

    def __len__(self):
        return len(self.val_data)


class WebVision(Dataset):
    def __init__(self, root, transform, meta_info):
        self.root = root
        if isinstance(transform, dict):
            self.transform = transform['standard']
            self.augmentation_transform = transform['augment']
        else:
            self.transform = transform
        self.mode = meta_info['mode']
        self.num_classes = meta_info['num_classes']
        self.multi_crop = meta_info['multi_crop']
        self.targets = []

        if self.mode in ['val', 'test']:
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < self.num_classes:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        else:
            with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
                lines = f.readlines()
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < self.num_classes:
                    train_imgs.append(img)
                    self.train_labels[img] = target
                    self.targets.append(target)

            if self.mode == "labeled":
                train_imgs = meta_info['paths']
                self.train_imgs = [train_imgs[i] for i in meta_info['pred_clean']]
                self.probability = [meta_info['probability'][i] for i in meta_info['pred_clean']]
                print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
            elif self.mode == "unlabeled":
                train_imgs = meta_info['paths']
                self.train_imgs = [train_imgs[i] for i in meta_info['pred_noisy']]
                self.probability = [meta_info['probability'][i] for i in meta_info['pred_noisy']]
                print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
            elif self.mode in ["eval", "all"]:
                self.train_imgs = train_imgs
                random.shuffle(self.train_imgs)

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            img3 = self.augmentation_transform[0](image) if self.multi_crop else self.augmentation_transform(image)
            img4 = self.augmentation_transform[1](image) if self.multi_crop else self.augmentation_transform(image)
            if self.multi_crop:
                small_imgs = [transform(image) for transform in self.augmentation_transform[2:]]
                return img1, img2, img3, img4, small_imgs, target, prob, index
            else:
                return img1, img2, img3, img4, target, prob, index
        elif self.mode == 'unlabeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img1 = self.transform(image)
            img2 = self.transform(image)
            img3 = self.augmentation_transform[0](image) if self.multi_crop else self.augmentation_transform(image)
            img4 = self.augmentation_transform[1](image) if self.multi_crop else self.augmentation_transform(image)
            if self.multi_crop:
                small_imgs = [transform(image) for transform in self.augmentation_transform[2:]]
                return img1, img2, img3, img4, small_imgs, target, index
            else:
                return img1, img2, img3, img4, target, index
        elif self.mode == 'eval':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img = self.transform(image)
            return img, target, {'index': index, 'path': img_path}
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img1 = self.transform(image)
            img2 = self.augmentation_transform[0](image) if self.multi_crop else self.augmentation_transform(image)
            img3 = self.augmentation_transform[1](image) if self.multi_crop else self.augmentation_transform(image)
            if self.multi_crop:
                small_imgs = [transform(image) for transform in self.augmentation_transform[2:]]
                return img1, img2, img3, small_imgs, target, {'index': index, 'path': img_path}
            else:
                return img1, img2, img3, target, {'index': index, 'path': img_path}
        elif self.mode in ['val', 'test']:
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(os.path.join(self.root, 'val_images_256', img_path)).convert('RGB')
            img = self.transform(image)
            return {'image': img, 'target': target}

    def __len__(self):
        if self.mode in ['val', 'test']:
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)
