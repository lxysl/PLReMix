import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset


class Clothing1M(Dataset):
    def __init__(self, root, transform, meta_info):
        self.root = root
        self.mode = meta_info['mode']
        if isinstance(transform, dict):
            self.transform = transform['standard']
            self.augmentation_transform = transform['augment']
        else:
            self.transform = transform
        self.multi_crop = meta_info['multi_crop']
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        self.targets = []
        num_samples = 1000 * 64
        num_classes = meta_info['num_classes']

        with open(os.path.join(self.root, 'noisy_label_kv.txt'), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                entry = line.split()
                img_path = os.path.join(self.root, entry[0])
                self.train_labels[img_path] = int(entry[1])
                self.targets.append(int(entry[1]))
        with open(os.path.join(self.root, 'clean_label_kv.txt'), 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                entry = line.split()
                img_path = os.path.join(self.root, entry[0])
                self.test_labels[img_path] = int(entry[1])

        if self.mode in ['eval', 'all']:
            train_imgs = []
            with open(os.path.join(self.root, 'noisy_train_key_list.txt'), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    img_path = os.path.join(self.root, line)
                    train_imgs.append(img_path)
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_classes)
            self.train_imgs = []
            for im_path in train_imgs:
                label = self.train_labels[im_path]
                if class_num[label] < (num_samples / 14) and len(self.train_imgs) < num_samples:
                    self.train_imgs.append(im_path)
                    class_num[label] += 1
            random.shuffle(self.train_imgs)
        elif self.mode == 'labeled':
            train_imgs = meta_info['paths']
            self.train_imgs = [train_imgs[i] for i in meta_info['pred_clean']]
            self.probability = [meta_info['probability'][i] for i in meta_info['pred_clean']]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
        elif self.mode == 'unlabeled':
            train_imgs = meta_info['paths']
            self.train_imgs = [train_imgs[i] for i in meta_info['pred_noisy']]
            self.probability = [meta_info['probability'][i] for i in meta_info['pred_noisy']]
            print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))
        elif self.mode == 'val':
            self.val_imgs = []
            with open(os.path.join(self.root, 'clean_val_key_list.txt'), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    img_path = os.path.join(self.root, line)
                    self.val_imgs.append(img_path)
        elif self.mode == 'test':
            self.test_imgs = []
            with open(os.path.join(self.root, 'clean_test_key_list.txt'), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    img_path = os.path.join(self.root, line)
                    self.test_imgs.append(img_path)
        else:
            raise ValueError('Invalid noisy dataset mode')

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
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
            image = Image.open(img_path).convert('RGB')
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
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, {'index': index, 'path': img_path}
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            if isinstance(self.transform, list):
                img1 = self.transform[0](image)
            else:
                img1 = self.transform(image)
            img2 = self.augmentation_transform[0](image) if self.multi_crop else self.augmentation_transform(image)
            img3 = self.augmentation_transform[1](image) if self.multi_crop else self.augmentation_transform(image)
            if self.multi_crop:
                small_imgs = [transform(image) for transform in self.augmentation_transform[2:]]
                return img1, img2, img3, small_imgs, target, {'index': index, 'path': img_path}
            else:
                return img1, img2, img3, target, {'index': index, 'path': img_path}
        elif self.mode == 'val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return {'image': img, 'target': target}
        elif self.mode == 'test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return {'image': img, 'target': target}

    def __len__(self):
        if self.mode == 'val':
            return len(self.val_imgs)
        elif self.mode == 'test':
            return len(self.test_imgs)
        else:
            return len(self.train_imgs)
