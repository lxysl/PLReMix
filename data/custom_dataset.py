import os
import json
import random
import numpy as np
from torch.utils.data import Dataset

from data.asymmetric_noise import noisify_cifar100_asymmetric, noisify


class NoisyDataset(Dataset):
    def __init__(self, dataset, meta_info):
        super(NoisyDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None

        if isinstance(transform, dict):
            self.transform = transform['standard']
            self.augmentation_transform = transform['augment']
        else:
            self.transform = transform

        self.dataset = dataset
        self.probability = meta_info['probability']
        self.pred_clean = meta_info['pred_clean']
        self.pred_noisy = meta_info['pred_noisy']
        self.mode = meta_info['mode']
        # class transition for asymmetric noise
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

        r = meta_info['r']
        num_classes = meta_info['num_classes']
        noise_file = meta_info['noise_file']
        noise_mode = meta_info['noise_mode']
        dataset_name = meta_info['dataset']

        if os.path.exists(noise_file):
            noisy_labels = json.load(open(noise_file, "r"))
        else:
            # inject noise
            noisy_labels = []
            idx = list(range(len(dataset)))
            random.shuffle(idx)
            num_noise = int(r * len(dataset))
            noise_idx = idx[:num_noise]
            if noise_mode == 'asym' and dataset_name == 'cifar100':
                noisy_labels, _ = noisify_cifar100_asymmetric(self.dataset.targets, r)
                noisy_labels = noisy_labels.tolist()
            elif noise_mode == 'asym' and dataset_name == 'tiny_imagenet':
                noisy_label, _ = noisify('tiny_imagenet', num_classes, np.array(self.dataset.targets),
                                         'pair_flip', r, 0)
                noisy_labels = noisy_label.tolist()
            else:
                for i in range(len(dataset)):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            noisy_label = None
                            if dataset_name == 'cifar10':
                                noisy_label = random.randint(0, 9)
                            elif dataset_name == 'cifar100':
                                noisy_label = random.randint(0, 99)
                            elif dataset_name == 'tiny_imagenet':
                                noisy_label = random.randint(0, 199)
                            noisy_labels.append(noisy_label)
                        elif noise_mode == 'asym':
                            noisy_label = self.transition[self.dataset.targets[i]]
                            noisy_labels.append(noisy_label)
                    else:
                        noisy_labels.append(dataset.targets[i])
            print("save noisy labels to %s ..." % noise_file)
            if not os.path.exists(os.path.dirname(noise_file)):
                os.makedirs(os.path.dirname(noise_file))
            json.dump(noisy_labels, open(noise_file, "w"))
        self.noise_labels = noisy_labels

        self.labels = noisy_labels

        if self.mode == 'labeled':
            self.indices = self.pred_clean
        elif self.mode == 'unlabeled':
            self.indices = self.pred_noisy
        elif self.mode in ['eval', 'all']:
            self.indices = list(range(len(self.dataset)))
        else:
            raise ValueError('Invalid noisy dataset mode')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        true_index = self.indices[index]
        image = self.dataset.__getitem__(true_index)
        if self.mode == 'labeled':
            img, target, prob = image, self.labels[true_index], self.probability[true_index]
            img = img['image']
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.augmentation_transform(img)
            img4 = self.augmentation_transform(img)
            return img1, img2, img3, img4, target, prob, true_index
        elif self.mode == 'unlabeled':
            img = image['image']
            target = self.labels[true_index]
            img1 = self.transform(img)
            img2 = self.transform(img)
            img3 = self.augmentation_transform(img)
            img4 = self.augmentation_transform(img)
            return img1, img2, img3, img4, target, true_index
        elif self.mode == 'eval':
            img, target = image, self.labels[true_index]
            img = img['image']
            img = self.transform(img)
            return img, target, true_index
        elif self.mode == 'all':
            img, target = image, self.labels[true_index]
            img = img['image']
            img1 = self.transform(img)
            img2 = self.augmentation_transform(img)
            img3 = self.augmentation_transform(img)
            return img1, img2, img3, target, true_index
