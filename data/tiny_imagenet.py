import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, cast, Dict, List, Optional, Tuple

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    See :class:`DatasetFolder` for details.
    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)
    cls, class_to_idx = find_classes(directory)
    # print(cls,class_to_idx)
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, f_names in sorted(os.walk(target_dir, followlinks=True)):
            for f_name in sorted(f_names):
                if is_valid_file(f_name):
                    path = os.path.join(root, f_name)
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances, class_to_idx


class TinyImageNet(Dataset):
    def __init__(self, root='./data/tiny-imagenet-200', train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        # Get the instances and check if it is right
        data_folder = os.path.join(self.root, 'train')
        train_instances, dict_classes = make_dataset(data_folder, extensions=IMG_EXTENSIONS)

        # Validation Files
        data_folder = os.path.join(self.root, 'val')
        val_text = './data/tiny-imagenet-200/val/val_annotations.txt'
        val_img_files = './data/tiny-imagenet-200/val/'
        data_folder = './data/tiny-imagenet-200/test/'

        # Load these instances->(data, label) into custom dataloader
        self.label_set = {}
        self.targets = []
        self.data = []

        if self.train:
            for l in range(len(train_instances)):
                img_path, label = list(train_instances[l])
                self.label_set[img_path] = int(label)
                self.targets.append(int(label))
                self.data.append(img_path)
        else:
            with open(val_text, 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    path, label_class = l.split()[0], l.split()[1]
                    img_path = os.path.join(val_img_files, label_class, path)
                    self.label_set[img_path] = int(dict_classes[label_class])
                    self.targets.append(int(dict_classes[label_class]))
                    self.data.append(img_path)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'index': index}}

        return out

    def __len__(self):
        return len(self.data)
