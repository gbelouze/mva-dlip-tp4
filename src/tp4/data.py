import pandas as pd
import torch
import torchvision as V
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import BatchSampler, DataLoader, WeightedRandomSampler


class TinyCIFAR(V.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(TinyCIFAR, self).__init__(*args, **kwargs)
        self.data = self.data[:100]
        self.targets = self.targets[:100]


class CIFARnoLabel(V.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CIFARnoLabel, self).__init__(*args, **kwargs)
        self.data = self.data[100:]
        self.targets = self.targets[100:]

    def __getitem__(self, index):
        x, _ = super(CIFARnoLabel, self).__getitem__(index)
        return x


def balancing_weights(train_set):
    count = pd.Series([y for _, y in train_set]).value_counts()
    class_weights = 1 / torch.Tensor(
        [count[class_index] for class_index in range(len(count))]
    )
    return [class_weights[class_index] for _, class_index in train_set]


def balancedDataLoader(train_set, batch_size):
    sampler = WeightedRandomSampler(balancing_weights(train_set), len(train_set))
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    return DataLoader(train_set, batch_sampler=batch_sampler)


def RandomColorJitter(p=0.3, *args, **kwargs):
    return T.RandomApply(torch.nn.ModuleList([T.ColorJitter(*args, **kwargs)]), p)


transforms = {
    "minimal": T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "auto": T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "geometric": T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                interpolation=TF.InterpolationMode.BILINEAR,
            ),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
    "colour": T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(p=0.1),
            T.RandomPosterize(bits=6, p=0.2),
            T.RandomAutocontrast(p=0.3),
            T.RandomEqualize(p=0.2),
            T.ToTensor(),
            RandomColorJitter(
                p=0.3, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            ),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "colour_and_geometric": T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(p=0.1),
            T.RandomPosterize(bits=6, p=0.2),
            T.RandomAutocontrast(p=0.3),
            T.RandomEqualize(p=0.2),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            RandomColorJitter(
                p=0.3, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            T.RandomHorizontalFlip(),
            T.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.95, 1.05),
                interpolation=TF.InterpolationMode.BILINEAR,
            ),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
}
