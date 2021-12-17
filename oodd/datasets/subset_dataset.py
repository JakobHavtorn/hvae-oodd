from torch.utils.data import Subset
import numpy as np

# TODO: will it be circular?
from oodd.datasets import CIFAR10Dequantized, CIFAR100Dequantized

CIFAR10_ANIMALS = [2,3,4,5,6,7]
CIFAR10_TRANSPORTATION = [0,1,8,9]

CIFAR100_MAPPING = {
    'aquatic mammals': [4, 30, 55, 72, 95],
    'fish': [1, 32, 67, 73, 91],
    'flowers': [54, 62, 70, 82, 92],
    'food containers': [9, 10, 16, 28, 61],
    'fruit and vegetables': [0, 51, 53, 57, 83],
    'household electrical device': [22, 39, 40, 86, 87],
    'household furniture': [5, 20, 25, 84, 94],
    'insects': [6, 7, 14, 18, 24],
    'large carnivores': [3, 42, 43, 88, 97],
    'large man-made outdoor things': [12, 17, 37, 68, 76],
    'large natural outdoor scenes': [23, 33, 49, 60, 71],
    'large omnivores and herbivores': [15, 19, 21, 31, 38],
    'medium-sized mammals': [34, 63, 64, 66, 75],
    'non-insect invertebrates': [26, 45, 77, 79, 99],
    'people': [2, 11, 35, 46, 98],
    'reptiles': [27, 29, 44, 78, 93],
    'small mammals': [36, 50, 65, 74, 80],
    'trees': [47, 52, 56, 59, 96],
    'vehicles 1': [8, 13, 48, 58, 90],
    'vehicles 2': [41, 69, 81, 85, 89]
}

def get_cifar100_num(groups):
    return sum([CIFAR100_MAPPING[g] for g in groups], [])

CIFAR100_ANIMALS = get_cifar100_num([
    'aquatic mammals',
    'small mammals',
    'medium-sized mammals',
    'reptiles',
    'insects',
    'non-insect invertebrates',
    'fish',
    'large carnivores',
    'large omnivores and herbivores'
])

CIFAR100_PLANTS = get_cifar100_num([
    'trees', 'flowers', 'fruit and vegetables'
])

CIFAR100_THINGS = get_cifar100_num([
'large man-made outdoor things',
'vehicles 1',
'vehicles 2',
'household electrical device',
'household furniture',
'food containers'
])

CIFAR100_REST = get_cifar100_num([
    'large natural outdoor scenes',
    'people'
])

def make_subset(dataset, targets):
    dataset.dataset = Subset(dataset.dataset, np.where(np.isin(dataset.dataset.targets, targets))[0])

class CIFAR10DequantizedSubset(CIFAR10Dequantized):
    _subset_labels = [0]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        make_subset(self, self._subset_labels)


class CIFAR100DequantizedSubset(CIFAR100Dequantized):
    _subset_labels = [0]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        make_subset(self, self._subset_labels)


class CIFAR10DequantizedAnimals(CIFAR10DequantizedSubset):
    _subset_labels = CIFAR10_ANIMALS

class CIFAR10DequantizedTransportation(CIFAR10DequantizedSubset):
    _subset_labels = CIFAR10_TRANSPORTATION

class CIFAR10DequantizedFirstHalf(CIFAR10DequantizedSubset):
    _subset_labels = [0,1,2,3,4]

class CIFAR10DequantizedSecondHalf(CIFAR10DequantizedSubset):
    _subset_labels = [5,6,7,8,9]

class CIFAR10DequantizedOdd(CIFAR10DequantizedSubset):
    _subset_labels = [1,3,5,7,9]

class CIFAR10DequantizedEven(CIFAR10DequantizedSubset):
    _subset_labels = [0,2,4,6,8]


class CIFAR100DequantizedAnimals(CIFAR100DequantizedSubset):
    _subset_labels = CIFAR100_ANIMALS

class CIFAR100DequantizedPlants(CIFAR100DequantizedSubset):
    _subset_labels = CIFAR100_PLANTS

class CIFAR100DequantizedThings(CIFAR100DequantizedSubset):
    _subset_labels = CIFAR100_THINGS

class CIFAR100DequantizedRest(CIFAR100DequantizedSubset):
    _subset_labels = CIFAR100_REST

class CIFAR100DequantizedFirstHalf(CIFAR100DequantizedSubset):
    _subset_labels = list(range(50))

class CIFAR100DequantizedSecondHalf(CIFAR100DequantizedSubset):
    _subset_labels = list(range(50,100))

class CIFAR100DequantizedOdd(CIFAR100DequantizedSubset):
    _subset_labels = list(range(1,100, 2))

class CIFAR100DequantizedEven(CIFAR100DequantizedSubset):
    _subset_labels = list(range(0,100, 2))