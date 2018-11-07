from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
from PIL import Image
from torch import Tensor
from torchvision import datasets
from torchvision.transforms import transforms


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        is_train = mode == Dataset.Mode.TRAIN
        self._mnist = datasets.MNIST(path_to_data_dir, train=is_train, download=True)

    def __len__(self) -> int:
        # TODO: CODE START
        raise NotImplementedError
        # TODO: CODE END

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        # TODO: CODE START
        raise NotImplementedError
        # TODO: CODE END

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tensor:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image)
        return image
