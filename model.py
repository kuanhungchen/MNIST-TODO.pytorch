import os
import time

import numpy as np
import torch
import torch.nn.functional
from torch import nn, Tensor


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # TODO: CODE START
        # raise NotImplementedError
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*7*7, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 10),
        )
        # TODO: CODE END

    def forward(self, images: Tensor) -> Tensor:
        # TODO: CODE START
        # raise NotImplementedError
        x = self.network(images)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
        # TODO: CODE END

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        # TODO: CODE START
        # raise NotImplementedError
        loss_function = nn.CrossEntropyLoss()
        return loss_function(logits, labels)
        # TODO: CODE END

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
