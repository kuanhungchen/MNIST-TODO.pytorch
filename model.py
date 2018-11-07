import os
import time

import torch
import torch.nn.functional
from torch import nn, Tensor


class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self._features = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self._classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 10)
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self._features(images)
        features = features.view(features.shape[0], -1)
        logits = self._classifier(features)
        return logits

    def loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        cross_entropy = torch.nn.functional.cross_entropy(logits, labels)
        return cross_entropy

    def save(self, path_to_checkpoints_dir: str, step: int) -> str:
        path_to_checkpoint = os.path.join(path_to_checkpoints_dir,
                                          'model-{:s}-{:d}.pth'.format(time.strftime('%Y%m%d%H%M'), step))
        torch.save(self.state_dict(), path_to_checkpoint)
        return path_to_checkpoint

    def load(self, path_to_checkpoint: str) -> 'Model':
        self.load_state_dict(torch.load(path_to_checkpoint))
        return self
