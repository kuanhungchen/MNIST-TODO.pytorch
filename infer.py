import argparse

import torch
from PIL import Image

from dataset import Dataset
from model import Model


def _infer(path_to_image: str, path_to_checkpoint: str):
    image = Image.open(path_to_image)
    image = Dataset.preprocess(image)

    model = Model().cuda()
    model.load(path_to_checkpoint)

    print('Start inferring')

    with torch.no_grad():
        images = image.unsqueeze(dim=0).cuda()
        logits = model.eval().forward(images)
        _, predictions = logits.max(dim=1)
        prediction = predictions.item()
        print(f'Prediction = {prediction:d}')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('image', type=str, help='path to image')
        parser.add_argument('-c', '--checkpoint', help='path to checkpoint')
        args = parser.parse_args()

        path_to_image = args.image
        path_to_checkpoint = args.checkpoint

        _infer(path_to_image, path_to_checkpoint)

    main()
