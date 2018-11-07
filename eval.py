import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from model import Model


def _eval(path_to_checkpoint: str, path_to_data_dir: str):
    dataset = Dataset(path_to_data_dir, Dataset.Mode.TEST)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = Model().cuda()
    model.load(path_to_checkpoint)

    num_hits = 0

    print('Start evaluating')

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.cuda()
            labels = labels.cuda()

            logits = model.eval().forward(images)
            _, predictions = logits.max(dim=1)
            num_hits += (predictions == labels).sum().item()

        accuracy = num_hits / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir

        _eval(path_to_checkpoint, path_to_data_dir)

    main()
