import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from model import Model
from config import Config


def _eval(path_to_checkpoint: str, path_to_data_dir: str, path_to_results_dir: str):
    os.makedirs(path_to_results_dir, exist_ok=True)

    dataset = Dataset(path_to_data_dir, Dataset.Mode.TEST)
    dataloader = DataLoader(dataset, batch_size=Config.Batch_Size, shuffle=False)

    model = Model()
    if Config.Device == 'gpu':
        model = model.cuda()

    model.load(path_to_checkpoint)
    num_hits = 0

    print('Start evaluating')

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(tqdm(dataloader)):
            if Config.Device == 'gpu':
                images = images.cuda()
                labels = labels.cuda()

            logits = model.eval().forward(images)
            _, predictions = logits.max(dim=1)
            num_hits += (predictions == labels).sum().item()

        accuracy = num_hits / len(dataset)
        print(f'Accuracy = {accuracy:.4f}')

    with open(os.path.join(path_to_results_dir, 'accuracy.txt'), 'a') as fp:
        fp.write(f'{accuracy:.4f}\n')
    with open(os.path.join(path_to_results_dir, 'detail.txt'), 'a') as fp:
        fp.write(f'{path_to_checkpoint}: Acc={accuracy:.4f}\n')

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g.: ./checkpoints/model-100.pth')
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-r', '--results_dir', default='./results', help='path to results directory')
        args = parser.parse_args()

        path_to_checkpoint = args.checkpoint
        path_to_data_dir = args.data_dir
        path_to_results_dir = args.results_dir
        # path_to_checkpoint = './checkpoints/model-201811091646-10000.pth'
        _eval(path_to_checkpoint, path_to_data_dir, path_to_results_dir)

    main()
