import argparse
import os
import time
from collections import deque

import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model
from config import Config


def _train(path_to_data_dir: str, path_to_checkpoints_dir: str):
    os.makedirs(path_to_checkpoints_dir, exist_ok=True)

    dataset = Dataset(path_to_data_dir, mode=Dataset.Mode.TRAIN)
    dataloader = DataLoader(dataset, batch_size=Config.Batch_Size, shuffle=True)

    # TODO: CODE START
    # raise NotImplementedError
    model = Model()
    # model.load('checkpoints/model-201811091709-100000.pth')
    if Config.Device == 'gpu':
        model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=Config.Learning_Rate)
    # TODO: CODE END

    # num_steps_to_display = 20
    # num_steps_to_snapshot = 1000
    # num_steps_to_finish = 10000

    step = 0+100000
    time_checkpoint = time.time()
    losses = deque(maxlen=100)
    should_stop = False

    print('Start training')

    while not should_stop:
        for batch_idx, (images, labels) in enumerate(dataloader):
            if Config.Device == 'gpu':
                images = images.cuda()
                labels = labels.cuda()

            # TODO: CODE START
            # raise NotImplementedError
            logits = model.train().forward(images)
            loss = model.loss(logits, labels)
            # TODO: CODE END

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1

            if step % Config.EveryStepsToDisplay == 0:
                elapsed_time = time.time() - time_checkpoint
                time_checkpoint = time.time()
                steps_per_sec = Config.EveryStepsToDisplay / elapsed_time
                avg_loss = sum(losses) / len(losses)
                print(f'[Step {step}] Avg. Loss = {avg_loss:.6f} ({steps_per_sec:.2f} steps/sec)')

            if step % Config.EveryStepsToSnapshot == 0:
                path_to_checkpoint = model.save(path_to_checkpoints_dir, step)
                print(f'Model saved to {path_to_checkpoint}')

            #if step == 40000:
            #    optimizer = optim.Adam(model.parameters(), lr=Config.Learning_Rate/2)
            #    print(f'Learning rate changed to {Config.Learning_Rate/2}')

            #if step == 80000:
            #    optimizer = optim.Adam(model.parameters(), lr=Config.Learning_Rate/4)
            #    print(f'Learning rate changed to {Config.Learning_Rate/4}')

            # if step % Config.EveryStepsToDecay == 0:
            #     _new_learning_rate = Config.Learning_Rate / (1 + step/Config.EveryStepsToDecay)
            #     optimizer = optim.Adam(model.parameters(), lr=_new_learning_rate)
            #     print(f'Learning rate changed to {_new_learning_rate}')

            if step == Config.EveryStepsToFinish:
                should_stop = True
                break

    print('Done')


if __name__ == '__main__':
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data_dir', default='./data', help='path to data directory')
        parser.add_argument('-c', '--checkpoints_dir', default='./checkpoints', help='path to checkpoints directory')
        args = parser.parse_args()

        path_to_data_dir = args.data_dir
        path_to_checkpoints_dir = args.checkpoints_dir

        _train(path_to_data_dir, path_to_checkpoints_dir)

    main()
