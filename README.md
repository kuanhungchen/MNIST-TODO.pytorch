_Copied from potterhsu_
# MNIST-TODO.pytorch


## Requirements

* Python 3.6
* PyTorch 0.4.1


## Usage

1. Train
    ```
    $ python train.py -d=./data -c=./checkpoints
    ```

1. Evaluate
    ```
    $ python eval.py ./checkpoint/model-100.pth -d=./data
    ```

1. Infer
    ```
    $ python infer.py ./images/7.jpg -c=./checkpoints/model-100.pth
    ```
