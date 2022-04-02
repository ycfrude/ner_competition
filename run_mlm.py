# author: 
# contact: ycfrude@163.com
# datetime:2022/4/2 3:31 PM

from src.config import Config
from src.pretrain.pl_mlm import train

if __name__ == "__main__":
    train_path = "train_data/train.txt"
    with open(train_path, "r") as f :
        train_data = f.read().splitlines()
    params = {
        "num_epochs":20,
        "learning_rate":1e-5,
        "batch_size":64,
    }
    config = Config(params)
    train(train_data, config, gpus=[1])