import logging
import os.path
import shutil
import json
import torch
from dataclasses import dataclass


class RunningAverage:
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self, *args, **kwargs):
        return self.total / float(self.steps)


@dataclass
class params():
    cuda = torch.cuda.is_available()
    save_summary_steps = 1

    model_dir = './experiments/last_model'
    model_dir = './experiments/last_model'

    learning_rate = 0.01
    num_epochs = 120
    batch_size = 256
    l1_weight = 3e-5
    l2_weight = 7e-6


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    with open(log_path, 'a') as file:
        pass

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(
            checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    if not os.path.exists(checkpoint):
        raise f"File doesn't exist {format(checkpoint)}"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_dict_to_json(d, json_path):
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

