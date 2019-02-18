# Tensorboard writer
import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
class TensorboardWriter():
    ''' Tensorboard Writer '''
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
    def write(self, result, n_iter):
        for metric in result:
            self.writer.add_scalar(metric, result[metric], n_iter)
    def export(self):
        json_path = os.path.join(self.log_dir, 'results.json')
        self.writer.export_scalars_to_json(json_path)
    def close(self):
        self.writer.close()
