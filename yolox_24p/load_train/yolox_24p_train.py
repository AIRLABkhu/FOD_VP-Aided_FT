import os
import torch.distributed as dist
from exp import Exp as MyExp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        
        self.num_classes = 80
        self.max_epoch = 300
        self.L1_epoch = 100
        self.data_num_workers = 8
        # self.eval_interval = 1
        self.exp_name = "dark_sc"

    