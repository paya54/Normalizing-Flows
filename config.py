import os

output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

class Config:
    def __init__(self):
        self.x_dim = 2
        self.hidden_dim = 40
        self.window_size = 4
        self.z_dim = 1
        self.flow_num = 4
        self.flow_type = 'planar'
        self.batch_size = 100
        self.epoch_num = 1000
