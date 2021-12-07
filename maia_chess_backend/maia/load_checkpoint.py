import os

import numpy as np
np.set_printoptions(precision=3,linewidth=150,threshold=10)

import tensorflow as tf
import yaml
from tfprocess import TFProcess
from net import Net

from tensorflow import keras
from tensorflow.keras.utils import plot_model

from lcztools import LeelaBoard

print(tf.config.list_physical_devices())

def getmodel(level):
    pb_file = f'../../model_files/{level}/final_{level}-40.pb.gz'
    cfg_file = '../../maia_config.yaml'

    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    net = Net()
    net.parse_proto(pb_file)
    
    filters, blocks = net.filters(), net.blocks()
    if cfg['model']['filters'] != filters:
        raise ValueError("Number of filters in YAML doesn't match the network")
    if cfg['model']['residual_blocks'] != blocks:
        raise ValueError("Number of blocks in YAML doesn't match the network")

    weights = net.get_weights()
    len(weights)

    tfp = TFProcess(cfg, '', '')
    tfp.init_net_v2()
    tfp.replace_weights_v2(weights)
    return tfp.model

model = getmodel('1800')
print(model)

board = LeelaBoard()
x = board.lcz_features().reshape(112,64)
print('x')
print(x.shape)
x = np.array((x,)*64)
print('x')
print(x.shape)

y = model.predict(x, batch_size=64)[0]
print('y')
print(y.shape)
y = y[0,:]

print('y')
print(y.shape)

m = np.argmax(y)
print('m')
print(m)



