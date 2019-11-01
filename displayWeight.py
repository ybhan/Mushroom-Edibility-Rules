# By Jeff Yuanbo Han (u6617017), 2018-05-01.
import torch
import numpy as np
from load_data import attributes

w = torch.load('net_weights')['0.weight'][0,:].numpy()  # File 'net_weights' is derived by bpNN.py
w = w.astype(np.int)

valid_attr = {}

end = 0
for attr in attributes:
    if attr == 'class':
        continue
    start = end
    end = start + attributes[attr]
    if np.size(np.nonzero(w[start:end])):
        valid_attr[attr] = w[start:end]

for attr in valid_attr:
    print('{}: {}'.format(attr, valid_attr[attr]))
