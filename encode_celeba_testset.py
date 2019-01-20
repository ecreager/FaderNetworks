# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import pdb

import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.image

from src.logger import create_logger
from src.loader import load_images, DataSampler
from src.utils import bool_flag
from src.my_loader import celeba_loader


# parse parameters
parser = argparse.ArgumentParser(description='Attributes swapping')
parser.add_argument("--model_path", type=str, default="",
                    help="Trained model path")
parser.add_argument("--n_images", type=int, default=10,
                    help="Number of images to modify")
parser.add_argument("--offset", type=int, default=0,
                    help="First image index")
parser.add_argument("--n_interpolations", type=int, default=10,
                    help="Number of interpolations per image")
parser.add_argument("--alpha_min", type=float, default=1,
                    help="Min interpolation value")
parser.add_argument("--alpha_max", type=float, default=1,
                    help="Max interpolation value")
parser.add_argument("--plot_size", type=int, default=5,
                    help="Size of images in the grid")
parser.add_argument("--row_wise", type=bool_flag, default=True,
                    help="Represent image interpolations horizontally")
parser.add_argument("--output_path", type=str, default="output.png",
                    help="Output path")
params = parser.parse_args()

# check parameters
print(params.model_path)
assert os.path.isfile(params.model_path)
output_dir = os.path.splitext(params.model_path)[0]
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
output_npz_filename = os.path.join(output_dir, 'encoded_test_set.npz')
assert params.n_images >= 1 and params.n_interpolations >= 2

# create logger / load trained model
logger = create_logger(None)
ae = torch.load(params.model_path).eval()

# restore main parameters
params.debug = True
params.batch_size = 32
params.v_flip = False
params.h_flip = False
params.img_sz = ae.img_sz
params.attr = ae.attr
params.n_attr = ae.n_attr
print(1, params.n_attr)
#if not (len(params.attr) == 1 and params.n_attr == 2):
    #raise Exception("The model must use a single boolean attribute only.")


output_dir = os.path.splitext(params.model_path)[0]
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        output_npz_filename = os.path.join(output_dir, 'encoded_test_set.npz')

# load dataset
#data, attributes = load_images(params)
#test_data = DataSampler(data[2], attributes[2], params)

use_cuda = True
test_loader = celeba_loader(256, 'test', True)

x_test = []
a_test = []
z_test = []
pbar = tqdm(range(len(test_loader)))
for x, a in test_loader:
    pbar.update()
    x_test.append(x)
    a_test.append(a)

    x = x.cuda(async=True)
    a = a.float()
    a = a.cuda(async=True)  

    z_all_layers = ae.encode(x)
    z = z_all_layers[-1].squeeze()
    #pdb.set_trace()
    z_test.append(z.detach().cpu())


x_test = torch.cat(x_test, 0).numpy()
a_test = torch.cat(a_test, 0).numpy()
z_test = torch.cat(z_test, 0).numpy()


np.savez(output_npz_filename, x=x_test, a=a_test, z=z_test, args=params)
print('done encoding test set:\n', output_npz_filename)

