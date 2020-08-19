# This code was originally written by Simon Alexanderson (github.com/simonalexanderson),
# but the current version is taken from the following commit by Taras Kucherenko: 
# https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder/commit/daffa184779fb45c5a6ff1279a01c5e85b0eda37#diff-d95865ebf5428aa201840818a88d0fb5
""" 
This script converts joint angles back to the BVH format.
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pymo.parsers import BVHParser
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.writers import *

from argparse import ArgumentParser

import joblib as jl

def feat2bvh(feat_file, bvh_file):

    features = np.load(feat_file)
    print("Original features shape: ", features.shape)

    # shorten sequence length for visualization
    features = features[:1200]
    print("Shortened features shape: ", features.shape)

    # transform the data back to it's original shape
    # note: in a real scenario this is usually done with predicted data   
    # note: some transformations (such as transforming to joint positions) are not inversible
    bvh_data=pipeline.inverse_transform([features])

    # Test to write some of it to file for visualization in blender or motion builder
    writer = BVHWriter()
    with open(bvh_file,'w') as f:
        writer.write(bvh_data[0], f)

if __name__ == '__main__':

    # Setup parameter parser
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--feat_dir', '-feat', required=True,
                                   help="Path where motion features are stored")
    parser.add_argument('--bvh_dir', '-bvh', required=True,
                                   help="Path where produced motion files (in BVH format) will be stored")
    parser.add_argument('--pipeline_dir', '-pipe', default="./utils/",
                        help="Path where the motion data processing pipeline is be stored")

    params = parser.parse_args()


    # load data pipeline
    pipeline = jl.load(params.pipeline_dir + 'data_pipe.sav')

    # convert a file
    feat2bvh(params.feat_dir, params.bvh_dir)
