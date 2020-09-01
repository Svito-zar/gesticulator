import os
import sys

import numpy as np
import torch
from argparse import ArgumentParser

from config.model_config import construct_model_config_parser
from gesticulator.model.model import GesticulatorModel
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks.base import Callback

from visualization.motion_visualizer.generate_videos import generate_videos
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

class ModelSavingCallback(Callback):
    """ 
    Saves the model to the <results>/<run_name> directory during training.
    The saving frequency is configured by the --save_model_every_n_epochs command-line argument.

    The model can be loaded from the checkpoints with:
        model = GesticulatorModel.load_from_checkpoint(<checkpoint_path>)
    """
    def on_validation_end(self, trainer, model):
        if trainer.current_epoch % model.hparams.save_model_every_n_epochs == 0:
            checkpoint_fname = f"model_ep{model.current_epoch}.ckpt"
            checkpoint_dir = os.path.abspath(model.save_dir)

            trainer.save_checkpoint(os.path.join(model.save_dir, checkpoint_fname))
            print("\n\n  Saved checkpoint to", os.path.join(checkpoint_dir, checkpoint_fname), end="\n")

def main(hparams):
    model = GesticulatorModel(hparams)
    logger = create_logger(model.save_dir)
    callbacks = [ModelSavingCallback()] if hparams.save_model_every_n_epochs > 0 else []
    
    trainer = Trainer.from_argparse_args(hparams, logger=logger, callbacks = callbacks,
        checkpoint_callback=False, early_stop_callback=False)

    trainer.fit(model)
    trainer.save_checkpoint(os.path.join(model.save_dir, "trained_model.ckpt"))

   
def create_logger(model_save_dir):
    # str.rpartition(separator) cuts up the string into a 3-tuple of (a,b,c), where
    #   a: everything before the last occurrence of the separator
    #   b: the separator
    #   c: everything after the last occurrence of the separator)
    result_dir, _, run_name = model_save_dir.rpartition('/')
    
    return TensorBoardLogger(save_dir=result_dir, version=run_name, name="")

def add_training_script_arguments(parser):
    parser.add_argument('--save_model_every_n_epochs', '-ckpt_freq', type=int, default=0,
                        help="The frequency of model checkpoint saving.")
    return parser

if __name__ == '__main__':
    # Model parameters are added here
    parser = construct_model_config_parser()
    
    # Add training-script specific parameters
    parser = add_training_script_arguments(parser) 

    hyperparams = parser.parse_args()
    main(hyperparams)

