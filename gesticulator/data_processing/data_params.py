"""
Argument parsers for split_dataset.py and process_dataset.py.

By default, we assume that the dataset is found in the <repo>/dataset/raw/ folder,
and the preprocessed datasets will be created in the <repo>/dataset/processed/ folder.
These paths can be changed with the arguments below.
"""
import argparse

# Dataset folder arguments
dataset_argparser = argparse.ArgumentParser(
    description="Paths to the Trinity Speech-Gesture dataset and the processed output folder."
)

dataset_argparser.add_argument('--raw_data_dir', '-data_raw', default="../../dataset/raw_data/",
                help='Path to the folder with the raw dataset')
dataset_argparser.add_argument('--proc_data_dir', '-data_proc', default="../../dataset/processed_data/",
                help='Path to the folder with the processed dataset')

# -------------------------------------------------------------------------------------------------


# Data processing arguments
processing_argparser = argparse.ArgumentParser(
    description="""Parameters for data processing for the paper `Gesticulator: 
                   A framework for semantically-aware speech-driven gesture generation""",
    parents=[dataset_argparser], add_help=False) # NOTE: we include the dataset_parser here as a parent!

# Sequence processing
processing_argparser.add_argument('--seq_len', '-seq_l', default=40,
                    help='Length of the sequences during training (used only to avoid vanishing gradients)')
processing_argparser.add_argument('--past_context', '-p_cont', default=10, type=int,
                    help='Length of a past context for speech to be used for gestures')
processing_argparser.add_argument('--future_context', '-f_cont', default=20, type=int,
                    help='Length of a future context for speech to be used for gestures')

# Features
processing_argparser.add_argument('--text_embedding', '-embed', default="BERT",
                    help='The text embedding method to use (can be "BERT" or "FastText"),'
                         ' but FastText is currently disabled')

processing_argparser.add_argument('--feature_type', '-feat', default="Spectro",
                    help="""Describes the type of the input features 
                            (can be 'Spectro', 'MFCC', 'Pros', 'MFCC+Pros' or 'Spectro+Pos')""")