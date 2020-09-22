import configargparse as cfgparse
import argparse
import os
from pytorch_lightning import Trainer

def construct_model_config_parser(add_trainer_args = True):
    """Construct the configuration parser for the Gesticulator model and (optionally) for the Trainer.
    
    The path to the config file must be provided with the -config option, e.g.
        'python train.py -config config/model_config.yaml'
    
    The parameter names with two dashes (e.g. --data_dir) can be used in the .yaml file,
    while the parameter names with a single dash (e.g. -data) are for the command line.

    Command line values override values found in the config file.
    """
    if add_trainer_args:
        # if we call Trainer.add_argparse_args() after creating the cfgparse.ArgumentParser,
        # then it will be upcasted to the base class argparse.ArgumentParser, and we would
        # lose some of the functionality (e.g. yaml config file reading)
        
        # therefore we pass the Trainer arguments to the model parser as a parent_parser
        # NOTE: for details, please visit the official Pytorch Lightning documentation for Trainer 
        trainer_parser = argparse.ArgumentParser(add_help = False)
        trainer_parser = Trainer.add_argparse_args(trainer_parser)
 
    parser = cfgparse.ArgumentParser(
                    config_file_parser_class = cfgparse.YAMLConfigFileParser, 
                    add_help = True, 
                    parents = [trainer_parser] if add_trainer_args else [])

    parser.add('--config', '-c', default='config/default_model_config.yaml',
               help='Path to the .yaml config file', is_config_file=True)

    # Directories
    parser.add('--data_dir',   '-data',  default='../dataset/processed',
               help='Path to a folder with the dataset')

    parser.add('--result_dir', '-res_d', default='../results',
               help='Path to the <results> directory, where all results are saved')

    parser.add('--run_name',   '-name',  default='last_run',
               help='Name of the subdirectory within <results> '
                    'where the results of this run will be saved')
    
    parser.add('--generated_gestures_dir', default=None,
               help="Path to the directory where final test gestures and the predicted validation or training gestures"
                    " will be saved (default: <results>/<run_name>/generated_gestures")

    parser.add('--saved_models_dir', '-model_d',  default=None,
               help='Path to the directory where models will be saved '
                    '(default: <results>/<run_name>/models/')

    # Data processing parameters
    parser.add('--sequence_length', '-seq_l',  default=40, type=int,
               help='Length of each training sequence')

    parser.add('--past_context',    '-p_cont', default=10, type=int,
               help='Length of past speech context to be used for generating gestures')

    parser.add('--future_context', '-f_cont',  default=20, type=int,
               help='Length of future speech context to be used for generating gestures')

    parser.add('--text_context',    '-txt_l',  default=10, type=int,
               help='Length of (future) text context to be used for generating gestures')

    parser.add('--speech_enc_frame_dim', '-speech_t_e', default=124, type=int,
               help='Dimensionality of the speech frame encoding')
    parser.add('--full_speech_enc_dim', '-speech_f_e',  default=612, type=int,
               help='Dimensionality of the full speech encoding')

    # Network architecture
    parser.add('--n_layers',        '-lay',      default=1, type=int, 
               choices=[1,2,3], help='Number of hidden layers (excluding RNN)')

    parser.add('--activation',      '-act',      default="TanH", 
               choices=["TanH", "LeakyReLU"], help='The activation function')

    parser.add('--first_l_sz',      '-first_l',  default=256, type=int,
               help='Dimensionality of the first layer')

    parser.add('--second_l_sz',     '-second_l', default=512, type=int,
               help='Dimensionality of the second layer')

    parser.add('--third_l_sz',     '-third_l',   default=384, type=int,
               help='Dimensionality of the third layer')

    parser.add('--n_prev_poses',   '-pose_numb', default=3,   type=int,
               help='Number of previous poses to consider for auto-regression')

    parser.add('--text_embedding', '-text_emb',  default="BERT", choices=["BERT", "FastText"],
               help='Which text embedding do we use (\'BERT\' or \'FastText\')')

    # Training params
    parser.add('--batch_size',    '-btch',  default=64,     type=int,   help='Batch size')
    parser.add('--learning_rate', '-lr',    default=0.0001, type=float, help='Learning rate')
    parser.add('--vel_coef',      '-vel_c', default=0.6,    type=float, help='Coefficient for the velocity loss')
    parser.add('--dropout',       '-drop',  default=0.2,    type=float, help='Dropout probability')
    parser.add('--dropout_multiplier', '-d_mult', default=4.0, type=float, help='The dropout is multiplied by this factor in the conditioning layer')
    parser.add('--n_epochs_with_no_autoregression', '-n_teacher', default=7, type=int, help='The number of epochs with full teacher forcing enabled')
    # Prediction saving parameters
    parser.add('--save_val_predictions_every_n_epoch', '-val_save_rate', default=0, type=int, 
               help='If n > 0, generate and save the predicted gestures on the first validation sequence '
                    'every n training epochs (default: 0 i.e. saving is disabled)')
    
    parser.add('--save_train_predictions_every_n_epoch', '-train_save_rate', default=0, type=int,
               help='If n > 0, generate and save the predicted gestures on the first training sequence '
                    'every n training epochs (default: 0 i.e. saving is disabled)')
    
    parser.add('--saved_prediction_duration_sec', '-gesture_len', default=9, type=int,
               help='The length of the saved gesture predictions in seconds')

    parser.add('--prediction_save_formats', '-save_formats', action='append', default=[],
               choices=["bvh_file", "raw_gesture", "video", "3d_coordinates"],
               help='The format(s) in which the predictions will be saved.'
                    'To enable multiple formats, provide the formats separately e.g. '
                    '--prediction_save_formats bvh_file --prediction_save_formats videos')
                    
    parser.add('--use_pca', '-pca', action='store_true',
               help='If set, use PCA on the gestures')

    parser.add('--use_recurrent_speech_enc', '-use_rnn', action='store_true',
               help='If set, use only the rnn for encoding speech frames')
     
    parser.add('--no_overwrite_warning', '-no_warn', action='store_true',
               help='If this flag is set, and the given <run_name> directory already exists, '
                    'it will be cleared without displaying any warnings')


    return parser
