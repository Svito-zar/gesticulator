"""
This file contains our model, defined using Pytorch Lightning.
By default, it is an autoregressive neural network with only fully connected layers (no convolutions, no recurrency).

author: Taras Kucherenko
contact: tarask@kth.se
"""

import os
from os import path
from collections import OrderedDict
import math
from argparse import ArgumentParser
import warnings

from shutil import rmtree
from joblib import load

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from gesticulator.model.prediction_saving import PredictionSavingMixin
from gesticulator.data_processing.SGdataset import SpeechGestureDataset, ValidationDataset

warnings.filterwarnings("ignore")
torch.set_default_tensor_type('torch.FloatTensor')
torch.autograd.set_detect_anomaly(True)

def weights_init_he(m):
    """Initialize the given linear layer using He initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        n = m.in_features * m.out_features
        # m.weight.data shoud be taken from a normal distribution 
        m.weight.data.normal_(0.0, np.sqrt(2 / n))
        # m.bias.data should be 0
        m.bias.data.fill_(0)

def weights_init_zeros(m):
    """Initialize the given linear layer with zeroes."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.zeros_(m.bias.data)
        nn.init.zeros_(m.weight.data)

class GesticulatorModel(pl.LightningModule, PredictionSavingMixin):
    """
    Our autoregressive model definition.

    For details regarding the code structure, please visit the documentation for Pytorch-Lightning: 
        https://pytorch-lightning.readthedocs.io/en/stable/new-project.html
    """
    data_fps = 20
  
    # ----------- Initialization -----------

    def __init__(self, args, inference_mode=False):
        """ Constructor.
        Args:
            args:            command-line arguments, see add_model_specific_args() for details
            inference_mode:  if True, then construct the model without loading the datasets into memory (used for loading saved models)
        """
        super().__init__()
        self.save_hyperparameters(args)

        if not inference_mode:
            self.create_result_folder()
            # The datasets are loaded in this constructor because they contain 
            # necessary information for building the layers (namely the audio dimensionality)
            self.load_datasets()
            self.hparams.audio_dim = self.train_dataset.audio_dim
            self.calculate_mean_pose()
        
        self.construct_layers(self.hparams)
        self.init_layers()
        
        self.rnn_is_initialized = False
        self.loss = nn.MSELoss()
        self.teaching_freq = 0
    
    def setup(self, stage):
        """ 
        Called at the beginning of trainer.fit() and trainer.test().
        """
        self.init_prediction_saving_params()

    def load_datasets(self):
        try:
            self.train_dataset = SpeechGestureDataset(self.hparams.data_dir, self.hparams.use_pca, train=True)
            self.val_dataset   = SpeechGestureDataset(self.hparams.data_dir, self.hparams.use_pca, train=False)
            self.val_sequence  = ValidationDataset(self.hparams.data_dir, self.hparams.past_context, self.hparams.future_context)
        except FileNotFoundError as err:
            abs_data_dir = os.path.abspath(self.hparams.data_dir)
            if not os.path.isdir(abs_data_dir):
                print(f"ERROR: The given dataset directory {abs_data_dir} does not exist!")
                print("Please, set the correct path with the --data option!")
            else:
                print("ERROR: Missing data in the dataset!")
            exit(-1)
    
    def create_result_folder(self):
        """Create the <results>/<run_name> folder."""
        run_name = self.hparams.run_name
        self.save_dir = path.join(self.hparams.result_dir, run_name)
        # Clear the save directory for this run if it exists
        if path.isdir(self.save_dir):
            if run_name == 'last_run' or self.hparams.no_overwrite_warning:
                rmtree(self.save_dir)
            else:
                print(f"WARNING: Result directory '{self.save_dir}' already exists!", end=' ')
                print("All files in this directory will be deleted!")
                print("(this warning can be disabled by setting the --no_overwrite_warning parameter True)")
                print("\nType 'ok' to clear the directory, and anything else to abort the program.")

                if input() == 'ok':
                    rmtree(self.save_dir)
                else:
                    exit(-1)

    def construct_layers(self, args):
        """Construct the layers of the model."""
        if args.activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif args.activation == "TanH":
            self.activation = nn.Tanh()
        else:
            print(f"ERROR: Unknown activation function '{self.activation}'!")
            exit(-1)

        self.n_layers = args.n_layers

        if self.n_layers == 1:
            final_hid_l_sz = args.first_l_sz
        elif self.n_layers == 2:
            final_hid_l_sz = args.second_l_sz
        else:
            final_hid_l_sz = args.third_l_sz

        if args.use_pca:
            self.output_dim = 12
        else:
            self.output_dim = 45

        if args.text_embedding == "BERT":
            self.text_dim = 773
        else:
            raise "Unknown word embedding"

        # The best model uses just one layer here, but we still define several
        self.first_layer = nn.Sequential(nn.Linear(args.full_speech_enc_dim, args.first_l_sz),
                                         self.activation, nn.Dropout(args.dropout))
        self.second_layer = nn.Sequential(nn.Linear(args.first_l_sz, args.second_l_sz),
                                         self.activation, nn.Dropout(args.dropout))
        self.third_layer = nn.Sequential(nn.Linear(args.second_l_sz, args.third_l_sz),
                                         self.activation, nn.Dropout(args.dropout))

        self.hidden_to_output = nn.Sequential(nn.Linear(final_hid_l_sz, self.output_dim),
                                              nn.Tanh(), nn.Dropout(args.dropout),
                                              nn.Linear(self.output_dim, self.output_dim))

        # Speech frame-level Encodigs
        if args.use_recurrent_speech_enc:
            self.gru_size = int(args.speech_enc_frame_dim / 2)
            self.gru_seq_l = args.past_context + args.future_context
            self.hidden = None
            self.encode_speech = nn.GRU(self.train_dataset.audio_dim + self.text_dim, self.gru_size, 2,
                                        dropout=args.dropout, bidirectional=True)
        else:
            self.encode_speech = nn.Sequential(nn.Linear(self.hparams.audio_dim + self.text_dim,
                                               args.speech_enc_frame_dim * 2), self.activation,
                                               nn.Dropout(args.dropout), nn.Linear(args.speech_enc_frame_dim*2,
                                                                                   args.speech_enc_frame_dim),
                                               self.activation, nn.Dropout(args.dropout))

        # To reduce deminsionality of the speech encoding
        self.reduce_speech_enc = nn.Sequential(nn.Linear(int(args.speech_enc_frame_dim * \
                                                        (args.past_context + args.future_context)),
                                                         args.full_speech_enc_dim),
                                               self.activation, nn.Dropout(args.dropout))

        self.conditioning_1 = nn.Sequential(nn.Linear(self.output_dim * args.n_prev_poses,
                                                args.first_l_sz * 2), self.activation,
                                            nn.Dropout(args.dropout * args.dropout_multiplier))

    def init_layers(self):
        # Use He initialization for most layers
        self.first_layer.apply(weights_init_he)
        self.second_layer.apply(weights_init_he)
        self.third_layer.apply(weights_init_he)
        self.hidden_to_output.apply(weights_init_he)
        self.reduce_speech_enc.apply(weights_init_he)

        # Initialize conditioning with zeros
        self.conditioning_1.apply(weights_init_zeros)

    def calculate_mean_pose(self):
        self.hparams.mean_pose = np.mean(self.val_dataset.gesture, axis=(0, 1))
        np.save("./utils/mean_pose.npy", self.hparams.mean_pose)

    def load_mean_pose(self):
        self.hparams.mean_pose = np.load("./utils/mean_pose.npy")

    def initialize_rnn_hid_state(self):
        """Initialize the hidden state for the RNN."""
      
        self.hidden = torch.ones([4, self.gru_seq_l, self.gru_size], dtype=torch.float32)

        self.rnn_is_initialized = True

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True)
            
        return loader
    
    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False)

        return loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # ----------- Model -----------

    def on_train_start(self):
        """
        Load the input data for generating prediction videos on the training
        and the validation data. These were not used in the paper, but they can
        be used to track the model performance better than the validation loss.
        
        NOTE: The data has to be loaded here because in the constructor,
        the model is still on the CPU, therefore the data would be always loaded
        onto the CPU as well, causing a device mismatch if the GPU is enabled.
        """
         # Load the first training sequence as input
        if "training" in self.enabled_phases:
            self.train_input = \
                self.load_train_or_val_input(self.train_dataset[0])
                
        # Load the first validation sequence as input
        if "validation" in self.enabled_phases:
            self.val_input = \
                self.load_train_or_val_input(self.val_sequence[0])
       
                
    def forward(self, audio, text, use_conditioning, motion, use_teacher_forcing=True):
        """
        Generate a sequence of gestures based on a sequence of speech features (audio and text)

        Args:
            audio [N, T, D_a]:    a batch of sequences of audio features
            text  [N, T/2, D_t]:  a batch of sequences of text BERT embedding
            use_conditioning:     a flag indicating if we are using autoregressive conditioning
            motion: [N, T, D_m]   the true motion corresponding to the input (NOTE: it can be None during testing and validation)
            use_teacher_forcing:  a flag indicating if we use teacher forcing

        Returns:
            motion [N, T, D_m]:   a batch of corresponding motion sequences
        """

        # initialize the motion sequence
        motion_seq = None

        # initialize RNN state if needed
        if self.hparams.use_recurrent_speech_enc and (not self.rnn_is_initialized or motion is None):
            self.initialize_rnn_hid_state()
        # initialize all the previous poses with the mean pose
        init_poses = np.array([self.hparams.mean_pose for it in range(audio.shape[0])])
        # we have to put these Tensors to the same device as the model because 
        # numpy arrays are always on the CPU
        # store the 3 previous poses
        prev_poses = [torch.from_numpy(init_poses).to(self.device)] * 3
        
        past_context   = self.hparams.past_context
        future_context = self.hparams.future_context
        for time_st in range(past_context, audio.shape[1] - future_context):
            # take current audio and text of the speech
            curr_audio = audio[:, time_st - past_context:time_st+future_context]
            curr_text = text[:, time_st-past_context:time_st+future_context]
            curr_speech = torch.cat((curr_audio, curr_text), 2)
            # encode speech
            if self.hparams.use_recurrent_speech_enc:
                speech_encoding_full, hh = self.encode_speech(curr_speech)
            else:
                speech_encoding_full = self.encode_speech(curr_speech)

            speech_encoding_concat = torch.flatten(speech_encoding_full, start_dim=1)
            speech_enc_reduced = self.reduce_speech_enc(speech_encoding_concat)

            if use_conditioning:
                # Take several previous poses for conditioning
                if self.hparams.n_prev_poses == 3:
                    pose_condition_info = torch.cat((prev_poses[-1], prev_poses[-2],
                                                     prev_poses[-3]), 1)
                elif self.hparams.n_prev_poses == 2:
                    pose_condition_info = torch.cat((prev_poses[-1], prev_poses[-2]), 1)
                else:
                    pose_condition_info = prev_poses[-1]

                conditioning_vector_1 = self.conditioning_1(pose_condition_info)

            else:
                conditioning_vector_1 = None


            first_h = self.first_layer(speech_enc_reduced)
            first_o = self.FiLM(conditioning_vector_1, first_h,
                                self.hparams.first_l_sz, use_conditioning)

            if self.n_layers == 1:
                final_h = first_o
            else:
                second_h = self.second_layer(first_o)

                if self.n_layers == 2:
                    final_h = second_h
                else:
                    final_h = self.third_layer(second_h)
            
            curr_pose = self.hidden_to_output(final_h)

            if motion is not None and use_teacher_forcing and time_st % self.teaching_freq < 2:
                # teacher forcing
                prev_poses[-3] = motion[:, time_st - 2, :]
                prev_poses[-2] = motion[:, time_st - 1, :]
                prev_poses[-1] = motion[:, time_st, :]
            else:
                # no teacher
                prev_poses[-3] = prev_poses[-2]
                prev_poses[-2] = prev_poses[-1]
                prev_poses[-1] = curr_pose

            # add current frame to the total motion sequence
            if motion_seq is None:
                motion_seq = curr_pose.unsqueeze(1)
            else:
                motion_seq = torch.cat((motion_seq, curr_pose.unsqueeze(1)), 1)               

        # Sanity check
        if motion_seq is None:
            print("ERROR: GesticulatorModel.forward() returned None!")
            print("Possible causes: corrupt dataset or a problem with the environment.")
            exit(-1)

        return motion_seq

    def FiLM(self, conditioning, nn_layer, hidden_size, use_conditioning):
        """
        Execute FiLM conditioning of the model
        (see https://distill.pub/2018/feature-wise-transformations/)
        Args:
            conditioning:     a vector of conditioning information
            nn_layer:         input to the FiLM layer
            hidden_size:      size of the hidden layer to which coniditioning is applied
            use_conditioning: a flag if we are going to condition or not

        Returns:
            output:           the conditioned output
        """

        # no conditioning initially
        if not use_conditioning:
            output = nn_layer
        else:
            alpha, beta = torch.split(conditioning, (hidden_size, hidden_size), dim=1)
            output = nn_layer * (alpha+1) + beta

        return output

    def tr_loss(self, y_hat, y):
        """
        Training loss
        Args:
            y_hat:  prediction
            y:      ground truth

        Returns:
            The total loss: a sum of MSE error and velocity penalty
        """

        # calculate corresponding speed
        pred_speed = y_hat[:,1:] - y_hat[:,:-1]
        actual_speed = y[:,1:] - y[:,:-1]
        vel_loss = self.loss(pred_speed, actual_speed)

        return [self.loss(y_hat, y), vel_loss * self.hparams.vel_coef]

    def val_loss(self, y_hat, y):
        # calculate corresponding speed
        pred_speed = y_hat[:,1:] - y_hat[:,:-1]
        actual_speed = y[:,1:] - y[:,:-1]

        return self.loss(pred_speed, actual_speed)

    def training_step(self, batch, batch_nb):
        """
        Training step, used by Pytorch Lightning to train the model
        Args:
            batch:      mini-batch data
            batch_nb:   mini-batch index/number

        Returns:
            logs
        """

        audio = batch["audio"]
        text = batch["text"]
        true_gesture = batch["output"]

        # first decide if we are going to condition
        if self.current_epoch < self.hparams.n_epochs_with_no_autoregression:
           use_conditioning = False
        else:
           use_conditioning = True

        # scheduled sampling for teacher forcing
        predicted_gesture = self.forward(audio, text, use_conditioning, true_gesture)

        # remove last frames which had no future info and hence were not predicted
        true_gesture = true_gesture[:,  
                       self.hparams.past_context:-self.hparams.future_context]
        
        # Get training loss
        mse_loss, vel_loss = self.tr_loss(predicted_gesture, true_gesture)
        loss = mse_loss + vel_loss

        loss_val = loss.unsqueeze(0)
        mse_loss_val = mse_loss.unsqueeze(0)
        vel_loss_val = vel_loss.unsqueeze(0)

        tqdm_dict = {"train/full_loss": loss_val,
                      "train/mse_loss": mse_loss_val,
                      "train/cont_loss": vel_loss_val}

        output = OrderedDict({
            'loss': loss,
            'log': tqdm_dict})

        return output

    def training_epoch_end(self, outputs):
        elapsed_epochs = self.current_epoch - self.last_saved_train_prediction_epoch 
        
        if self.save_train_predictions and elapsed_epochs >= self.hparams.save_train_predictions_every_n_epoch:
            self.last_saved_train_prediction_epoch = self.current_epoch
            self.generate_training_predictions()

        return {} # The trainer expects a dictionary
        
    def validation_step(self, batch, batch_nb):
        speech = batch["audio"]
        text = batch["text"]
        true_gesture = batch["output"]

        # Text on validation sequences without teacher forcing
        predicted_gesture = self.forward(speech, text, use_conditioning=True, motion = None, use_teacher_forcing=False)

        # remove last frame which had no future info
        true_gesture = true_gesture[:,
                       self.hparams.past_context:-self.hparams.future_context]

        val_loss = self.val_loss(predicted_gesture, true_gesture)

        logger_logs = {'validation_loss': val_loss}

        return {'val_loss': val_loss, 'val_example':predicted_gesture, 'log': logger_logs}
 
    def validation_epoch_end(self, outputs):
        """
        This will be called at the end of the validation loop

        Args:
            outputs: whatever "validation_step" has returned
        """
        elapsed_epochs = self.current_epoch - self.last_saved_val_prediction_epoch 
        
        if self.save_val_predictions and elapsed_epochs >= self.hparams.save_val_predictions_every_n_epoch:
            self.last_saved_val_prediction_epoch = self.current_epoch
            self.generate_validation_predictions()


        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tqdm_dict = {'avg_val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, "log": tqdm_dict}

    def generate_evaluation_videos(self, semantic, random):
        """
        Load the selected semantic and/or random segments 
        for generating the evaluation videos.
        
        Args:
            semantic (bool):  if True, evaluate the model on the semantic input segments
            random (bool):  if True, evaluate the model on the random input segments

        NOTE: The data has to be loaded here because in the constructor,
        the model is still on the CPU, therefore the data would be always loaded
        onto the CPU as well, causing a device mismatch if the GPU is enabled.
        """
        self.test_prediction_inputs = {
            '04': self.load_test_prediction_input('04'),
            '05': self.load_test_prediction_input('05')}
        
        if semantic:
            self.generate_test_predictions(mode='seman')
    
        if random:
            self.generate_test_predictions(mode='rand')

        return
  
    
    def on_epoch_start(self):
        # Anneal teacher forcing schedule
        if self.current_epoch < self.hparams.n_epochs_with_no_autoregression:
            self.teaching_freq = 16 # full teacher forcing
        else:
            self.teaching_freq = max(int(self.teaching_freq/2), 2)
            print("Current teacher forcing frequency is: ", self.teaching_freq)
