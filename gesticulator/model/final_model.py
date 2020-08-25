"""
This file contains our final model

author: Taras Kucherenko
contact: tarask@kth.se
"""

import os
import torch.nn as nn
import torch
import numpy as np
import math

from joblib import load

import pytorch_lightning as pl

from collections import OrderedDict

from torchvision import transforms

torch.set_default_tensor_type('torch.FloatTensor')

# Dataset
from gesticulator.data_processing.SGdataset import SpeechGestureDataset, ValidationDataset

# Params
from argparse import ArgumentParser

from ray import tune

def weights_init_he(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        n = m.in_features * m.out_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, np.sqrt( 2 / n))
        # m.bias.data should be 0
        m.bias.data.fill_(0)

def weights_init_zeros(m):
    '''Takes in a module and initializes all linear layers with
        zeros.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        nn.init.zeros_(m.bias.data)
        nn.init.zeros_(m.weight.data)

class My_Model(pl.LightningModule):
    """
    Our model which contains GRU and MDN
    """

    def __init__(self, args):

        super(My_Model, self).__init__()

        self.hyper_params = args

        if args.activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif args.activation == "TanH":
            self.activation = nn.Tanh()
        else:
            raise "Unknown actication!"

        self.numb_layers = args.n_layers

        if args.use_pca:
            self.output_dim = 12
        else:
            self.output_dim =45

        if args.text_embedding == "BERT":
            self.text_dim =   773
        else:
            raise "Unknown word embedding"

        self.gru_size = int (args.speech_enc_frame_dim / 2)
        self.gru_seq_l = args.past_context + args.future_context
        self.dropout = args.dropout

        self.batch_size = args.batch_size

        self.first_later = nn.Sequential(nn.Linear(args.full_speech_enc_dim , args.first_l_sz), self.activation, nn.Dropout(self.dropout))
        self.second_layer = nn.Sequential(nn.Linear(args.first_l_sz,  args.second_l_sz),  self.activation, nn.Dropout(self.dropout))
        self.third_layer = nn.Sequential(nn.Linear(args.second_l_sz, args.third_l_sz), self.activation, nn.Dropout(self.dropout))
        if self.numb_layers == 1:
            final_hid_l_sz = args.first_l_sz
        elif self.numb_layers == 2:
            final_hid_l_sz = args.second_l_sz
        else:
            final_hid_l_sz = args.third_l_sz

        self.hidden_to_output = nn.Sequential(nn.Linear(final_hid_l_sz, self.output_dim), self.activation, nn.Dropout(self.dropout), nn.Linear(self.output_dim, self.output_dim))

        # Speech frame-level Encodigs
        if args.recurrent_speech_enc:
            self.encode_speech_rnn = nn.GRU(args.audio_dim + self.text_dim, self.gru_size, 2, dropout=self.dropout, bidirectional=True)
        else:
            self.encode_speech_rnn = nn.Sequential(nn.Linear(args.audio_dim + self.text_dim, args.speech_enc_frame_dim*2),
                                                   self.activation, nn.Dropout(self.dropout),nn.Linear(args.speech_enc_frame_dim*2, args.speech_enc_frame_dim),
                                                   self.activation, nn.Dropout(self.dropout))

        self.reduce_speech_enc = nn.Sequential(nn.Linear(int(args.speech_enc_frame_dim * (args.past_context + args.future_context)), args.full_speech_enc_dim),
                                               self.activation, nn.Dropout(self.dropout))
        
        if args.only_diff:
            self.conditioning_1 = nn.Sequential(nn.Linear(self.output_dim * 2, args.poses_enc_eff_dim),
                                                self.activation, nn.Dropout(self.dropout * 4), nn.Linear(args.poses_enc_eff_dim, args.first_l_sz * 2))
            
        elif args.hour_glass:
            self.conditioning_1 = nn.Sequential(nn.Linear(self.output_dim * args.numb_prev_poses, args.poses_enc_eff_dim),
                                                self.activation, nn.Dropout(self.dropout * 4), nn.Linear(args.poses_enc_eff_dim, args.first_l_sz * 2))
        else:
            self.conditioning_1 = nn.Sequential(nn.Linear(self.output_dim * args.numb_prev_poses, args.first_l_sz * 2),
                                                self.activation, nn.Dropout(self.dropout * 4))

        self.conditioning_2 = nn.Sequential(nn.Linear(args.full_speech_enc_dim, args.second_l_sz * 2),
                                            self.activation, nn.Dropout(self.dropout))

        # Use He initialization
        self.first_later.apply(weights_init_he)
        self.second_layer.apply(weights_init_he)
        self.third_layer.apply(weights_init_he)
        self.hidden_to_output.apply(weights_init_he) # ToDo - initialize with zeros
        self.reduce_speech_enc.apply(weights_init_he)

        # Initialize conditioning with zeros
        self.conditioning_1.apply(weights_init_zeros)
        self.conditioning_2.apply(weights_init_zeros)


        #self.load_mean_pose()
        self.calculate_mean_pose()

        # TodO: check normalization
        #self.mean_pose = mean_pose
        #self.max_val = max_val

        #self.silence = np.array([silence_features] * context_length)
        self.past_context = args.past_context
        self.future_context = args.future_context

        self.initialized = False
        self.hidden = None

        self.loss = nn.MSELoss()

        self.teaching_freq = 0

        print("PCA IS :", args.use_pca)
        print("RNN IS :", args.recurrent_speech_enc)


    def calculate_mean_pose(self):
        train_dataset = SpeechGestureDataset(self.hyper_params.data_dir, self.hyper_params.use_pca, train=False)

        self.mean_pose = np.mean(train_dataset.gesture, axis=(0, 1))

        #np.save("mean_pose.npy", self.mean_pose)

    def load_mean_pose(self):
        self.mean_pose = np.load("/home/tarask/Documents/Code/CVPR2020/gesticulator/utils/mean_pose.npy")

    def initialize_state(self):
        if torch.cuda.is_available():
            self.hidden = torch.ones([4, self.gru_seq_l, self.gru_size], dtype=torch.float32, device=torch.cuda.current_device())
        else:
            self.hidden = torch.ones([4, self.gru_seq_l, self.gru_size], dtype=torch.float32 )

        self.initialized = True


    def FiLM(self, conditioning, nn_layer, hidden_size, cond):
        """
        Execute film conditioning of the model
        Args:
            conditioning:  a vector of conditioning information
            nn_layer:      input to the FiLM layer

        Returns:
            output:        input already conditioned

        """

        # no conditioning initially
        if not cond:
            output = nn_layer
        else:
            alpha, beta = torch.split(conditioning, (hidden_size, hidden_size), dim=1)
            output = nn_layer * (alpha+1) + beta

            #alpha_values = alpha.cpu().detach().numpy()
            #print("Alpha range is [", alpha_values.min(), ", ", alpha_values.max(), "]")

        return output

    def forward(self, speech, text, condition, motion=None, teacher=True):
        """
        Generate a sequence of gestures based on a sequence of motion

        Args:
            speech [N, T, D]:      a batch of sequences of speech features
            speech [N, T/2, D_t]:  a batch of sequences of text BERT embedding

        Returns:
            motion [N, T, D2]:  a batch of corresponding motion sequences
        """

        motion_seq = None

        if self.initialized == False or motion is None:
            self.initialize_state()

        if False:  # motion is not None and teacher:
            pose_prev = motion[:, self.past_context - 1, :]  # teacher forcing
            pose_prev_prev = motion[:, self.past_context - 2, :]
            pose_prev_prev_prev = motion[:, self.past_context - 3, :]
        else:
            pose_prev = torch.tensor([self.mean_pose for it in range(len(speech))]).to(speech.device)
            pose_prev_prev = torch.tensor([self.mean_pose for it in range(len(speech))]).to(speech.device)
            pose_prev_prev_prev = torch.tensor([self.mean_pose for it in range(len(speech))]).to(speech.device)
        # ToDo : calculate mean couple


        for time_st in range(self.past_context, len(speech[0]) - self.future_context):

            # text current audio and text of the speech
            curr_audio = torch.tensor(speech[:, time_st - self.past_context:time_st + self.future_context, :]).to(
                speech.device)
            curr_text  = torch.tensor(text[:,time_st-self.past_context:time_st+self.future_context,:]).to(speech.device)
            curr_speech = torch.cat((curr_audio,curr_text),2)

            # encode speech

            if self.hyper_params.recurrent_speech_enc:
                speech_encoding_full, hh = self.encode_speech_rnn(curr_speech)
            else:
                speech_encoding_full = self.encode_speech_rnn(curr_speech)


            speech_encoding_concat = torch.flatten(speech_encoding_full, start_dim=1)
            speech_enc_reduced = self.reduce_speech_enc(speech_encoding_concat)

            if self.hyper_params.only_diff:
                pose_condition_info = torch.cat((pose_prev- pose_prev_prev, pose_prev_prev - pose_prev_prev_prev), 1)
            elif self.hyper_params.numb_prev_poses == 3:
                pose_condition_info = torch.cat((pose_prev, pose_prev_prev, pose_prev_prev_prev), 1)
            elif self.hyper_params.numb_prev_poses == 2:
                pose_condition_info = torch.cat((pose_prev, pose_prev_prev), 1)
            else:
                pose_condition_info = pose_prev

            curr_input = speech_enc_reduced

            conditioning_vector_1 = self.conditioning_1(pose_condition_info)
            first_h = self.first_later(curr_input)
            first_o = self.FiLM(conditioning_vector_1, first_h,
                                self.hyper_params.first_l_sz, condition)  # torch.cat((first_h, speech_enc_reduced), 1) #self.FiLM(conditioning_vector_1, first_h, self.hidden_size)

            if self.numb_layers == 1:
                final_h = first_o
            else:
                conditioning_vector_2 = self.conditioning_2(speech_enc_reduced)
                second_h = self.second_layer(first_o)
                second_o = self.FiLM(conditioning_vector_2, second_h,
                                     self.hyper_params.second_l_sz, condition)  # torch.cat((second_h, speech_enc_reduced), 1) # self.FiLM(conditioning_vector_2, second_h, self.hidden_size)

                if self.numb_layers == 2:
                    final_h = second_o
                else:
                    # conditioning_vector_3 = self.conditioning_3(speech_enc_reduced)
                    final_h = self.third_layer(second_o)

            curr_pose = self.hidden_to_output(final_h) * math.pi # because it is from -pi to pi

            if motion is not None and teacher:
                if time_st % self.teaching_freq < 2:
                    pose_prev_prev_prev = motion[:, time_st - 2, :]
                    pose_prev_prev = motion[:, time_st - 1, :]
                    pose_prev = motion[:, time_st, :]  # teacher forcing
                else:
                    # own prediction

                    pose_prev_prev_prev = pose_prev_prev
                    pose_prev_prev = pose_prev
                    pose_prev = curr_pose
            else:
                # no teacher

                pose_prev_prev_prev = pose_prev_prev
                pose_prev_prev = pose_prev
                pose_prev = curr_pose

            if motion_seq is not None:
                motion_seq = torch.cat((motion_seq, curr_pose.unsqueeze(1)), 1)
            else:
                motion_seq = curr_pose.unsqueeze(1)

        return motion_seq

    def tr_loss(self, y_hat, y):

        n_element = y_hat.numel()

        # calculate corresponding speed
        pred_speed = y_hat[1:] - y_hat[:-1]
        actual_speed = y[1:] - y[:-1]

        # and calculate variance as well
        norm = torch.norm(y_hat, 2, 1) 
        var_loss = -torch.sum(norm) / n_element
        return [self.loss(y_hat, y), self.loss(pred_speed, actual_speed)* self.hyper_params.vel_coef,
                var_loss * self.hyper_params.var_coef]

    def generate_video(self):
        # test generating script

        seq_len = 300
        text_len = int(seq_len/2)

        # read data
        dev_dir = "/home/tarask/Documents/storage/SpeechToMotion/Irish/WithTextV6/dev_inputs"
        speech_data = np.load(dev_dir + "/X_test_NaturalTalking_01.npy")[:seq_len]
        text =  np.load(dev_dir + "/T_test_NaturalTalking_01.npy")[:text_len]

        # upsample text to get the same sampling rate as the audio
        cols = np.linspace(0, text.shape[0], endpoint=False, num=text.shape[0]*2, dtype=int)
        text_data = text[cols,:]

        # Convert to float tensors and put on GPU
        speech = torch.tensor([speech_data]).float().cuda()
        text =  torch.tensor([text_data]).float().cuda()

        # Text on validation sequences without teacher forcing
        predicted_gesture = self.forward(speech, text, condition=True, motion = None, teacher=False)

        gen_dir = "/home/tarask/Documents/Code/CVPR2020/gesticulator/log/gestures/"

        ges_file = gen_dir+ "test"+str(self.current_epoch)+".npy"
        np.save(ges_file, predicted_gesture.detach().cpu().numpy())


    def val_loss(self, y_hat, y):

        # calculate corresponding speed
        pred_speed = y_hat[1:] - y_hat[:-1]
        actual_speed = y[1:] - y[:-1]

        return self.loss(pred_speed, actual_speed)
    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        speech = batch["input"]
        text = batch["text"]
        true_gesture = batch["output"]

        # first deside if you are going to condition
        if self.current_epoch<7:
            condition = False
        else:
            condition = True

        # scheduled sampling for teacher forcing
        predicted_gesture = self.forward(speech, text, condition, true_gesture)

        # remove last frame which had no future info
        true_gesture = true_gesture[:, self.hyper_params.past_context:-self.hyper_params.future_context]
        
        # Try to get the same speed as well
        mse_loss, vel_loss, var_loss = self.tr_loss(predicted_gesture, true_gesture)
        loss = mse_loss + vel_loss + var_loss

        loss_val = loss.unsqueeze(0)
        mse_loss_val = mse_loss.unsqueeze(0)
        vel_loss_val = vel_loss.unsqueeze(0)
        var_loss_val = var_loss.unsqueeze(0)

        tqdm_dict = { "train_loss": loss_val,
                      "mse_loss_val": mse_loss_val,
                      "cont_loss_val": vel_loss_val,
                      "var_loss_val": var_loss_val}

        output = OrderedDict({
            'loss': loss,  # required
            'log': tqdm_dict
        })


        return output

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        speech = batch["input"]
        text = batch["text"]
        true_gesture = batch["output"]

        # Text on validation sequences without teacher forcing
        predicted_gesture = self.forward(speech, text, condition=True, motion = None, teacher=False)

        # remove last frame which had no future info
        true_gesture = true_gesture[:, self.hyper_params.past_context:-self.hyper_params.future_context]
        
        val_loss = self.val_loss(predicted_gesture, true_gesture)
        
        # if using TestTubeLogger or TensorboardLogger you can nest scalars:
        logger_logs = {'validation_loss': val_loss}

        return {'val_loss': val_loss, 'val_example':predicted_gesture, 'log': logger_logs}

    # TODO we provide motion during the validation?

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        #self.generate_video()

        # Save resulting gestures without teacher forcing
        sample_prediction = outputs[0]['val_example'][:3].cpu().detach().numpy()

        if self.hyper_params.use_pca:
            # apply PCA
            pca = load('/home/tarask/Documents/Code/CVPR2020/gesticulator/utils/pca_model_12.joblib')
            sample_gesture = pca.inverse_transform(sample_prediction)
        else:
            sample_gesture = sample_prediction

        resulting_ges_file = "/home/tarask/Documents/Code/CVPR2020/gesticulator/log/gestures/val_result_ep" + str(self.current_epoch + 1) + '_raw.npy'
        #resulting_ges_file = "/home/taras/Desktop/Work/Code/Git/CVPR2020/gesticulator/log/gestures/val_result_ep" + str( self.current_epoch + 1) + '_raw.npy'

        np.save(resulting_ges_file, sample_gesture)

        #self.valdiation_callback(avg_loss.detach().cpu().item(), self.current_epoch)

        #tune.track.log(mean_loss=avg_loss.cpu().item())
        tqdm_dict = {'avg_val_loss': avg_loss}

        return {'avg_val_loss': avg_loss, "log": tqdm_dict}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        speech = batch["input"]
        text = batch["text"]

        predicted_gesture = self.forward(speech, text, condition=True)

        return {'test_example': predicted_gesture}

    def test_end(self, outputs):

        print("Writing Test Examples!")
        
        for id in range(8):
            sample_gesture = outputs[id]['test_example'][0].cpu()
            #resulting_ges_file = "/home/taras/Desktop/Work/Code/Git/CVPR2020/gesticulator/log/gestures/test_result_ep" + str(self.current_epoch + 1) + "_" + str(id) + '_raw.npy'
            resulting_ges_file = "/home/tarask/Documents/Code/CVPR2020/gesticulator/log/gestures/test_result_ep" + str(self.current_epoch + 1) +  "_" + str(id) +'_raw.npy'

            np.save(resulting_ges_file, sample_gesture)
        return {'test_example': sample_gesture[0][0]}

    def configure_optimizers(self):
        # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=self.hyper_params.learning_rate)

    def on_epoch_start(self):
        # Anneal teacher forcing schedule
        if self.current_epoch < 7:
            self.teaching_freq = 16 # full teacher forcing
        else:
            self.teaching_freq = max(int(self.teaching_freq/2), 2)
        print("Current frequency is: ", self.teaching_freq)

    @pl.data_loader
    def train_dataloader(self):
        dataset =SpeechGestureDataset(self.hyper_params.data_dir, self.hyper_params.use_pca, train=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return loader

    @pl.data_loader
    def val_dataloader(self):
        dataset = SpeechGestureDataset(self.hyper_params.data_dir, self.hyper_params.use_pca, train=False)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return loader

    @pl.data_loader
    def test_dataloader(self):
        dataset = ValidationDataset(self.hyper_params.data_dir)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False
        )
        return loader
        

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--sequence_length', '-seq_l', default=40, type=int,
                            help='Length of each training sequence')
        parser.add_argument('--past_context', '-p_cont', default=10, type=int,
                            help='Length of a past context for speech to be used for gestures')
        parser.add_argument('--future_context', '-f_cont', default=20, type=int,
                            help='Length of a future context for speech to be used for gestures')
        parser.add_argument('--audio_dim', '-audio_d', default=64, type=int,
                            help='Number of dimensions in the audio features')
        parser.add_argument('--text_context', '-txt_l', default=10, type=int,
                            help='How many future text encoding should we take into account')
        
        parser.add_argument('--use_pca', '-pca', default=True, type=bool,
                            help='If we should use PCA on gestures')
        parser.add_argument('--n_layers', '-lay', default=1, type=int,
                            help='Number of hidden layer (excluding RNN)')
        parser.add_argument('--speech_enc_frame_dim', '-speech_t_e', default=124, type=int,
                            help='Dimensionality of the speech frame encoding')
        parser.add_argument('--full_speech_enc_dim', '-speech_f_e', default=612, type=int,
                            help='Dimensionality of the full speech encoding')

        # network architecture
        parser.add_argument('--activation', '-act',  default="TanH", #default="LeakyReLU",
                            help='which activation function to use')
        parser.add_argument('--first_l_sz', '-first_l', default=256, type=int,
                            help='Dimensionality of the first layer')
        parser.add_argument('--second_l_sz', '-second_l', default=512, type=int,
                            help='Dimensionality of the second layer')
        parser.add_argument('--third_l_sz', '-third_l', default=384, type=int,
                            help='Dimensionality of the third layer')

        parser.add_argument('--poses_enc_eff_dim', '-pose_enc_dim', default=32, type=int,
                            help='Dimensionality of the pose encoding')
        parser.add_argument('--numb_prev_poses', '-pose_numb', default=3, type=int,
                            help='Number of previous poses to consider for auto-regression')
        parser.add_argument('--hour_glass', '-h_glass', default=False, type=bool,
                            help='Weather we want to have a bottle_neck for the motion')
        parser.add_argument('--only_diff', '-diff', default=False, type=bool,
                            help='Weather we want to use only the speed of the previous frames')
        parser.add_argument('--recurrent_speech_enc', '-use_rnn', default=False, type=bool,
                            help='Weather we want to use only rnn for encoding speech frames')


        # Training params
        parser.add_argument('--batch_size', '-btch', default=64, type=int,
                            help='Batch size')
        parser.add_argument('--learning_rate', '-rt', default=0.0001, type=float,
                            help='Learning rate')
        parser.add_argument('--dropout', '-drop', default=0.2, type=float,
                            help='Dropout probability')
        parser.add_argument('--vel_coef', '-vel_c', default=0.6, type=float, #0.3
                            help='Coefficient for the velocity loss')
        parser.add_argument('--var_coef', '-var_c', default=0.00, type=float,
                            help='Coefficient for the variance loss')

        # Folders params
        parser.add_argument('--text_embedding', '-text_emb', default="BERT",
                            help='Which text embedding do we use')
        parser.add_argument('--data_dir', '-data',
                            default="/home/tarask/Documents/storage/SpeechToMotion/Irish/WithTextV5",
                            #default = "/home/taras/Documents/Datasets/SpeechToMotion/Irish/processed/Play",
                            help='Address to a folder with the dataset')
        parser.add_argument('--logdir', '-log', default="./log",
                            help='Address to save logs')
        parser.add_argument('--model_path', '-model_d', default="./saved_models/curr_model",
                            help='Address to save the trained model')
        return parser





