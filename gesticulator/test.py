"""
This script generates gestures output based on the speech input.
The gestures will be written in the text file:
3d coordinates together with the velocities.
Apart from that, video of the gesture will be generated.

author: Taras Kucherenko
contact: tarask@kth.se
"""

import torch
import numpy as np

from gesticulator.model.model import GesticulatorModel
from gesticulator.data_processing.SGdataset import SpeechGestureDataset

# Params
from gesticulator.parameters import parser

torch.set_default_tensor_type('torch.FloatTensor')


def predict(model, speech_file, text_file, gesture_file):
    """ Predict human gesture based on the speech

    Args:
        mean_pose:     mean_pose in the dataset
        model:       model to evaluate
        speech_file:    address to the speech file
        text_file:      address to the text file
        gesture_file:   file to save gestures in

    Returns:
        nothing, saves the gesture into a txt file

    """

    model.eval()

    test_length = 300

    S = np.load(speech_file)[:test_length]
    S_tensor = torch.tensor(S).float()

    T = np.load(text_file)[:int(test_length/2)]
    T_tensor = torch.tensor(T).float()
    # upsample text to get the same sampling rate as the audio
    cols = np.linspace(0, int(test_length/2), endpoint=False, num=test_length, dtype=int)
    T_tensor = T_tensor[cols, :]

    # create a "batch" and then take the first element of the resulting batch
    motion_seq = model.forward(S_tensor.unsqueeze(0), T_tensor.unsqueeze(0)).squeeze()

    # detach
    gestures_norm = np.array(motion_seq.detach().numpy())

    # denormalize
    gestures = gestures_norm * model.max_val + model.mean_pose[np.newaxis]
    print(gestures.shape)

    np.save(gesture_file, gestures)


if __name__ == "__main__":

    args = parser.parse_args()

    mean_pose = np.array([0 for i in range(45)])
    max_val = np.array([0 for i in range(45)])

    the_model = GesticulatorModel(args, mean_pose, max_val)
    the_model.load_state_dict(torch.load(args.model_file))

    train_dataset = SpeechGestureDataset(args.data_dir, train=True, apply_PCA=args.pca)

    # Produce gestures
    print("Generation gestures ...")

    gesture_file = "temp_ges.npy"
    predict(the_model, args.test_audio, args.test_text, gesture_file)


    """print("Making a video ... ")
    epoch = args.curr_epoch

    # define files
    resulting_bvh_file = "../log/gestures/result_ep" + str(epoch) + '.bvh'
    resulting_npy_file = "../log/gestures/result_ep" + str(epoch) + '.npy'
    resulting_video_file = "../log/gestures/result_ep" + str(epoch) + '.mp4'

    gestures = np.load(gesture_file)

    visualize(gestures, resulting_bvh_file, resulting_npy_file, resulting_video_file, start_t=0, end_t=20)"""

