# -*- coding: utf-8 -*-
"""
Calculating average point error

@authors: kaneko.naoshi, kucherenko.taras
"""

import argparse
import glob
import os

import numpy as np
from sklearn.metrics import mean_absolute_error


def read_joint_names(filename):
    """Read motion capture's body joint names from file

      Args:
          filename:     file name to read

      Returns:
          joint_names:  list of joint names
    """

    with open(filename, 'r') as f:
        org = f.read()
        joint_names = org.split(',')

    return joint_names


def remove_velocity(data, dim=3):
    """Remove velocity values from raw prediction data

      Args:
          data:         array containing both position and velocity values
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   array containing only position values
    """

    starts = np.arange(0, data.shape[1], dim * 2)
    stops = np.arange(dim, data.shape[1], dim * 2)
    return np.hstack([data[:, i:j] for i, j in zip(starts, stops)])


def MAE(original, predicted, dim=3):
    """Compute Mean Absolute Error (MAE)

      Args:
          original:     array containing joint positions of original gesture
          predicted:    array containing joint positions of predicted gesture
          dim:          gesture dimensionality

      Returns:
          mae:          MAE between original and predicted for each joint
    """

    num_frames = predicted.shape[0]

    diffs = mean_absolute_error(original[:num_frames], predicted,
                                multioutput='raw_values')

    num_joints = predicted.shape[1] // dim
    mae = np.empty(num_joints)

    for i in range(num_joints):
        x1 = i * dim + 0
        x2 = i * dim + dim
        mae[i] = np.mean(diffs[x1:x2])

    return mae


def APE(original, predicted, dim=3):
    """Compute Average Position Error (APE)

      Args:
          original:     array containing joint positions of original gesture
          predicted:    array containing joint positions of predicted gesture
          dim:          gesture dimensionality

      Returns:
          np.ndarray:   APE between original and predicted for each joint
    """

    num_frames = predicted.shape[0]
    num_joints = predicted.shape[1] // dim

    diffs = np.zeros((num_frames, num_joints))

    for i in range(num_frames):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            diffs[i, j] = np.linalg.norm(
                original[i, x1:x2] - predicted[i, x1:x2])

    return np.mean(diffs, axis=0)


def main():
    metrics = {
        'mae': MAE,
        'ape': APE,
    }

    parser = argparse.ArgumentParser(
        description='Calculate prediction errors')
    parser.add_argument('--original', '-o', default='GT',
                        help='Original gesture directory')
    parser.add_argument('--predicted', '-p', default='NoAutoregression',
                        help='Predicted gesture directory')
    parser.add_argument('--metric', '-m', default='ape',
                        help='Error metric (ape or mae)')
    parser.add_argument('--select', '-s', nargs='+',
                        help='Joint subset to compute (if omitted, use all)')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    predicted_dir = "data/" + args.predicted + "/"  # os.path.join(args.predicted, args.gesture)
    original_dir = "data/" + args.original + "/"

    original_files = sorted(glob.glob(os.path.join(original_dir, '*.npy')))

    predicted_files = sorted(glob.glob(os.path.join(predicted_dir, '*.npy')))

    # Check number of files
    if len(original_files) != len(predicted_files):
        raise ValueError('Inconsistent number of files : {} vs {}'
                         ''.format(len(original_files), len(predicted_files)))

    # Check if error metric was correct
    if args.metric not in metrics:
        raise ValueError('Unknown metric: \'{}\'. Choose from {}'
                         ''.format(args.metric, list(metrics.keys())))

    errors = []
    for original_file, predicted_file in zip(original_files, predicted_files):
        original = np.load(original_file)
        predicted = np.load(predicted_file)

        #print("Min: ", original, "\nMax: ", max(original.all()))

        if original.shape[0] != predicted.shape[0]:
            # Cut them to the same length
            length = min(original.shape[0], predicted.shape[0])
            original = original[:length]
            predicted = predicted[:length]

        if predicted.shape[1] == 192 * 2:
            print(predicted.shape)
            print("Removing the velocity")
            # Remove the velocity
            predicted = remove_velocity(predicted)

        error = metrics[args.metric](original, predicted)
        errors.append(error)

        basename = os.path.basename(predicted_file)
        line = basename
        for e in error:
            line += ',' + str(e)
        line += '\n'


    out_dir = os.path.join(args.out, args.predicted)

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('{}: {:.2f} +- {:.2F}'.format(args.metric.upper(), np.mean(errors), np.std(errors)))


if __name__ == '__main__':
    main()
