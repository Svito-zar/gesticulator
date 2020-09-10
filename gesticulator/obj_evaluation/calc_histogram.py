# -*- coding: utf-8 -*-
"""
Calculating statistics over the produced and ground truth gestures

@author: kaneko.naoshi
"""

import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np



def compute_velocity(data, dim=3):
    """Compute velocity between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          vel_norms:    velocities of each joint between each adjacent frame
    """

    # Flatten the array of 3d coords
    coords = data.reshape(data.shape[0], -1)

    # First derivative of position is velocity
    vels = np.diff(coords, n=1, axis=0)

    num_vels = vels.shape[0]
    num_joints = vels.shape[1] // dim

    vel_norms = np.zeros((num_vels, num_joints))

    for i in range(num_vels):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            vel_norms[i, j] = np.linalg.norm(vels[i, x1:x2])

    return vel_norms * 20


def compute_acceleration(data, dim=3):
    """Compute acceleration between adjacent frames

      Args:
          data:         array containing joint positions of gesture
          dim:          gesture dimensionality

      Returns:
          acc_norms:    accelerations of each joint between each adjacent frame
    """

    # Second derivative of position is acceleration
    accs = np.diff(data, n=2, axis=0)

    num_accs = accs.shape[0]
    num_joints = accs.shape[1] // dim

    acc_norms = np.zeros((num_accs, num_joints))

    for i in range(num_accs):
        for j in range(num_joints):
            x1 = j * dim + 0
            x2 = j * dim + dim
            acc_norms[i, j] = np.linalg.norm(accs[i, x1:x2])

    return acc_norms * 20 * 20


def save_result(lines, out_dir, width, measure):
    """Write computed histogram to CSV

      Args:
          lines:        list of strings to be written
          out_dir:      output directory
          width:        bin width of the histogram
          measure:      used measure for histogram calculation
    """

    # Make output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    hist_type = measure[:3]  # 'vel' or 'acc'
    filename = 'hmd_{}_{}.csv'.format(hist_type, width)
    outname = os.path.join(out_dir, filename)

    with open(outname, 'w') as out_file:
        out_file.writelines(lines)

    print('More detailed result was writen to the file: ' + outname)
    print('')


def main():
    measures = {
        'velocity': compute_velocity,
        'acceleration': compute_acceleration,
    }

    parser = argparse.ArgumentParser(
        description='Calculate histograms of moving distances')
    parser.add_argument('--original', '-o', default='original',
                        help='Original gesture directory')
    parser.add_argument('--predicted', '-p', default='predicted',
                        help='Predicted gesture directory')
    parser.add_argument('--gesture', '-g', #required=True,
                        help='Directory storing predicted txt files')
    parser.add_argument('--width', '-w', type=float, default=1,
                        help='Bin width of the histogram')
    parser.add_argument('--measure', '-m', default='velocity',
                        help='Measure to calculate (velocity or acceleration)')
    parser.add_argument('--select', '-s', nargs='+',
                        help='Joint subset to compute (if omitted, use all)')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Visualize histograms')
    parser.add_argument('--out', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    
    predicted_dir = os.path.join("data/", args.predicted)
    original_dir = os.path.join("data/", args.original)

    original_files = sorted(glob.glob(os.path.join(original_dir, '*.npy')))

    predicted_files = sorted(glob.glob(os.path.join(predicted_dir, '*.npy')))

    # Check number of files
    if len(original_files) != len(predicted_files):
        warnings.warn('Inconsistent number of files : {} vs {}'
                      ''.format(len(original_files), len(predicted_files)),
                      RuntimeWarning)

    # Check if error measure was correct
    if args.measure not in measures:
        raise ValueError('Unknown measure: \'{}\'. Choose from {}'
                         ''.format(args.measure, list(measures.keys())))

    original_out_lines = [','.join([''] + ['Total']) + '\n']
    predicted_out_lines = [','.join([''] + ['Total']) + '\n']

    original_distances = []
    predicted_distances = []
    for original_file, predicted_file in zip(original_files, predicted_files):
        original = np.load(original_file)
        predicted = np.load(predicted_file)

        if original.shape[0] != predicted.shape[0]:
            # Cut them to the same length
            length = min(original.shape[0], predicted.shape[0])
            original = original[:length]
            predicted = predicted[:length]

        original_distance = measures[args.measure](
            original)#[:, selected_joints]
        predicted_distance = measures[args.measure](
            predicted)#[:, selected_joints]

        original_distances.append(original_distance)
        predicted_distances.append(predicted_distance)

    original_distances = np.concatenate(original_distances)
    predicted_distances = np.concatenate(predicted_distances)

    # Compute histogram for each joint
    bins = np.arange(0, 19+args.width, args.width)
    num_joints = original_distances.shape[1]
    original_hists = []
    predicted_hists = []
    for i in range(num_joints):
        original_hist, _ = np.histogram(original_distances[:, i], bins=bins)
        predicted_hist, _ = np.histogram(predicted_distances[:, i], bins=bins)

        original_hists.append(original_hist)
        predicted_hists.append(predicted_hist)

    # Sum over all joints
    original_total = np.sum(original_hists, axis=0)
    predicted_total = np.sum(predicted_hists, axis=0)

    # Append total number of bin counts to the last
    original_hists = np.stack(original_hists + [original_total], axis=1)
    predicted_hists = np.stack(predicted_hists + [predicted_total], axis=1)

    num_bins = bins.size - 1
    for i in range(num_bins):
        original_line = str(bins[i])
        predicted_line = str(bins[i])
        for j in range(num_joints + 1):
            original_line += ',' + str(original_hists[i, j])
            predicted_line += ',' + str(predicted_hists[i, j])
        original_line += '\n'
        predicted_line += '\n'
        original_out_lines.append(original_line)
        predicted_out_lines.append(predicted_line)

    original_out_dir = os.path.join(args.out, args.original)
    predicted_out_dir = os.path.join(args.out, args.predicted)

    if args.visualize:
        plt.plot(bins[:-1], original_total, label=args.original)
        plt.plot(bins[:-1], predicted_total, label=args.predicted)
        plt.legend()
        xlabel = 'Velocity (cm/s)' if args.measure == 'velocity' else 'Acceleration (cm / $s^2$)'
        plt.xlabel(xlabel)
        plt.ylabel('Bin counts')
        plt.title('Histograms of Moving Distance ({})'.format(args.measure))
        plt.tight_layout()
        plt.show()

    save_result(original_out_lines, original_out_dir,
                args.width, args.measure)
    save_result(predicted_out_lines, predicted_out_dir,
                args.width, args.measure)

    print('HMD ({}):'.format(args.measure))
    print('bins: {}'.format(bins))
    print('original: {}'.format(original_total))
    print('predicted: {}'.format(predicted_total))


if __name__ == '__main__':
    main()
