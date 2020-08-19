from motion_visualizer.read_bvh import read_bvh_to_array
import numpy as np
import argparse


def convert_bvh2npy(bvh_file, npy_file):
    # read the coordinates
    all_coords = read_bvh_to_array(bvh_file)

    # remove fingers and lower body
    torso = all_coords[:,0:8]
    right_arm = all_coords[:,9:12]
    left_arm = all_coords[:,28:31]

    modeled_coords = np.concatenate((torso, right_arm, left_arm), axis=1)

    np.save(npy_file, modeled_coords)


if __name__ == "__main__":

    # Parse command line params
    parser = argparse.ArgumentParser(
        description="Transforming BVH file into an array of feature-vectors"
    )

    # Folders params
    parser.add_argument(
        "--bvh_file",
        "-bvh",
        default="/home/tarask/Documents/Experimental_Results/GENEA/GT_results/TestSeq001.bvh",
        help="Address to the bvh file",
    )
    parser.add_argument(
        "--npy_file", "-npy", default="ges_test.npy", help="Address to the npy file"
    )

    args = parser.parse_args()

    convert_bvh2npy(args.bvh_file, args.npy_file)
