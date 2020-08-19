import tempfile
import argparse
import numpy as np
import os

from motion_visualizer.model_animator import create_video
from pymo.writers import BVHWriter
from motion_visualizer.read_bvh import read_bvh_to_array
from motion_visualizer.convert2bvh import write_bvh


__DEFAULT_DAT_FILE_LOCATION = os.path.join(
    os.path.dirname(__file__), "data", "data_pipe.sav"
)


def convert_bvh2npy(bvh_file, npy_file):
    # read the coordinates
    coords = read_bvh_to_array(bvh_file)

    # shorten the sequence
    output_vectors = coords[:6000]

    np.save(npy_file, output_vectors)


def visualize(
    input_array,
    output_path,
    start_t=0,
    end_t=10,
    fps=20,
    data_pipe_path=__DEFAULT_DAT_FILE_LOCATION,
):
    assert isinstance(input_array, np.ndarray), "input_array must be a numpy array"
    with tempfile.NamedTemporaryFile(
        suffix=".bhv"
    ) as bvh_file, tempfile.NamedTemporaryFile(suffix=".npy") as npy_file:
        write_bvh(data_pipe_path, input_array, bvh_file.name, fps)

        # Extract 3D coordinates
        convert_bvh2npy(bvh_file.name, npy_file.name)

        # Visualize those 3D coordinates
        create_video(npy_file.name, output_path, start_t, end_t)


if __name__ == "__main__":
    # Parse command line params
    parser = argparse.ArgumentParser(description="Create video from npy file")

    parser.add_argument(
        "input", help="Motion output from the network (in quaterinions)"
    )
    parser.add_argument("output_path", help="Output path for video")

    # Video params
    parser.add_argument("--fps", default=20, help="Video fps")
    parser.add_argument(
        "--start_t", "-st", default=0, help="Start time for the sequence"
    )
    parser.add_argument("--end_t", "-end", default=10, help="End time for the sequence")

    # Misc params
    parser.add_argument(
        "--data_pipe_path",
        default=__DEFAULT_DAT_FILE_LOCATION,
        help="Data pipeline path",
    )

    args = parser.parse_args()

    input_data = np.load(args.input)
    if input_data.ndim == 2:
        input_data = np.expand_dims(input_data, axis=0)

    visualize(
        input_data,
        args.output_path,
        start_t=args.start_t,
        end_t=args.end_t,
        fps=args.fps,
        data_pipe_path=args.data_pipe_path,
    )
