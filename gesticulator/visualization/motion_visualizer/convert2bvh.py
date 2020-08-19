import numpy as np
import subprocess
import os
import sys

from pymo.writers import BVHWriter
import joblib


def write_bvh(datapipe_file, anim_clip, filename, fps):
    data_pipeline = joblib.load(datapipe_file[0])
    inv_data = data_pipeline.inverse_transform(anim_clip)
    writer = BVHWriter()
    for i in range(0, anim_clip.shape[0]):
        with open(filename, "w") as f:
            writer.write(inv_data[i], f, framerate=fps)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: python convert2bvh.py <path to data_pipe file> <path to prediction file>"
        )
        sys.exit(0)

    datapipe_file = sys.argv[1]
    pred_file = sys.argv[2]
    print("data pipline: " + datapipe_file)
    print("prediction file: " + pred_file)

    jt_data = np.load(pred_file)
    if jt_data.ndim == 2:
        jt_data = np.expand_dims(jt_data, axis=0)
    out_filename = os.path.splitext(os.path.basename(pred_file))[0] + ".bvh"
    print("writing:" + out_filename)
    write_bvh(datapipe_file, jt_data[:, :, :], out_filename, fps=20)

