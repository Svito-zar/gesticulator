"""
This file is used to visualize the results in a simple stick figure

authors: Pieter Wolfert, Taras Kucherenko
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

import argparse


class ModelSkeletons:
    def __init__(self, data_path, PCA, start_t, end_t):

        skeletons = np.load(data_path)

        # Shorten if needed
        if start_t != -1:
            skeletons = skeletons[start_t * 20 : end_t * 20]  # sec -> min (20 fps)

        # apply PCA if needed
        if PCA is not None:
            skeletons = PCA.inverse_transform(skeletons)

        self.skeletons = skeletons

    def getSkeletons(self):
        return self.skeletons


class AnimateSkeletons:
    """Animate plots for drawing Taras' skeleton sequences in 2D."""

    def __init__(self, skeletons):
        """Instantiate an object to visualize the generated poses."""
        self.fig = plt.figure()

        """ Set plot boundaries accordingly to the data ranges """
        x = skeletons[:, :, 0]
        xmin = x.min()
        xmax = x.max()
        y = skeletons[:, :, 1]
        ymin = y.min()
        ymax = y.max()
        self.ax = plt.axes(
            xlim=(xmin - 10, xmax + 10), ylim=(ymin - 10, ymax + 10)
        )
        self.ax.axis("off")
        self.line_one = self.ax.plot([], [], lw=5, c="b", marker="s")[0]
        self.line_two = self.ax.plot([], [], lw=5, c="b", marker="s")[0]
        self.line_three = self.ax.plot([], [], lw=5, c="b", marker="s")[0]

    def initLines(self):
        """Initialize the lines for plotting the limbs."""
        self.line_one.set_data([], [])
        self.line_two.set_data([], [])
        self.line_three.set_data([], [])

        return (
            self.line_one,
            self.line_two,
            self.line_three
        )

    def animateframe(self, skeleton):
        """Animate frame plot with two arms."""

        # Torso
        self.line_one.set_data(skeleton[0:8, 0], skeleton[0:8, 1])
        # Right arm
        x = np.concatenate(([skeleton[4, 0]], skeleton[8:11, 0]))
        z = np.concatenate(([skeleton[4, 1]], skeleton[8:11, 1]))
        self.line_two.set_data(x, z)
        # Left arm
        x = np.concatenate(([skeleton[4, 0]], skeleton[11:, 0]))
        z = np.concatenate(([skeleton[4, 1]], skeleton[11:, 1]))
        self.line_three.set_data(x, z)

        return (
            self.line_one,
            self.line_two,
            self.line_three,
        )

    def animate(self, frames_to_play):
        """Return an animation object that can be saved as a video."""
        anim = animation.FuncAnimation(
            self.fig,
            self.animateframe,
            init_func=self.initLines,
            frames=frames_to_play,
            blit=True,
        )

        return anim


def create_video(input_file, output_file, start=-1, end=20, pca=None):

    ml = ModelSkeletons(input_file, pca, start, end)
    skeletons = ml.getSkeletons()

    am = AnimateSkeletons(skeletons)
    am.initLines()

    anim = am.animate(frames_to_play=skeletons)
    anim.save(output_file, writer="ffmpeg", fps=20)


if __name__ == "__main__":
    # Parse command line params

    parser = argparse.ArgumentParser(description="Visualize 3d coords. seq. into a video file")
    
    parser.add_argument("--input", "-d", default="ges_test.npy",
                        help="Path to the input file with the motion")
    parser.add_argument("--out", "-o", default="Result.mp4",
                        help="Path to the output file with the video")
    parser.add_argument("--start", "-s", default=-1, type=int,
                        help="Start time (in sec)")
    parser.add_argument("--end", "-e", default=20, type=int, 
                        help="End time (in sec)")

    args = parser.parse_args()

    ml = ModelSkeletons(args.input, None, args.start, args.end)
    skeletons = ml.getSkeletons()

    am = AnimateSkeletons(skeletons)
    am.initLines()

    anim = am.animate(frames_to_play=skeletons)
    anim.save(args.out, writer="ffmpeg", fps=20)