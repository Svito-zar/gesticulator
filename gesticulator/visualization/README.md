# Visualization folder

In order to visualize the result of the model you need to

1. Go through the script `generate_videos.py` and modify 4 place where I wrote "ToDo: "

2. Run the `generate_videos.py` script

------------------------------
use as a python lib
```
from cvpr20_visualizer.visualize import visualize
import numpy as np

motion_data = np.load('test_files/ex1_joint_angles.npy')

visualize(motion_data, 'output.mp4')
```