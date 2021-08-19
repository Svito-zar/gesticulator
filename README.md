[![Explanation video](https://Svito-zar.github.io/assets/gesticulator.png)](https://youtu.be/VQ8he6jjW08)

This repository contains PyTorch based implementation of the ICMI 2020 Best Paper Award recipient paper [Gesticulator: A framework for semantically-aware speech-driven gesture generation](https://svito-zar.github.io/gesticulator/).

## 0. Set up

### Requirements
- python3.6+
- ffmpeg (for visualization)

### Installation
**NOTE**: during installation, there will be several error messages (one for bert-embedding and one for mxnet) about conflicting packages - please ignore them, they don't affect the functionality of the repository.

- Clone the repository:
  ```
  git clone git@github.com:Svito-zar/gesticulator.git
  ```
- (optional) Create and activate virtual environment:
  ```
  virtualenv gest_env --py=3.6.9
  source gest_env/bin/activate
  ```
  or 
  ```
  conda create -n gest_env python=3.6.9
  conda activate gest_env
  ```
  
- Install the dependencies:
  ```
  python install_script.py
  ```

### Demonstration
Head over to the [demo](https://github.com/Svito-zar/gesticulator/tree/master/demo) folder for a quick demonstration if you're not interested in training the model yourself.

### Documentation
For all the scripts which we refer to in this repo description there are several command line arguments which you can see by calling them with the `--help` argument.

### Loading and saving models
- Pretrained model files can be loaded with the following command
  ```
  from gesticulator.model.model import GesticulatorModel
  
  loaded_model = GesticulatorModel.load_from_checkpoint(<PATH_TO_MODEL_FILE>)
  ```
- If the `--save_model_every_n_epochs argument` is provided to `train.py`, then the model will be saved regularly during training. 

___
## Training the model
### 1. Obtain the data
- Sign the license for the [Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/)
- Obtain training data from the `GENEA_Challenge_2020_data_release` folder of the [Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/), using the acquired credentials:
  ```
  cd dataset
  mkdir genea_data && cd genea_data
  
  # Change USERNAME to the actual username you received for the dataset
  wget --user USERNAME --ask-password -r -np -nH --cut-dirs=2 -R index.html* https://trinityspeechgesture.scss.tcd.ie/data/Trinity%20Speech-Gesture%20I/GENEA_Challenge_2020_data_release/ 
  ```

### 2.1 Rename and move files
```
# rename files from the GENEA Challenge names to the Trinity Speech-Gesture dataset naming
python rename_data_files.py

# Go back to the gesticulator/gesticulator directory
cd ..
```

### 2.2 Pre-process the data
```
cd gesticulator/data_processing

# encode motion from BVH files into exponensial map representation
python bvh2features.py
# ( this will take a while)

# Split the dataset into training and validation
python split_dataset.py

# Encode all the features
python process_dataset.py

# Go back to the gesticulator/gesticulator directory
cd ..
```

By default, the model expects the dataset in the `dataset/raw_data` folder, and the processed dataset will be available in the `dataset/processed_data folder`. If your dataset is elsewhere, please provide the correct paths with the `--raw_data_dir` and `--proc_data_dir` command line arguments.

### 3. Learn speech- and text-driven gesture generation model
In order to train the model, run
```
python train.py 
```
The model configuration and the training parameters are automatically read from the `gesticulator/config/default_model_config.yaml` file. 

#### Notes

The results will be available in the `results/last_run/` folder, where you will find the Tensorboard logs alongside with the trained model file. 

It is possible to visualize the predicted motion on the validation data during training by setting the `save_val_predictions_every_n_epoch` parameter in the config file.

If the `--run_name <name>` command-line argument is provided, the `results/<name>` folder will be created and the results will be stored there. This can be very useful when you want to keep your logs and outputs for separate runs.

To **train the model on the GPU**, provide the `--gpus` argument [as described here](https://pytorch-lightning.readthedocs.io/en/0.8.4/trainer.html#gpus). For details regarding training parameters, please [visit this link](https://pytorch-lightning.readthedocs.io/en/0.8.4/trainer.html).
___
## Evaluating the model
### Visualizing the results
In order to generate and visualize gestures on the test dataset, run

```
python evaluate.py --use_semantic_input --use_random_input
```

If you set the `run_name` argument during training, then please provide the path to the saved model checkpoint by using the `--model_file` option.

The generated motion is stored in the `results/<run_name>/generated_gestures` folder 1) in the exponential map format 2) as `.mp4` videos and 3) as 3D coordinates (which can be used for objective evaluation).

For nice visualization you can use the following repository: https://github.com/jonepatr/genea_visualizer

### Quantitative evaluation

For the quantitative evaluation (velocity histograms and jerk), you may use the scripts in the `gesticulator/obj_evaluation` folder.

## Citing

If you use this code in your research please cite it:
```
@inproceedings{kucherenko2020gesticulator,
  title={Gesticulator: A framework for semantically-aware speech-driven gesture generation},
  author={Kucherenko, Taras and Jonell, Patrik and van Waveren, Sanne and Henter, Gustav Eje and Alexanderson, Simon and Leite, Iolanda and Kjellstr{\"o}m, Hedvig},
  booktitle={Proceedings of the ACM International Conference on Multimodal Interaction},
  year={2020}
}
```

For using the dataset used in this work, please don't forget to cite [Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/) and [GENEA Gesture Generation Challenge](https://svito-zar.github.io/GENEAchallenge2020/) using the following bib files:
```
@inproceedings{ferstl2018investigating,
author = {Ferstl, Ylva and McDonnell, Rachel},
title = {Investigating the Use of Recurrent Motion Modelling for Speech Gesture Generation},
year = {2018},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
booktitle = {Proceedings of the 18th International Conference on Intelligent Virtual Agents},
series = {IVA '18}
}

@inproceedings{kucherenko2021large,
  author = {Kucherenko, Taras and Jonell, Patrik and Yoon, Youngwoo and Wolfert, Pieter and Henter, Gustav Eje},
  title = {A Large, Crowdsourced Evaluation of Gesture Generation Systems on Common Data: {T}he {GENEA} {C}hallenge 2020},
  year = {2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3397481.3450692},
  booktitle = {26th International Conference on Intelligent User Interfaces},
  pages = {11--21},
  numpages = {11},
  keywords = {evaluation paradigms, conversational agents, gesture generation},
  location = {College Station, TX, USA},
  series = {IUI '21}
}
```


## Contact
If you have any questions - please use the [Discussion](https://github.com/Svito-zar/gesticulator/discussions) tab.

If you encounter any problems/bugs/issues please create an issue on Github.
