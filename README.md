# Gesticulator: A framework for semantically-aware speech-driven gesture generation
This repository contains PyTorch based implementation of the framework for semantically-aware speech-driven gesture generation, which can be used to reproduce the experiments in the ICMI paper [Gesticulator](https://svito-zar.github.io/gesticulator/).


## 0. Set up

### Requirements
- python3.6+
- ffmpeg (for visualization)

### Installation
NOTE: during installation, there will be several error messages (one for bert-embedding and one for mxnet) about conflicting packages - those can be ignored.

```
git clone git@github.com:Svito-zar/gesticulator.git
cd gesticulator
pip install -r gesticulator/requirements.txt
pip install -e .
pip install -e gesticulator/visualization
```

### Documentation

For all the scripts which we refer to in this repo description there are several command line arguments which you can see by calling them with the `--help` argument.

## Demo
Head over to the `demo` folder for a quick demonstration if you're not interested in training the model yourself.


## Loading and saving models
- Pretrained model files can be loaded with the following command
  ```
  from gesticulator.model import GesticulatorModel
  
  loaded_model = GesticulatorModel.load_from_checkpoint(<PATH_TO_MODEL_FILE>)
  ```
- If the `--save_model_every_n_epochs argument` is provided to `train.py`, then the model will be saved regularly during training. 

## 1. Obtain the data
- Download the [Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/)
- Either obtain transcriptions by yourself:
  - Transcribe the audio using Automatic Speech Recognition (ASR), such as [Google ASR](https://cloud.google.com/speech-to-text/)
  - Manually correct the transcriptions and add punctuations
- Place the dataset in the `dataset` folder next to `gesticulator` folder in three subfolders: `speech`, `motion` and `transcript`.

## 2. Pre-process the data
```
cd gesticulator/data_processing

# encode motion from BVH files into exponensial map representation
python bvh2features.py

# Split the dataset into training and validation
python split_dataset.py

# Encode all the features
python process_dataset.py
```

By default, the model expects the dataset in the `dataset/raw` folder, and the processed dataset will be available in the `dataset/processed folder`. If your dataset is elsewhere, please provide the correct paths with the `--raw_data_dir` and `--proc_data_dir` command line arguments.

## 3. Learn speech-driven gesture generation model
In order to train and evaluate the model, run

```
cd ..
python train.py
```

The model configuration and the training parameters are automatically read from the `gesticulator/config/default_model_config.yaml` file. 

### Notes

The results will be available in the `results/last_run/` folder, where you will find the Tensorboard logs alongside with the trained model file and the generated output on the semantic test segments (described in the paper).

If the `--run_name <name>` command-line argument is provided, the `results/<name>` folder will be created and the results will be stored there. This can be very useful when you want to keep your logs and outputs for separate runs.

To train the model on the GPU, provide the `--gpus` argument. For details regarding training parameters, please [visit this link](https://pytorch-lightning.readthedocs.io/en/0.8.4/trainer.html#gpus).

## 4. Visualize gestures
The gestures generated during training, validation and testing can be found in the `results/<run_name>/generated_gestures` folder. By default, we only store the outputs on the semantic test segments, but other outputs can be saved as well - see the config file for the corresponding parameters.

In order to manually generate the the videos from the raw coordinates, run 

```
cd visualization/aamas20_visualizer
python generate_videos.py
```

If you changed the arguments of `train.py` (e.g. `run_name`), you might have to provide them for `generate_videos.py` as well.
Please check the required arguments by running

`python generate_videos.py --help`

## 5. Quantitative evaluation

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

For using the dataset I have used in this work, please don't forget to cite [Trinity Speech-Gesture dataset](https://trinityspeechgesture.scss.tcd.ie/) using their [IVA'18 paper](https://www.scss.tcd.ie/Rachel.McDonnell/papers/IVA2018b.pdf).

## Contact
If you encounter any problems/bugs/issues please contact me on Github or by emailing me at tarask@kth.se for any bug reports/questions/suggestions. I prefer questions and bug reports on Github as that provides visibility to others who might be encountering same issues or who have the same questions.
