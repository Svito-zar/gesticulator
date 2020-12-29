import os
from os import path
from time import sleep
from abc import ABC

import torch
import numpy as np
from joblib import load

from gesticulator.visualization.motion_visualizer.generate_videos import visualize

class PredictionSavingMixin(ABC):
    """
    A mixin for the Gesticulator class that provides the capability to save 
    model predictions during training, validation and testing. 
    
    Useful for tracking the model performance, because the 
    loss function doesn't capture it that well.
    """
    def init_prediction_saving_params(self):
        """Create the output directories and initialize the parameters."""
        # Convert the prediction durations to frames
        self.hparams.saved_prediction_duration_frames = \
            self.hparams.past_context \
            + self.hparams.future_context \
            + self.data_fps * self.hparams.saved_prediction_duration_sec
        
        if self.hparams.generated_gestures_dir is None:
            self.hparams.generated_gestures_dir = path.join(self.save_dir, "generated_gestures")
        
        # Check which phases is the prediction generation enabled in
        self.enabled_phases = []
        
        self.last_saved_train_prediction_epoch = 0
        self.last_saved_val_prediction_epoch = 0
        # NOTE: testing has no epochs 

        self.save_train_predictions = False
        self.save_val_predictions = False
        # Training
        if self.hparams.save_train_predictions_every_n_epoch > 0:
            self.enabled_phases.append("training")
            self.save_train_predictions = True

        # Validation
        if self.hparams.save_val_predictions_every_n_epoch > 0:
            self.enabled_phases.append("validation")
            self.save_val_predictions = True

        # Create the output directories
        for phase in self.enabled_phases: 
            for save_format in self.hparams.prediction_save_formats:
                    # make the directory name plural
                    save_format = save_format + 's' if not save_format.endswith('s') else save_format
                    save_dir_path = path.join(self.hparams.generated_gestures_dir, 
                        phase, save_format)
                    if not os.path.isdir(save_dir_path):
                        try:
                            os.makedirs(save_dir_path) # e.g. <results>/<run_name>/generated_gestures/test/videos
                        except:
                            print("-----------------------------------------------------------------------")
                            print(f"WARNING: cannot create '{save_format}' directory for saving model outputs.")
                            print("Perhaps the save formats are duplicated?")
                            print(f"\t(enabled formats: {self.hparams.prediction_save_formats})")
                            print("-----------------------------------------------------------------------")
                            sleep(1)

    def generate_training_predictions(self):
        """Predict gestures for the training input and save the results."""
        predicted_gestures = self.forward(
            audio = self.train_input['audio'],
            text = self.train_input['text'],
            use_conditioning=True, 
            motion=None).cpu().detach().numpy()

        if self.hparams.use_pca:
            pca = load('utils/pca_model_12.joblib')
            predicted_gestures = pca.inverse_transform(predicted_gestures)
      
        # Save the prediction
        self.save_prediction(predicted_gestures, "training")

    def generate_validation_predictions(self):
        """Predict gestures for the validation input and save the results."""
        predicted_gestures = self.forward(
            audio = self.val_input['audio'],
            text = self.val_input['text'],
            use_conditioning=True, 
            motion=None).cpu().detach().numpy()

        if self.hparams.use_pca:
            pca = load('utils/pca_model_12.joblib')
            predicted_gestures = pca.inverse_transform(predicted_gestures)
      
        self.save_prediction(predicted_gestures, "validation")

    def generate_test_predictions(self, mode):
        """
        Generate gestures either for the semantic or the random test input
        segments (depending on the mode argument), and save the results.
        """
        if mode == 'seman':
            # The start times of the semantic test segments in the paper
            segment_start_times = {
                # These correspond to the NaturalTalking_04/05 files
                '04': [55, 150, 215, 258, 320, 520, 531], 
                '05': [15, 53, 74, 91, 118, 127, 157, 168, 193, 220, 270, 283, 300] }
        elif mode == 'rand':
            # Random segment start times from the paper
            segment_start_times = {
                '04': [ 5.5, 20.8, 45.6, 66, 86.3, 106.5, 120.4, 163.7,
                        180.8, 242.3, 283.5, 300.8, 330.8, 349.6, 377 ],
                '05': [ 30, 42, 102, 140, 179, 205, 234, 253, 329, 345,
                        384, 402, 419, 437, 450 ] }
        else:
            print(f"Unknown test prediction mode '{mode}'! Possible values: 'seman' or 'rand'.")
            exit(-1)
            
        print(f"\nGenerating {mode} test gestures:", flush=True)
        segment_lengths_sec = 10
        duration_in_frames = segment_lengths_sec * self.data_fps \
                             + self.hparams.past_context \
                             + self.hparams.future_context 

        segment_idx = 0
        for file_num in segment_start_times.keys():
            audio_full = self.test_prediction_inputs[file_num]['audio']
            text_full = self.test_prediction_inputs[file_num]['text']

            for start_time in segment_start_times[file_num]:
                segment_idx += 1
                start_frame = int(start_time * self.data_fps - self.hparams.past_context)
                end_frame = start_frame + duration_in_frames
                # Crop and add the batch dimension
                audio = audio_full[start_frame:end_frame].unsqueeze(0) 
                text = text_full[start_frame:end_frame].unsqueeze(0)
                
                predicted_gestures = self.forward(
                    audio, text, use_conditioning=True, 
                    motion=None).cpu().detach().numpy()

                if self.hparams.use_pca:
                    pca = load('utils/pca_model_12.joblib')
                    predicted_gestures = pca.inverse_transform(predicted_gestures)
        
                filename = f"{mode}_{segment_idx}"
                print("\t-", filename)
                
                self.save_prediction(predicted_gestures, "evaluation", filename)
            
        print(f"Generated {mode} test predictions to {self.hparams.generated_gestures_dir}", flush=True)

    # ---- Private functions ----

    def load_train_or_val_input(self, input_array):
        """
        Load an input sequence that will be used during training or validation,
        and crop it to the given duration in the 'saved_prediction_duration_sec' hyperparameter.
        """
        audio = torch.as_tensor(input_array['audio'], device=self.device)
        text = torch.as_tensor(input_array['text'], device=self.device)
        
        # Crop the data to the required duration and add back the batch_dimension
        audio = audio[:self.hparams.saved_prediction_duration_frames].unsqueeze(0)
        text = text[:self.hparams.saved_prediction_duration_frames].unsqueeze(0)

        return {'audio': audio, 'text': text}

    def load_test_prediction_input(self, num):
        """Load a sequence from the test inputs for semantic test predictions."""
        audio = self.load_test_file('audio', num)
        text = self.load_test_file('text', num)
        text = self.upsample_text(text)

        return {'audio': audio, 'text': text }

    def load_test_file(self, file_type, num):
        """Load the tensor that will be used for generating semantic test predictions."""
        if file_type == 'audio':
            filename = f"X_test_NaturalTalking_0{num}.npy"
        elif file_type == 'text':
            filename = f"T_test_NaturalTalking_0{num}.npy"
        else:
            print("ERROR: unknown semantic test input type:", file_type)
            exit(-1)
        
        input_path = path.join(
            self.hparams.data_dir, "test_inputs", filename)

        input_tensor = torch.as_tensor(
            torch.from_numpy(np.load(input_path)), device=self.device)
        
        return input_tensor.float()

    def save_prediction(self, gestures, phase, filename = None):
        """
        Save the given gestures to the <generated_gestures_dir>/<phase> folder 
        using the formats found in hparams.prediction_save_formats.

        The possible formats are: BVH file, MP4 video and raw numpy array.

        Args:
            gestures:  The output of the model
            phase:  Can be "training", "validation" or "test"
            filename:  The filename of the saved outputs (default: epoch_<current_epoch>.<extension>)
        """
        if filename is None:
            filename = f"epoch_{self.current_epoch + 1}"
        
        enabled_save_paths, disabled_save_paths = \
            self.get_prediction_save_paths(phase, filename)
       
        data_pipe = path.join(os.getcwd(), 'utils/data_pipe.sav')
       
        if "raw_gesture" in enabled_save_paths.keys():
            np.save(enabled_save_paths["raw_gesture"], gestures)

        get_save_path = \
            lambda key: enabled_save_paths[key] if key in enabled_save_paths \
                                                else disabled_save_paths[key]
        visualize(
            gestures, 
            bvh_file = get_save_path("bvh"),
            mp4_file = get_save_path("video"),
            npy_file = get_save_path("3d_coordinates"),
            start_t = 0, 
            end_t = self.hparams.saved_prediction_duration_sec,
            data_pipe_dir = data_pipe)

        # Clean up the temporary files
        for temp_file in disabled_save_paths.values():
            os.remove(temp_file)

    def get_prediction_save_paths(self, phase, filename):
        """Return the output file paths for each possible format in which the gestures can be saved.
        
        Args:
            phase:  Can be "training", "validation" or "test"
            filename:  The filename without the file format extension
        
        Returns:
            enabled_format_paths:  a dictionary containing the save path for each enabled file format
            disabled_format_paths:  a dictionary containing the save path for each disabled file format
        """
        is_enabled = \
            lambda fmt : fmt in self.hparams.prediction_save_formats
        
        get_persistent_path = \
            lambda subdir, extension : path.join(
                self.hparams.generated_gestures_dir,
                phase, subdir, filename + extension)
        
        get_temporary_path = \
            lambda extension : path.join(
                self.hparams.generated_gestures_dir,
                phase, "temp" + extension)
                
        enabled_format_paths = {}
        disabled_format_paths = {}

        # BVH format
        if is_enabled("bvh_file"):
            enabled_format_paths["bvh"] = get_persistent_path("bvh_files", ".bvh")
        else:
            disabled_format_paths["bvh"] = get_temporary_path(".bvh")
        
        # Raw numpy array format
        if is_enabled("raw_gesture"):
            enabled_format_paths["raw_gesture"] = get_persistent_path("raw_gestures", "_exp_map.npy")      
        # NOTE: there's no need for a temporary path if the raw gestures are disabled
        #       because only the visualize() call creates temporary files

        # 3D coordinates
        if is_enabled("3d_coordinates"):
            enabled_format_paths["3d_coordinates"] = get_persistent_path("3d_coordinates", "_3d.npy")
        else:
            disabled_format_paths["3d_coordinates"] = get_temporary_path("_coordinates.npy")

        # Video format
        if is_enabled("video"):
            enabled_format_paths["video"] = get_persistent_path("videos", ".mp4")      
        else:
            disabled_format_paths["video"] = get_temporary_path(".mp4")
        
        return enabled_format_paths, disabled_format_paths
 
    def upsample_text(self, text):
        """Upsample the given text input with twice the original frequency (so that it matches the audio)."""  
        cols = np.linspace(0, text.shape[0], dtype=int, endpoint=False, num=text.shape[0] * 2)
        # NOTE: because of the dtype, 'cols' contains each index in 0:text.shape[0] twice
        
        return text[cols, :]

