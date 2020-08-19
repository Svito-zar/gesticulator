import os
from argparse import Namespace
from collections import defaultdict

import numpy as np

from gesticulator.model.model import GesticulatorModel
import ray
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ray import tune
import joblib

torch.set_default_tensor_type('torch.FloatTensor')

class TrainableTrainer(tune.Trainable):
    def _setup(self, config):
        self.hparams = config["hparams"]  # Namespace()
        for key, val in config.items():
            if key == "hparams":
                continue
            try:
                val = val.item()
            except AttributeError:
                pass

            setattr(self.hparams, key, val)

        self.model = GesticulatorModel(self.hparams)

        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.logdir, "checkpoint"),
            save_best_only=True,
            verbose=True,
            monitor="avg_val_loss",
            mode="min",
        )

        try:
            gpus = len(ray.get_gpu_ids())
        except:
            print("failed to get gpus")
            gpus = 1

        self.trainer = Trainer(
            gpus=gpus,
            distributed_backend="dp",
            max_nb_epochs=1,
            checkpoint_callback=checkpoint_callback,
            nb_sanity_val_steps=2,
            log_gpu_memory="all",
            weights_summary=None,
            early_stop_callback=None,
            # show_progress_bar=False,
            train_percent_check=0.00001 if self.hparams.dev_test else 1,
        )
        self.val_loss = float("inf")

    def _train(self):
        self.trainer.fit(self.model)
        self.val_loss = self.trainer.callback_metrics["avg_val_loss"]
        return {"mean_loss": self.val_loss}

    def _generate_video(self):
        print("generating video!")

        seq_len = 300
        text_len = int(seq_len/2)

        # read data
        dev_dir = "/home/tarask/Documents/storage/SpeechToMotion/Irish/WithTextV5/dev_inputs"
        speech_data = np.load(dev_dir + "/X_test_NaturalTalking_01.npy")[:seq_len]
        text =  np.load(dev_dir + "/T_test_NaturalTalking_01.npy")[:text_len]

        # upsample text to get the same sampling rate as the audio
        cols = np.linspace(0, text.shape[0], endpoint=False, num=text.shape[0]*2, dtype=int)
        text_data = text[cols,:]

        # Convert to float tensors and put on GPU
        speech = torch.tensor([speech_data]).float().cuda()
        text =  torch.tensor([text_data]).float().cuda()
        # Text on validation sequences without teacher forcing
        predicted_gesture = self.model.forward(speech, text, condition=True, motion = None, teacher=False)

        """if self.hparams.pca_model:
            pca_output = pca.inverse_transform(val.reshape(-1, self.hparams.pose_dim))
            output = pca_output.reshape(val.shape[0], val.shape[1], -1)
        else:
            output = val"""

        gen_dir = "/home/tarask/Documents/Code/CVPR2020/gesticulator/log/gestures/"
        ges_file = gen_dir+ self.logdir[88:95]+".npy"
        np.save(ges_file, predicted_gesture.detach().cpu().numpy())
        print("Writing into: ",gen_dir+ self.logdir[88:95]+".npy")

    def _stop(self):
        if self.val_loss < 0.3 and self.iteration >= 4:
            self._generate_video()

    def _save(self, tmp_checkpoint_dir):
        print("Saving!")
        return {}

    def _restore(self, checkpoint):
        print("Restoring!")


class MyEarlyStopping(object):
    def __init__(self, patience=2, max_loss=None, max_epochs=None):
        self.patience = patience
        self.max_loss = max_loss
        self.max_epochs = max_epochs
        self.wait = defaultdict(int)
        self.best = defaultdict(lambda: float("inf"))

    def __call__(self, trial_id, result):
        val_loss = result["mean_loss"]
        epochs = result["training_iteration"]

        if val_loss < self.best[trial_id]:
            self.best[trial_id] = val_loss
            self.wait[trial_id] = 0
        else:
            self.wait[trial_id] += 1
            if self.wait[trial_id] >= self.patience:
                return True

        if self.max_epochs and epochs >= self.max_epochs:
            return True

        if self.max_loss and val_loss > self.max_loss:
            return True

        return False

