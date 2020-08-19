import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch

from gesticulator.model.model import GesticulatorModel
from pytorch_lightning import Trainer
from test_tube import HyperOptArgumentParser
from ray import tune
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ray.tune.schedulers import AsyncHyperBandScheduler
import ray
from ray.tune.resources import Resources

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(config):
    hparams = Namespace()
    for key, val in config.items():
        setattr(hparams, key, val)

    model = GesticulatorModel(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.model_path,
        save_best_only=True,
        verbose=True,
        monitor="avg_val_loss",
        mode="min",
    )

    trainer = Trainer(
        gpus=len(ray.get_gpu_ids()),
        distributed_backend=hparams.distributed_backend,
        max_nb_epochs=20,
        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(model)

if __name__ == "__main__":
    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument("--gpus", default=[0, 1, 2,3,4,5], help="how many gpus")
    parent_parser.add_argument(
        "--distributed_backend",
        type=str,
        default="dp",
        help="supports three options dp, ddp, ddp2",
    )

    parser = GesticulatorModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    config = {}
    for hparam, val in vars(hyperparams).items():
        if isinstance(val, list):
            config[hparam] = tune.sample_from(val)
        else:
            config[hparam] = val

    class MyAsyncHyperBandScheduler(AsyncHyperBandScheduler):
        def on_trial_error(self, trial_runner, trial):
            if trial.resources.gpu < 4:
                trial.resources = Resources(
                    cpu=trial.resources.cpu * 2, gpu=trial.resources.gpu * 2
                )
            super().on_trial_error(trial_runner, trial)

    async_hb_scheduler = MyAsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_loss",
        mode="min",
        max_t=13,
        grace_period=7,
    )

    ray.init(num_cpus=16)
    analysis = tune.run(
        main,
        scheduler=async_hb_scheduler,
        config={
            "batch_size": tune.choice([32, 64, 128, 256]),
            "learning_rate": tune.sample_from(
                lambda spec: 10 ** (- np.random.randint(3, 6))
            ),
            "n_layers": tune.randint(1, 4),
            "dropout": tune.sample_from(lambda _: np.random.randint(25) / 100),
            "speech_enc_frame_dim": tune.sample_from(lambda _: np.random.randint(5, 15) * 10),
            "full_speech_enc_dim": tune.sample_from(lambda _: np.random.randint(4, 9) * 124),
            "condition_dim": tune.sample_from(lambda _: np.random.randint(4, 13) * 124),
            "poses_enc_eff_dim" : tune.sample_from(lambda _: np.random.randint(5,32) ),
            "n_prev_poses": tune.sample_from(lambda _: np.random.randint(1,4) ),

            "vel_coef": tune.sample_from(lambda _: np.random.randint(50) / 100),
            "var_coef": tune.sample_from(lambda _: np.random.randint(10) / 100),
            "hour_glass" : tune.choice([True, False]),

            # Fixed params
            "activation": hyperparams.activation,
            "first_l_sz": hyperparams.first_l_sz,
            "second_l_sz": hyperparams.second_l_sz,
            "third_l_sz": hyperparams.third_l_sz,
            "sequence_length": hyperparams.sequence_length,
            "past_context": hyperparams.past_context,
            "future_context": hyperparams.future_context,
            "use_pca": hyperparams.use_pca,
            "recurrent_speech_enc": hyperparams.recurrent_speech_enc,
            "audio_dim": hyperparams.audio_dim,
            "text_context": hyperparams.text_context,
            "text_embedding": hyperparams.text_embedding,
            "data_dir": hyperparams.data_dir,
            "model_path": hyperparams.model_path,

            "distributed_backend": hyperparams.distributed_backend,
            "gpus": hyperparams.gpus
        },


        resources_per_trial={"cpu": 1, "gpu": 0.5},
        local_dir="./logdir", #hyperparams.logdir,
        num_samples=120,
    )

    print("Best config: ", analysis.get_best_config(metric="mean_loss"))
    import pdb

    pdb.set_trace()
