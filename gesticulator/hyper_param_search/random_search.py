import os
from argparse import ArgumentParser

import numpy as np

import ray
import torch
from gesticulator.model.model import GesticulatorModel
from ray import tune
from ray.tune import track
from gesticulator.hyper_param_search.ray_trainable import TrainableTrainer, MyEarlyStopping
from gesticulator.hyper_param_search.schedulers import MyAsyncHyperBandScheduler, MyFIFOScheduler

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


if __name__ == "__main__":
    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument("--gpus", default=[0, 1, 2, 3], help="how many gpus")
    parent_parser.add_argument("--dev_test", action="store_true")

    parser = GesticulatorModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    MAX_ITERATIONS = 12

    ray.init(num_cpus=26, local_mode=hyperparams.dev_test)

    schdlr = MyAsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_loss",
        mode="min",
        max_t=3,
        reduction_factor=4,
    )
    es = MyEarlyStopping(patience=2, max_loss=0.5)

    analysis = tune.run(
        TrainableTrainer,
        stop=lambda trial_id, result: es(trial_id, result),
        scheduler=schdlr,
        config={
            "batch_size": tune.choice([32, 64, 128, 256]),
            "learning_rate": tune.sample_from(
                        lambda spec: 10 ** (- np.random.randint(3, 6))),
            "n_layers": tune.randint(1, 4),
            "speech_enc_frame_dim": tune.sample_from(lambda _: np.random.randint(2, 15) * 10),
            "full_speech_enc_dim": tune.sample_from(lambda _: np.random.randint(3, 9) * 124),
            "condition_dim": tune.sample_from(lambda _: np.random.randint(3, 13) * 124),
            "dropout": tune.sample_from(lambda _: np.random.randint(30) / 100),
            "poses_enc_eff_dim" : tune.sample_from(lambda _: np.random.randint(3,32) ),
            "n_prev_poses": tune.sample_from(lambda _: np.random.randint(1,4) ),

            "vel_coef": tune.sample_from(lambda _: np.random.randint(50) / 100),
            "var_coef": tune.sample_from(lambda _: np.random.randint(10) / 100),
            "hour_glass" : tune.choice([True, False]),
            "only_diff" : tune.choice([True, False]),
            
            # Fixed params
            "hparams": hyperparams
                                                                                                                                                                                                                                                                                                        },
        checkpoint_freq=1,
        resources_per_trial={"cpu": 1, "gpu": 0.5},
        local_dir=hyperparams.logdir,
        num_samples=150,
        verbose=2,
        raise_on_failed_trial=False,
    )

    print("Best config: ", analysis.get_best_config(metric="mean_loss"))
    import pdb

    pdb.set_trace()
