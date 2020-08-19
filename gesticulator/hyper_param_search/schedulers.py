import logging
import numpy as np

from ray.tune.trial import Trial
from ray.tune.schedulers import FIFOScheduler, TrialScheduler, AsyncHyperBandScheduler

logger = logging.getLogger(__name__)


class ASHAv2(FIFOScheduler):
    """Implements the Async Successive Halving with better termination."""

    def __init__(
        self,
        time_attr="training_iteration",
        reward_attr=None,
        metric="episode_reward_mean",
        mode="max",
        max_t=100,
        grace_period=1,
        reduction_factor=4,
        brackets=1,
    ):
        assert max_t > 0, "Max (time_attr) not valid!"
        assert max_t >= grace_period, "grace_period must be <= max_t!"
        assert grace_period > 0, "grace_period must be positive!"
        assert reduction_factor > 1, "Reduction Factor not valid!"
        assert brackets > 0, "brackets must be positive!"
        assert mode in ["min", "max"], "`mode` must be 'min' or 'max'!"

        if reward_attr is not None:
            mode = "max"
            metric = reward_attr
            logger.warning(
                "`reward_attr` is deprecated and will be removed in a future "
                "version of Tune. "
                "Setting `metric={}` and `mode=max`.".format(reward_attr)
            )

        FIFOScheduler.__init__(self)
        self._reduction_factor = reduction_factor
        self._max_t = max_t

        # Tracks state for new trial add
        self._brackets = [
            _Bracket(grace_period, max_t, reduction_factor, s) for s in range(brackets)
        ]
        self._counter = 0  # for
        self._num_stopped = 0
        self._metric = metric
        if mode == "max":
            self._metric_op = 1.0
        elif mode == "min":
            self._metric_op = -1.0
        self._time_attr = time_attr
        self._num_paused = 0

    def on_trial_result(self, trial_runner, trial, result):
        action = TrialScheduler.CONTINUE
        if self._time_attr not in result or self._metric not in result:
            return action
        if result[self._time_attr] >= self._max_t:
            action = TrialScheduler.STOP
        else:
            bracket = self._brackets[0]
            action = bracket.on_result(
                trial, result[self._time_attr], self._metric_op * result[self._metric]
            )
        if action == TrialScheduler.STOP:
            self._num_stopped += 1
        if action == TrialScheduler.PAUSE:
            self._num_paused += 1
        return action

    def on_trial_complete(self, trial_runner, trial, result):
        if self._time_attr not in result or self._metric not in result:
            return
        bracket = self._brackets[0]
        bracket.on_result(
            trial,
            result[self._time_attr],
            self._metric_op * result[self._metric],
            complete=True,
        )

    def choose_trial_to_run(self, trial_runner):
        for bracket in self._brackets:
            for trial in bracket.promotable_trials():
                if trial and trial_runner.has_resources(trial.resources):
                    assert trial.status == Trial.PAUSED
                    logger.warning(f"Promoting trial [{trial.config}].")
                    bracket.unpause_trial(trial)
                    return trial
        trial = FIFOScheduler.choose_trial_to_run(self, trial_runner)
        if trial:
            self._brackets[0].unpause_trial(trial)
            logger.info(f"Choosing trial {trial.config} to run from trialrunner.")
        return trial

    def debug_string(self):
        out = "Using ASHAv2: num_stopped={}".format(self._num_stopped)
        out += "\n" + "\n".join([b.debug_str() for b in self._brackets])
        return out


class _Bracket:
    """Bookkeeping system to track the cutoffs.

    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.

    Example:
        >>> b = _Bracket(1, 10, 2, 3)
        >>> b.on_result(trial1, 1, 2)  # CONTINUE
        >>> b.on_result(trial2, 1, 4)  # CONTINUE
        >>> b.cutoff(b._rungs[-1][1]) == 3.0  # rungs are reversed
        >>> b.on_result(trial3, 1, 1)  # STOP
        >>> b.cutoff(b._rungs[0][1]) == 2.0
    """

    def __init__(self, min_t, max_t, reduction_factor, s):
        self.rf = reduction_factor
        MAX_RUNGS = int(np.log(max_t / min_t) / np.log(self.rf) - s + 1)
        self._rungs = [
            (min_t * self.rf ** (k + s), {}, []) for k in reversed(range(MAX_RUNGS))
        ]

    def cutoff(self, recorded):
        if len(recorded) < self.rf:
            return None
        return np.percentile(list(recorded.values()), (1 - 1 / self.rf) * 100)

    def top_k_ids(self, recorded):
        entries = list(recorded.items())
        k = int(len(entries) / self.rf)
        top_rung = sorted(entries, key=lambda kv: kv[1], reverse=True)[0:k]
        print("TOP RUNG:", top_rung)
        return [tid for tid, value in top_rung]

    def on_result(self, trial, cur_iter, cur_rew, complete=False):
        action = TrialScheduler.CONTINUE
        if cur_rew is None:
            logger.warning(
                "Reward attribute is None! Consider"
                " reporting using a different field."
            )
            return action
        for milestone, recorded, paused in self._rungs:
            if cur_iter < milestone or trial.trial_id in recorded:
                continue
            else:
                recorded[trial.trial_id] = cur_rew
                top_k_trial_ids = self.top_k_ids(recorded)
                if complete or trial.status != Trial.RUNNING:
                    break
                if trial.trial_id not in top_k_trial_ids:
                    action = TrialScheduler.PAUSE
                    paused += [trial]
                break
        if action == TrialScheduler.PAUSE:
            print(trial, cur_iter)
        return action

    def debug_str(self):
        iters = " | ".join(
            [
                "Iter {:.3f}: {} [{} paused]".format(
                    milestone, self.cutoff(recorded), len(paused)
                )
                for milestone, recorded, paused in self._rungs
            ]
        )
        return "Bracket: " + iters

    def promotable_trials(self):
        for _, recorded, paused in self._rungs:
            for tid in self.top_k_ids(recorded):
                paused_trials = {p.trial_id: p for p in paused}
                if tid in paused_trials:
                    yield paused_trials[tid]

    def unpause_trial(self, trial):
        for _, _, paused in self._rungs:
            if trial in paused:
                paused.pop(paused.index(trial))
            assert trial not in paused


class MyAsyncHyperBandScheduler(ASHAv2):
    def on_trial_add(self, trial_runner, trial):
        #if trial.config["hidden_size"] > 400 or trial.config["batch_size"] > 128:
        #    trial.update_resources(cpu=4, gpu=2)
        return super().on_trial_add(trial_runner, trial)


class MyFIFOScheduler(FIFOScheduler):
    def on_trial_add(self, trial_runner, trial):
        #if trial.config["hidden_size"] > 400 or trial.config["batch_size"] > 128:
        #    trial.update_resources(cpu=4, gpu=2)
        return super().on_trial_add(trial_runner, trial)
