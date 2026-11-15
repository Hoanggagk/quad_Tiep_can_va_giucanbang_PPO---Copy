"""Compatibility wrapper for the 10-second velocity-NED profile.

The environment semantics live in :mod:`velocity_ned.env`; keeping a wrapper
prevents the short training entry point from drifting on timeout, success and
termination/truncation behavior.
"""

from drone_control.action_spaces.velocity_ned.env import (
    DroneHoveringEnv as _BaseDroneHoveringEnv,
)


class DroneHoveringEnv(_BaseDroneHoveringEnv):
    MAX_STEPS_PER_EPISODE = 100

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("max_steps_per_episode", self.MAX_STEPS_PER_EPISODE)
        super().__init__(**kwargs)
