import cv2
import numpy as np
import gymnasium as gym


class AtariPreprocess(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84,84),
            dtype=np.uint8
        )


    def observation(self, obs):

        obs = cv2.cvtColor(
            obs,
            cv2.COLOR_RGB2GRAY
        )

        obs = cv2.resize(
            obs,
            (84,84),
            interpolation=cv2.INTER_AREA
        )

        return obs