#*utils - typically comprise helper functions
import cv2#massive software library used for real-time computer vision, image processing, and machine learning tasks
import numpy as np
import gymnasium as gym


class AtariPreprocess(gym.ObservationWrapper):#converts the raw game image into a smaller grayscale image, modifies the observation

    def __init__(self, env):
        super().__init__(env)#Without it, the wrapper would not properly know what environment it is wrapping            ????????????????

        self.observation_space = gym.spaces.Box(#"After preprocessing, observations will have this format."
            low=0,
            high=255,
            shape=(84,84),
            dtype=np.uint8#uint8 - unsigned integer, 8 bits per value; stores numbers: 0 → 255
        )


    def observation(self, obs):

        obs = cv2.cvtColor(
            obs,
            cv2.COLOR_RGB2GRAY#Convert RGB to grayscale
        )#Instead of Pixel: R = 200, G = 150, B = 100; you get: Pixel brightness: 157

        obs = cv2.resize(#Resizing reduces resolution, Cropping removes parts of the image
            obs,
            (84,84),
            interpolation=cv2.INTER_AREA#It averages nearby pixels rather than simply deleting them.
        )

        return obs