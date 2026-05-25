
"""
chest image preprocessor：
    window position、histogram equalization, size normalization
    1.histogram equalization enhances the contrast between bones and soft tissues
    2.unifor the size for the inputting
"""

import cv2
import numpy as np


class ChestXrayPreprocessor:
    def __init__(self,image_size=(512,512)):
        self._image_size = image_size

    def __call__(self, img:np.ndarray) -> np.ndarray:
        """
        img: RGB numpy array(H,W,3) from cv2.imread
        returns:
            RGB numpy array (_image_size[0],_image_size[1],3)
        """
        # grey equalization
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        # merge the three channels，keep RGB format
        processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

        # adjust the size
        processed = cv2.resize(processed, self._image_size,interpolation = cv2.INTER_LINEAR)

        return processed