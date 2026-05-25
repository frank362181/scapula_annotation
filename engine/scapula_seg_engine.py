
"""
prediction engine： integration processing -> prediction model -> post processing -> visualization
"""
from typing import Union, Dict

import cv2
import numpy as np

from mmseg.apis import init_model, inference_model

from engine.chest_xray_postprocessor import ChestXrayPostprocessor
from engine.chest_xray_preprocessor import ChestXrayPreprocessor
from engine.overlay_renderer import OverlayRenderer


class ScapulaSegEngine:
    def __init__(self,config_path: str, checkpoint_path: str, device: str = "cuda:0",image_size: tuple=(512,512),over_opacity: float = 0.4):
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._device = device
        self._image_size = image_size

        self._model = init_model(self._config_path,self._checkpoint_path,device=self._device)

        # initializes sub-module
        self._preprocessor = ChestXrayPreprocessor(image_size=self._image_size)
        self._postprocessor = ChestXrayPostprocessor()

        self._render = OverlayRenderer(opacity=over_opacity)

        # classification mapping
        self._class_names = {0: "background", 1: "scapula"}


    def run(self,image: Union[str,np.ndarray],return_overlay: bool = True) -> Dict:
        """
        split a single image and return the result
        :param image:
            path of image or RGB numpy array
            return_overlay:
                generate the overlay image or not
        :return:
            - "original":original image
            - "mask": split mask(H,W)/ 0/1
            - "overlay":overlay the image(BGR) OR None
            - "area_px": pix area of scapula
        """
        # 1 ---- loading original image--
        if isinstance(image, str):
            original_image = cv2.imread(image)
            if original_image is None:
                raise FileNotFoundError(f"can not read image:{image}!")
        else:
            original_image = image.copy()

        # 2 ---- preprocess the image
        processed = self._preprocessor(original_image)

        #3 ---- prediction ----
        result = inference_model(self._model, processed)

        # --- pick up predict mask(from SegDataSample)
        pred_mask = result.pred_sem_seg.data.cpu().numpy()
        pred_mask = pred_ask.squeeze(0)

        # type 1 (scapula)
        scapula_mask = (pred_mask == 1).astype(np.uint8)

        # ---- 4 post processing
        filtered_mask = self._postprocessor(scapula_mask)

        # ----5 overlay visualization
        overlay = None
        if return_overlay:
            overlay = self._render.render(original_image,filtered_mask)

        area_px = int(filtered_mask.sum())

        return {
            "original": original_image,
            "mask": filtered_mask,
            "overlay": overlay,
            "area_px": area_px,
        }