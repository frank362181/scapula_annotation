
"""
prediction engine： integration processing -> prediction model -> post processing -> visualization
"""
from typing import Union, Dict, Optional

import cv2
import numpy as np

from mmseg.apis import init_model, inference_model
from mmseg.structures import SegDataSample

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

    @property
    def render(self) -> OverlayRenderer:
        return self._render

    def run(self,image: Union[str,np.ndarray],return_overlay: Optional[bool] = True) -> Dict:
        """
        split a single image and return the result
        :
            image: path of image or RGB numpy array
            return_overlay:
                generate the overlay image or not
        :
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
        # return value from mmsegmentation inference_model
        result = inference_model(self._model, processed)

        # --- pick up predict mask(from SegDataSample)，single or multi
        if isinstance(result, SegDataSample):
            # process the single image
            result_sample = result
        elif hasattr(result,'__iter__') and not isinstance(result, (str,np.ndarray)):
            # batch process
            result_list = list(result)
            if len(result_list) == 0:
                raise RuntimeError(f"return the NONE from the predicted result!")
            result_sample = result_list[0]
        else:
            try:
                result_sample = SegDataSample.from_dict(result)
            except Exception as e:
                raise TypeError(f"can not get result from model:{result}!") from e

        # type 1 (scapula)
        scapula_mask = self._extract_mask(result_sample) #(pred_mask == 1).astype(np.uint8)

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

    """ ================= private function & methods ================"""
    def _extract_mask(self, result_sample: SegDataSample) -> np.ndarray:
        pred_sem_seg = result_sample.pred_sem_seg
        if pred_sem_seg is None:
            raise ValueError(f"check the config: non-output from the model!")

        # get tensor from data and convert it to numpy
        pred_data = pred_sem_seg.data # shape
        if isinstance(pred_data, np.ndarray):
            pred_mask = pred_data
        else:
            pred_mask = pred_data.cpu().numpy()

        # remove the batch dim,convert to (num_classes,H,W)
        pred_mask = pred_mask.squeeze(0)

        if pred_mask.ndim == 3:
            scapula_mask = pred_mask[1]
        else:
            # (H,W) single channel ,check the value is 1 or not
            scapula_mask = (pred_mask == 1).astype(np.uint8)

        scapula_mask = (scapula_mask > 0.5).astype(np.uint8)
        return scapula_mask
