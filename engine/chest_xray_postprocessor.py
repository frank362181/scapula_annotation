import cv2
import numpy as np


class ChestXrayPostprocessor:
    """
    utilize anatomical priori to filter out incorrect analysis:
        - the scapula is only present in the lateral region of the lung apex(
            upper 60% of the image + the outer bands on both sides)
        _ retain the largest connected domain
    """
    def __init__(self,min_area = 500, max_area = 50000,position_ratio = 0.6):
        self._min_area = min_area
        self._max_area = max_area
        self._position_ratio = position_ratio


    def __call__(self,mask) -> np.ndarray:
        """
        :param mask:
            bin-value mask（H，W），byte values: 0，1
        :return:
            bin-value mask filtered
        """
        h,w = mask.shape

        # 1. position filter：scapula positions the upper of 60% always
        vertical_mask = np.zeros_like(mask)
        valid_region_top = int(h * self._position_ratio)
        vertical_mask[:valid_region_top,:] = 1
        mask = mask & vertical_mask

        # 2. area filter
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity = 8
        )
        filter_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if self._min_area <= area <= self._max_area:
                filter_mask[labels == 1] = 1

        return filter_mask