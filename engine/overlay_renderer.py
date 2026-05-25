
"""
overlay the segmentation mask onto the original image to generate the final result
image.

"""
import cv2
import numpy as np


class OverlayRenderer:
    """
    overlay a semi-transparent color mask of the scapula bone  onto the original image.
    in commercial products, the following are typically required：
    -- Controllable transparency
    -- Clear category legends
    -- Optional area statistics information
    """

    SCAPULA_COLOR = (0,0,255) # color of the scapula, high lighted

    def __init__(self,opacity: float = 0.4,line_thickness: int = 2):
        self._opacity = opacity
        self._line_thickness = line_thickness

    def set_opacity(self,opacity: float):
        self._opacity = opacity
        
    def render(self,original_image: np.ndarray,mask: np.ndarray,
               draw_contour: bool = True, draw_legend: bool = True,draw_area: bool = True) -> np.ndarray:
        """
        :param original_image: original chest image
        :param mask: separated mask
        :param draw_contour: draw the contour or not
        :param draw_legend: draw the legend or not
        :param draw_area: display area or not
        :return:
            RGB IMAGE OVERLAY
        """
        # make the mask size as the same with the original
        if mask.shape[:2] != original_image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (original_image.shape[1], original_image.shape[0]),
                interpolation = cv2.INTER_NEAREST
            ).astype(bool)

        result = original_image.copy()

        # semi-transparent mask overlay
        overlay = result.copy()
        overlay[mask > 0] = self.SCAPULA_COLOR
        result = cv2.addWeighted(result,1.0,overlay,self._opacity,0)

        # contour
        if draw_contour:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(result,contours,-1,self.SCAPULA_COLOR,self._line_thickness)

        # legend
        if draw_legend:
            h,w = result.shape[:2]
            legend_x, legend_y = 20, h - 50
            cv2.rectangle(
                result,(legend_x,legend_y),(legend_x+24,legend_y+20),self.SCAPULA_COLOR,-1
            )
            cv2.putText(result,"Scapula (should blade)",
                        (legend_x + 32,legend_y+16),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2
            )

        # area
        if draw_area:
            area_px = int(mask.sum())
            h,w = result.shape[:2]
            cv2.putText(result,f"area: {area_px}",
                        (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2
            )

        return result