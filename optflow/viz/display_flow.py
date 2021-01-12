import cv2
from ..flow import to_polar
import numpy as np

def flowHSV(flow:np.ndarray) -> np.ndarray:
    """
    Produces HSV visualization (converted to BGR color space for display)
    of normalized flow

    Inputs:
    - flow

    Outputs:
    - hsv
    """
    mag, ang = to_polar(flow)
    h,w,*_ = mag.shape

    hsv = np.zeros((h,w,3), dtype=np.uint8)
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return hsv