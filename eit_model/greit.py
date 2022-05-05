import copy
import logging

import numpy as np

from eit_model.data import EITImage
logger = logging.getLogger(__name__)







def greit_filter(image: EITImage, threshold: float = None, div:float=4.0) -> EITImage:
    image_n = copy.deepcopy(image)
    image_n.data = image.data.copy()
    # print(f"{image_n.data}")
    # print(f"{image.data}")

    # TODO test on normalized data and
    
    #
    #
    # not normalized data!!

    threshold_max = max(image_n.data)/div
    threshold_min = -max(-image_n.data)/div
    # logger.debug(f"{threshold_max=}{threshold_min=}")
    mask_max = image_n.data > threshold_max
    # logger.debug(f"{mask_max=}")
    image_n.data[mask_max] = 1
    mask_min = image_n.data < threshold_min
    # logger.debug(f"{mask_min=}")
    image_n.data[mask_min] = -1
    image_n.data[~np.logical_or(mask_max, mask_min)] = 0
    # logger.debug(f"{~np.logical_or(mask_max, mask_min)=}")
    return image_n
