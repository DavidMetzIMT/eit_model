from dataclasses import dataclass

import numpy as np

@dataclass
class EITImage(object):
    data:np.ndarray=np.array([])
    label:str=''
    fem:dict=None
