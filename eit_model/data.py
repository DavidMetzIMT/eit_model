from dataclasses import dataclass

import numpy as np

from eit_model import fwd_model

@dataclass
class EITData(object):
    meas:np.ndarray=np.array([])
    label:str=''

    @property
    def ref_frame(self)-> np.ndarray:
        return self.meas[:,0]

    @property
    def frame(self)-> np.ndarray:
        return self.meas[:,1]
