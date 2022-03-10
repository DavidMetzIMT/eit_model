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


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files
    import glob_utils.log.log
    glob_utils.log.log.main_log()
