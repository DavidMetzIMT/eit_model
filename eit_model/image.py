from dataclasses import dataclass

import numpy as np

@dataclass
class EITImage(object):
    data:np.ndarray=np.array([])
    label:str=''
    fem:dict=None


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files
    import glob_utils.log.log
    glob_utils.log.log.main_log()
