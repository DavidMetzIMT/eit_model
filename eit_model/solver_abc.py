import abc
from typing import Any

import numpy as np
import enum
from eit_model.data import EITData
from eit_model.image import EITImage
from eit_model.model import EITModel
import glob_utils.flags.flag

class RecCMDs(enum.Enum):
    initialize=enum.auto()
    reconstruct=enum.auto()


class SolverNotReadyError(BaseException):
    """"""

class Solver(abc.ABC):

    eit_model:EITModel
    ready:glob_utils.flags.flag.CustomFlag

    def __init__(self, model:EITModel) -> None:
        super().__init__()
        
        self.eit_model= model
        self.ready=glob_utils.flags.flag.CustomFlag()

        self.cmd_func= {
            RecCMDs.initialize:self.prepare_rec,
            RecCMDs.reconstruct:self.rec
        }
        self.__post_init__()

    def run(self, cmd:RecCMDs, *args, **kwargs):
        if not isinstance(cmd, RecCMDs):
            return None
        return self.cmd_func[cmd](*args, **kwargs)

    @abc.abstractmethod
    def __post_init__(self)-> None:
        """Custom post initialization
        """

    def prepare_rec(self, rec_params:Any=None)-> tuple[EITImage, EITData]:
        """Prepare the solver to be ready for reconstruction      

        Returns:
            tuple[EITImage, EITData]: a reconstructed EIT image and the 
            corresponding EIT data used for it (random generated, simulated 
            or loaded...)
            rec_params[Any]: Reconstruction parameters
        """
        return self._custom_preparation(rec_params)

    @abc.abstractmethod
    def _custom_preparation(self, rec_params:Any=None)-> tuple[EITImage, EITData]:
        """Custom preparation of the solver to be ready for reconstruction      

        Returns:
            tuple[EITImage, EITData]: a reconstructed EIT image and the 
            corresponding EIT data used for it (random generated, simulated 
            or loaded...)
            rec_params[Any]: Reconstruction parameters
        """
    
    def rec(self, data:EITData)-> EITImage:
        """Reconstruction of an EIT image using EIT data/measurements

        Args:
            data (EITData): EIT data/measurements for the reconstruction 

        Raises:
            SolverNotReadyError: raised if the solver has not been previously
            correctly prepared/initializated

        Returns:
            EITImage: a reconstructed EIT image corresponding to the EIT 
            data/measurements
        """

        if not self.ready.is_set():
            raise SolverNotReadyError('Solver not ready yet, please run an init')

        return self._custom_rec(data)
        
    @abc.abstractmethod
    def _custom_rec(self, data:EITData)-> EITImage:
        """Custom reconstruction of an EIT image using EIT data/measurements

        Args:
            data (EITData): EIT data/measurements for the reconstruction 

        Returns:
            EITImage: a reconstructed EIT image corresponding to the EIT 
            data/measurements
        """

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files
    import glob_utils.log.log
    glob_utils.log.log.main_log()