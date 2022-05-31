from dataclasses import dataclass, field
from typing import Any, Tuple, Union
import numpy as np
import eit_model.fwd_model
import eit_model.model



@dataclass
class EITData(object):
    """EITData are the data used from the solvers to reconstruct EITimage
    
    """
    ref_frame:np.ndarray
    frame:np.ndarray
    ds: np.ndarray
    label: str = ""

    def __post_init__(self):
        self.ref_frame=self.ref_frame.flatten()
        self.frame=self.frame.flatten()
        self.ds=self.ds.flatten()

        if self.ref_frame.shape!=self.frame.shape or self.ref_frame.shape!=self.ds.shape:
            raise TypeError(f'Wrong shape {self.ref_frame.shape=},{self.frame.shape=},{self.ds.shape=}')

@dataclass
class EITImage(object):
    """EITimage is a reconstruct eit image from the solver"""
    data: np.ndarray
    label: str
    nodes: np.ndarray
    elems: np.ndarray
    elec_pos: np.ndarray

    def get_data_for_plot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the nodes, elems and the elements data 
        e.g. for plotting purpose

        Returns:
            Tuple[np.ndarray, np.ndarray,np.ndarray]: self.nodes, self.elems, self.data
        """
        return self.nodes, self.elems, self.data

    @property
    def is_3D(self):
        return self.elems.shape[1] == 4

def build_EITImage(
    data: np.ndarray = None,
    label: str = "",
    model: Union[eit_model.fwd_model.FEModel,eit_model.model.EITModel] = None
) -> EITImage:
    """
    Return EITimage object with corresponding data
    
    The image can be build from an FEModel or an EITmodel
    if data is None or not passed the image.data are set to 1
    """
    if model is None:
        raise ValueError("'model' not passed")
    if isinstance(model, eit_model.fwd_model.FEModel):
        fem= model
    elif isinstance(model, eit_model.model.EITModel):
        fem=model.fem
    else:
        raise TypeError("argument 'model' should be FEModel or EITmodel")

    return EITImage(
        data=fem.format_perm(data) if data is not None else fem.get_elems_data(),
        label=label,
        nodes=fem.nodes,
        elems=fem.elems,
        elec_pos= fem.elec_pos_orient()
    )


@dataclass
class EITVoltMonitoring(object):
    """_summary_

    volt_ref (ndarray): array of eit voltages of each electrode of the model
        of shape(n_exc, n_elec), dtype = complex
    volt_frame (ndarray): array of eit voltages of each electrode of the model
        of shape(n_exc, n_elec), dtype = complex

    """
    volt_ref: np.ndarray 
    volt_frame: np.ndarray
    labels:Any=''


@dataclass
class EITMeasVoltage(object):
    """EITMeasVoltage of a frame 

    Args:
        volt (ndarray): array of eit voltages of each electrode of the model
        of shape(n_exc, n_elec), dtype = complex
        meas (ndarray): measurements corresponding to the measurente pattern 
        defined in the model of shape(n_meas, ), dtype = complex
        frame_name (str) : frame name , default to 'Frame #x'
        frame_freq (str) : frame frequency , default to 'Frequency xkHz'

    """    
    volt: np.ndarray 
    meas: np.ndarray
    frame_name:str= 'Frame #x'
    frame_freq:str= 'Frequency xkHz'

    def __post_init__(self):
        self.meas=self.meas.flatten()


# @dataclass
# class EITVoltageLabels(object):
#     """Gathers informations about an eit voltages 

#     Args:
#         frame_idx (int): frame indx
#         freq (float): frame frequency in Hz
#         lab_frame_idx (str): frame indx label string
#         lab_frame_freq (str): frame frequency label string

#     """    
#     frame_idx:int # frame indx
#     freq:float # frame frequency in Hz
#     lab_frame_idx:str # frame indx label string
#     lab_frame_freq:str # frame frequency label string


@dataclass
class EITFrameMeasuredChannelVoltage(object):
    """EITMeasVoltage of a frame 

    Args:
        volt (ndarray): array of eit voltages of each electrode of the model
        of shape(n_exc, n_ch), dtype = complex
        name (str) : frame name , default to 'Frame #x'
        freq (str) : frame frequency , default to 'Frequency xkHz'

    """    
    volt: np.ndarray 
    name:str= 'Frame #x'
    freq:str= 'Frequency xkHz'

@dataclass
class EITReconstructionData(object):
    """

    Args:
        ref_frame: EITFrameMeasuredChannelVoltage
        meas_frame: EITFrameMeasuredChannelVoltage
    """
    ref_frame: EITFrameMeasuredChannelVoltage
    meas_frame: EITFrameMeasuredChannelVoltage


@dataclass
class EITMeasMonitoringData(object):
    """_summary_

    volt_frame= dict of ndarray of shape (n_exc, n_ch) dtype = complex

    """

    volt_frame: dict [Any, np.ndarray]= field(default_factory=dict) #list[np.ndarray] 
    # frame_idx: list[int] = field(default_factory=[])

    def add(self, volt, frame_idx):
        self.volt_frame[frame_idx]=volt
        # self.frame_idx.append(frame_idx)


if __name__ == "__main__":

    import glob_utils.log.log

    glob_utils.log.log.main_log()
