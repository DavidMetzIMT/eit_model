from dataclasses import dataclass, field
from typing import Any, Tuple
import numpy as np
import eit_model.fwd_model


@dataclass
class EITData(object):
    meas: np.ndarray = np.array([])
    label: str = ""

    @property
    def ref_frame(self) -> np.ndarray:
        return self.meas[:, 0]

    @property
    def frame(self) -> np.ndarray:
        return self.meas[:, 1]

    @property
    def ds(self) -> np.ndarray:
        return self.meas[:, 2]


class EITImage(object):
    data: np.ndarray
    label: str
    nodes: np.ndarray
    elems: np.ndarray
    elec_pos: np.ndarray

    def __init__(
        self, data: np.ndarray = None, label: str = "", fem: eit_model.fwd_model.FEModel = None
    ) -> None:

        self.label = label

        # fem relevant data
        self.data = fem.format_perm(data) if data is not None else fem.elems_data
        self.nodes= fem.nodes
        self.elems= fem.elems
        self.elec_pos= fem.elec_pos_orient()

    def get_data_for_plot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the nodes, elems and the elements data 
        e.g. for plotting purpose

        Returns:
            Tuple[np.ndarray, np.ndarray,np.ndarray]: self.nodes, self.elems, self.data
        """
        return self.nodes, self.elems, self.data

@dataclass
class EITVoltMonitoring(object):
    """_summary_

    volt_frame= dict of ndarray of shape (n_exc, n_ch) dtype = complex

    """
    volt_ref: np.ndarray 
    volt_frame: np.ndarray
    labels:Any=''

@dataclass
class EITVoltageLabels(object):
    """_summary_

    volt= ndarray of shape (n_exc, n_ch) dtype = complex
    labels:

    """
    frame_idx:int # frame indx
    freq:float # frame frequency
    lab_frame_idx:str # frame indx label string
    lab_frame_freq:str # frame frequency label string


@dataclass
class EITVoltage(object):
    """eit voltages 

    volt (ndarray): array of eit voltages of shape(n_exc, n_ch) dtype = complex
    labels:EITVoltageLabels

    """
    volt: np.ndarray 
    labels:EITVoltageLabels
    
    def get_frame_name(self)->str:
        return self.labels.lab_frame_idx
    def get_frame_freq(self)->str:
        return self.labels.lab_frame_freq


@dataclass
class EITMeasMonitoring(object):
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
