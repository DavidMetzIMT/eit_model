from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from eit_model import fwd_model


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
    data: np.ndarray = np.array([])
    label: str = ""
    fem: dict = None

    def __init__(
        self, data: np.ndarray = None, label: str = "", fem: fwd_model.FEModel = None
    ) -> None:

        self.data = fem.format_perm(data) if data is not None else fem.elems_data
        self.fem = {
            "nodes": fem.nodes,
            "elems": fem.elems,
            "elec_pos": fem.elec_pos_orient(),
        }
        self.label = label

    def get_data_for_plot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the nodes, elems and the elements for plotting
        purpose for example

        Returns:
            Tuple[np.ndarray, np.ndarray,np.ndarray]: _description_
        """
        return self.fem["nodes"], self.fem["elems"], self.data


@dataclass
class EITMeasMonitoring(object):
    """_summary_

    volt_frame= dict of ndarray of shape (n_exc, n_ch) dtype = complex

    """

    volt_frame: dict [int, np.ndarray]= field(default_factory=dict) #list[np.ndarray] 
    # frame_idx: list[int] = field(default_factory=[])

    def add(self, volt, frame_idx):
        self.volt_frame[frame_idx]=volt
        # self.frame_idx.append(frame_idx)


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files
    import glob_utils.log.log

    glob_utils.log.log.main_log()
