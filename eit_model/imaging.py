from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np
from eit_model.plot import CustomLabels, EITPlotsType
from eit_model.model import EITModel
from eit_model.data import EITData, EITMeasMonitoring, EITVoltMonitoring, EITVoltage


def identity(x: np.ndarray) -> np.ndarray:
    """Return the passed ndarray x
    used for the transformation of the voltages
    """
    return x


DATA_TRANSFORMATIONS = {
    "Real": np.real,
    "Image": np.imag,
    "Magnitude": np.abs,
    "Phase": np.angle,
    "Abs": np.abs,
    "Identity": identity,
}

def eit_imaging_types()->list[str]:
    return list(IMAGING_TYPE.keys())

def eit_data_transformations()->list[str]:
    return list(DATA_TRANSFORMATIONS.keys())[:4]

@dataclass
class Transformer:
    transform: str
    show_abs: bool
    _transform_funcs:list[Callable]=field(init=False)

    def __post_init__(self):

        if self.transform not in DATA_TRANSFORMATIONS:
            raise Exception(f"The transformation {self.transform} unknown")

        self._transform_funcs = [
            DATA_TRANSFORMATIONS[self.transform],
            DATA_TRANSFORMATIONS["Abs"]
            if self.show_abs
            else DATA_TRANSFORMATIONS["Identity"],
        ]

    def get_label_trans(self)->str:

        for key, func in DATA_TRANSFORMATIONS.items():
            if func == self._transform_funcs[0]:
                trans_label = key

        return trans_label

    def add_abs_bars(self, lab:str)->str:

        if DATA_TRANSFORMATIONS["Abs"] == self._transform_funcs[1]:
            lab= f"||{lab}||"
        return lab

    def run(self,x: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x (np.ndarray): _description_
            transform_funcs (list): _description_

        Raises:
            Exception: _description_

        Returns:
            np.ndarray: _description_
        """
        if len(self._transform_funcs) != 2:
            raise Exception()

        for func in self._transform_funcs:
            if func is not None:
                x = func(x)
        x = np.reshape(x, (x.shape[0], 1))
        return x


class Imaging(ABC):

    transformer = [identity, identity]
    label_imaging: str = ""
    label_meas = None
    lab_data: str = ""


    def __init__(self, transform: str, show_abs: bool) -> None:
        super().__init__()
        self.transformer = Transformer(transform, show_abs)
        self._post_init_()

    @abstractmethod
    def _post_init_(self):
        """Custom initialization"""
        # label_imaging: str = ""

    def process_data(
        self,
        v_ref: EITVoltage, #np.ndarray = None
        v_meas: EITVoltage, #np.ndarray = None
        # labels=None,
        eit_model: EITModel = None,
    ) -> Tuple[EITData,EITVoltMonitoring, dict[EITPlotsType, CustomLabels]]:

        self.get_metadata(v_ref, v_meas)
        meas_voltages, volt_ref, volt_frame = self.transform_voltages(v_ref, v_meas, eit_model)
        return EITData(meas_voltages, self.lab_data), EITVoltMonitoring(volt_ref, volt_frame, self.lab_data), self.make_EITplots_labels()

    # @abstractmethod
    def transform_voltages(
        self, v_ref: EITVoltage, v_meas: EITVoltage, eit_model: EITModel
    ) -> np.ndarray:
        """"""
        meas_ref, volt_ref= eit_model.get_meas_voltages(v_ref.volt)
        meas_meas, volt_meas= eit_model.get_meas_voltages(v_meas.volt)

        meas_ref_t=self.transformer.run(meas_ref)
        meas_meas_t=self.transformer.run(meas_meas)

        return np.hstack((meas_ref_t, meas_meas_t, meas_meas_t -meas_ref_t)), volt_ref, volt_meas

    def get_metadata(self, v_ref: EITVoltage, v_meas: EITVoltage):
        """provide all posible metadata for ploting"""

        self.lab_ref_idx = v_ref.get_frame_name()
        self.lab_ref_freq = v_ref.get_frame_freq()
        self.lab_frm_idx = v_meas.get_frame_name()
        self.lab_frm_freq = v_meas.get_frame_freq()

        trans_label = self.transformer.get_label_trans()

        self.label_meas = [f"{trans_label}(U)", f"{trans_label}({self.label_imaging})"]
        self.label_meas = [
            self.transformer.add_abs_bars(lab) for lab in self.label_meas
        ]


    @abstractmethod
    def make_EITplots_labels(self) -> dict[EITPlotsType, CustomLabels]:

        """"""

class AbsoluteImaging(Imaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "U"

    def transform_voltages(
        self, v_ref: EITVoltage, v_meas: EITVoltage, eit_model: EITModel
    ) -> np.ndarray:
        """"""
        meas_ref, volt_ref= eit_model.get_meas_voltages(v_ref.volt)
        meas_meas, volt_meas= eit_model.get_meas_voltages(v_meas.volt)

        meas_ref_t=self.transformer.run(meas_ref)
        meas_meas_t=self.transformer.run(meas_meas)

        return np.hstack((meas_ref_t, meas_meas_t, meas_meas_t)), volt_ref, volt_meas


    def make_EITplots_labels(self) -> dict[EITPlotsType, CustomLabels]:

        # self.check_data(1, 1)
        t = f"({self.label_meas[1]});"
        self.lab_data= f"Absolute Imaging {t} {self.lab_frm_idx} ({self.lab_frm_freq})"
        return {
            EITPlotsType.Image_2D: CustomLabels(
                f"Absolute Imaging {t}",
                ["", ""],
                ["X", "Y"],
            ),
            EITPlotsType.U_plot: CustomLabels(
                f"Voltages {t}",
                [f"Meas {self.lab_frm_idx} ({self.lab_frm_freq})", ""],
                ["Measurements", "Voltages in [V]"],
            ),
            EITPlotsType.U_plot_diff: CustomLabels(
                f"Voltages {t}",
                ["", ""],
                ["Measurements", "Voltages in [V]"],
            ),
        }


class TimeDifferenceImaging(Imaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "\u0394U_t"  # ΔU_t


    def make_EITplots_labels(self) -> dict[EITPlotsType, CustomLabels]:

        t = f"({self.label_meas[1]}); {self.lab_ref_idx} - {self.lab_frm_idx} ({self.lab_frm_freq})"
        self.lab_data= f"Time difference Imaging {t}"

        return {
            EITPlotsType.Image_2D: CustomLabels(
                f"Time difference Imaging {t}",
                ["", ""],
                ["X", "Y", "Z"],
            ),
            EITPlotsType.U_plot: CustomLabels(
                f"Voltages ({self.label_meas[0]}); {self.lab_frm_freq}",
                [
                    f"Ref  {self.lab_ref_idx} ({self.lab_ref_freq})",
                    f"Meas {self.lab_frm_idx} ({self.lab_frm_freq})",
                ],
                ["Measurements", "Voltages in [V]"],
            ),
            EITPlotsType.U_plot_diff: CustomLabels(
                f"Voltage differences {t}",
                [f"{self.lab_ref_idx} - {self.lab_frm_idx} ({self.lab_frm_freq})", ""],
                ["Measurements", "Voltages in [V]"],
            ),
        }


class FrequenceDifferenceImaging(Imaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "\u0394U_f"  # ΔU_f

    def make_EITplots_labels(self) -> dict[EITPlotsType, CustomLabels]:

        t = (
            f" ({self.label_meas[1]}); {self.lab_ref_freq} - {self.lab_frm_freq} ({self.lab_frm_idx})"
        )
        self.lab_data= f"Frequency difference Imaging {t}"

        return {
            EITPlotsType.Image_2D: CustomLabels(
                f"Frequency difference Imaging {t}",
                ["", ""],
                ["X", "Y", "Z"],
            ),
            EITPlotsType.U_plot: CustomLabels(
                f"Voltages ({self.label_meas[0]}); {self.lab_frm_idx} ",
                [
                    f"Ref  {self.lab_ref_idx} ({self.lab_ref_freq})",
                    f"Meas {self.lab_frm_idx} ({self.lab_frm_freq})",
                ],
                ["Measurements", "Voltages in [V]"],
            ),
            EITPlotsType.U_plot_diff: CustomLabels(
                f"Voltage differences {t}",
                [f"{self.lab_ref_freq} - {self.lab_frm_freq} ({self.lab_frm_idx})", ""],
                ["Measurements", "Voltages in [V]"],
            ),
        }


class ChannelVoltageImaging(Imaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "U_ch"

    def make_EITplots_labels(self) -> dict[EITPlotsType, CustomLabels]:

        t = f" ({self.label_meas[1]});"
        self.lab_data= f"Channel Voltages {t} {self.lab_frm_idx} ({self.lab_frm_freq})"

        return {
            EITPlotsType.U_plot: CustomLabels(
                f"Channel Voltages {t}",
                [
                    f"Ref  {self.lab_ref_idx} ({self.lab_ref_freq})",
                    f"Meas {self.lab_frm_idx} ({self.lab_frm_freq})",
                ],
                ["Measurements", "Voltages in [V]"],
            )
        }


IMAGING_TYPE:dict[str, Imaging] = {
    "Absolute imaging": AbsoluteImaging,
    "Time difference imaging": TimeDifferenceImaging,
    "Frequence difference imaging": FrequenceDifferenceImaging,
}




if __name__ == "__main__":
    """"""
