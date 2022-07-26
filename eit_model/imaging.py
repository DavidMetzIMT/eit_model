from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Union
import numpy as np
from eit_model.plot import CustomLabels, EITPlotsType
from eit_model.model import EITModel
from eit_model.data import (
    EITData,
    EITMeasMonitoringData,
    EITMeasVoltage,
    EITVoltMonitoring,
)
from glob_utils.args.check_type import checkinstance


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


def eit_imaging_types() -> list[str]:
    return list(IMAGING_TYPE.keys())


def eit_data_transformations() -> list[str]:
    return list(DATA_TRANSFORMATIONS.keys())[:4]


@dataclass
class Transformer:
    transform: str
    show_abs: bool
    _abs: str = field(init=False)
    _transform_funcs: list[Callable] = field(init=False)

    def __post_init__(self):

        if self.transform not in DATA_TRANSFORMATIONS:
            raise Exception(f"The transformation {self.transform} unknown")

        self._abs = "Abs" if self.show_abs else "Identity"
        self._transform_funcs = [
            DATA_TRANSFORMATIONS[self.transform],
            DATA_TRANSFORMATIONS[self._abs],
        ]

    def get_label_trans(self) -> str:

        for key, func in DATA_TRANSFORMATIONS.items():
            if func == self._transform_funcs[0]:
                trans_label = key

        return trans_label

    def add_abs_bars(self, lab: str) -> str:

        if DATA_TRANSFORMATIONS["Abs"] == self._transform_funcs[1]:
            lab = f"||{lab}||"
        return lab

    def run(self, x: np.ndarray) -> np.ndarray:
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
        return x


class EITImaging(ABC):

    transformer: Transformer = None
    label_imaging: str = ""
    label_meas = None
    lab_data: str = ""
    _type_imaging: str = ""  # should be defined in postinit
    lab_ref_idx: str = ""
    lab_ref_freq: str = ""
    lab_frm_idx: str = ""
    lab_frm_freq: str = ""

    def __init__(self, transform: str, show_abs: bool) -> None:
        super().__init__()
        self.transformer = Transformer(transform, show_abs)
        self._post_init_()

    @abstractmethod
    def _post_init_(self):
        """Custom initialization"""
        # label_imaging: str = ""

    def compute_imaging_data(
        self, v_ref: EITMeasVoltage, v_frame: EITMeasVoltage
    ) -> Tuple[EITData, str, dict[EITPlotsType, CustomLabels]]:

        self.lab_data, self.plot_lab = self.get_metadata(v_ref, v_frame)
        eit_data = self.build_eit_data(v_ref, v_frame)
        print(eit_data)
        return eit_data, self.lab_data, self.plot_lab

    # @abstractmethod
    def build_eit_data(self, v_ref: EITMeasVoltage, v_meas: EITMeasVoltage) -> EITData:
        """"""
        data_ref = self.transformer.run(v_ref.meas)
        data_meas = self.transformer.run(v_meas.meas)
        data_ds = data_meas - data_ref
        return EITData(data_ref, data_meas, data_ds, self.lab_data)

    def get_metadata(self, v_ref: EITMeasVoltage, v_meas: EITMeasVoltage):
        """provide all posible metadata for ploting"""

        self.lab_ref_idx = v_ref.frame_name
        self.lab_ref_freq = v_ref.frame_freq
        self.lab_frm_idx = v_meas.frame_name
        self.lab_frm_freq = v_meas.frame_freq

        trans_label = self.transformer.get_label_trans()

        self.label_meas = [f"{trans_label}(U)", f"{trans_label}({self.label_imaging})"]
        self.label_meas = [
            self.transformer.add_abs_bars(lab) for lab in self.label_meas
        ]

        return self.make_EITplots_labels()

    def get_protocol_info(self) -> list[str]:
        """
        Return a list of string containing informaton saved in
        the analysis protocol

        Returns:
            list[str]: lines of informationss
        """

        return [
            f"Type: {self._type_imaging}",
            f"Data transformation: {self.transformer.transform}, show_abs: {self.transformer.show_abs}",
            f"{self.lab_frm_freq}",
            f"Ref. {self.lab_ref_idx}",
            f"Ref. {self.lab_ref_freq}",
        ]

    @abstractmethod
    def make_EITplots_labels(self) -> Union[str, dict[EITPlotsType, CustomLabels]]:
        """
        _summary_

        Returns:
            Union[str, dict[EITPlotsType, CustomLabels]]:
            -
        """


class AbsoluteImaging(EITImaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "U"
        self._type_imaging = "Absolute imaging"

    def build_eit_data(self, v_ref: EITMeasVoltage, v_meas: EITMeasVoltage) -> EITData:
        """Redfinition for Absolute"""
        data_ref = self.transformer.run(v_ref.meas)
        data_meas = self.transformer.run(v_meas.meas)
        return EITData(data_ref, data_meas, data_meas, self.lab_data)

    def make_EITplots_labels(self) -> Union[str, dict[EITPlotsType, CustomLabels]]:

        # self.check_data(1, 1)
        t = f"({self.label_meas[1]});"
        lab_data = f"Absolute Imaging {t} {self.lab_frm_idx} ({self.lab_frm_freq})"
        plot_lab = {
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
        return lab_data, plot_lab


class TimeDifferenceImaging(EITImaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "\u0394U_t"  # ΔU_t
        self._type_imaging = "Time difference imaging"

    def make_EITplots_labels(self) -> Union[str, dict[EITPlotsType, CustomLabels]]:

        t = f"({self.label_meas[1]}); {self.lab_ref_idx} - {self.lab_frm_idx} ({self.lab_frm_freq})"
        lab_data = f"Time difference Imaging {t}"

        plot_lab = {
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
        return lab_data, plot_lab


class FrequenceDifferenceImaging(EITImaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "\u0394U_f"  # ΔU_f
        self._type_imaging = "Frequence difference imaging"

    def make_EITplots_labels(self) -> Union[str, dict[EITPlotsType, CustomLabels]]:

        t = f" ({self.label_meas[1]}); {self.lab_ref_freq} - {self.lab_frm_freq} ({self.lab_frm_idx})"
        lab_data = f"Frequency difference Imaging {t}"

        plot_lab = {
            EITPlotsType.Image_2D: CustomLabels(
                f"Frequency difference Imaging {t}",
                ["", ""],
                ["X", "Y", "Z"],
            ),
            EITPlotsType.U_plot: CustomLabels(
                f"{self.label_meas[0]} ",
                [
                    f"Ref  {self.lab_ref_idx} ({self.lab_ref_freq})",
                    f"Meas {self.lab_frm_idx} ({self.lab_frm_freq})",
                ],
                ["Measurements", "Voltages in [V]"],
            ),
            EITPlotsType.U_plot_diff: CustomLabels(
                f"{self.label_meas[1]}",
                [f"{self.lab_frm_freq} - {self.lab_ref_freq} ({self.lab_frm_idx})", ""],
                ["Measurements", "Voltage differences in [V]"],
            ),
        }
        return lab_data, plot_lab


class ChannelVoltageMonitoring(EITImaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "U_ch"
        self._type_imaging = "ChannelVoltageImaging"

    def make_EITplots_labels(self) -> Union[str, dict[EITPlotsType, CustomLabels]]:

        t = f" ({self.label_meas[1]});"
        lab_data = f"Channel Voltages {t} {self.lab_frm_idx} ({self.lab_frm_freq})"

        plot_lab = {
            EITPlotsType.U_plot: CustomLabels(
                f"Channel Voltages {t}",
                [
                    f"Ref  {self.lab_ref_idx} ({self.lab_ref_freq})",
                    f"Meas {self.lab_frm_idx} ({self.lab_frm_freq})",
                ],
                ["Measurements", "Voltages in [V]"],
            )
        }
        return lab_data, plot_lab


IMAGING_TYPE: dict[str, EITImaging] = {
    "Absolute imaging": AbsoluteImaging,
    "Time difference imaging": TimeDifferenceImaging,
    "Frequence difference imaging": FrequenceDifferenceImaging,
}


def build_EITImaging(eit_imaging: str, transform: str, show_abs: bool) -> EITImaging:
    """Set ei imaging mode for reconstruction"""
    checkinstance(eit_imaging, str)

    return IMAGING_TYPE[eit_imaging](transform, show_abs)


if __name__ == "__main__":
    """"""
