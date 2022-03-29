from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from eit_model.plot import CustomLabels, EITPlotsType
from eit_model.model import EITModel
from eit_model.data import EITData


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




def make_voltage_vector(
    eit_model: EITModel,
    transform_funcs: list,
    voltages: np.ndarray,
    get_ch: bool = False,
) -> np.ndarray:
    """_summary_

    Args:
        eit_model (EITModel): _description_
        transform_funcs (list): _description_
        voltages (np.ndarray): shape(n_exc, n_channel)

    Returns:
        np.ndarray: _description_
    """
    if voltages is None:
        return np.array([])
    # get only the voltages of used electrode (0-n_el)
    meas_voltage = voltages[:, : eit_model.n_elec]
    # get the volgate corresponding to the meas_pattern and flatten
    meas = (
        meas_voltage.flatten()
        if get_ch
        else eit_model.meas_pattern(0).dot(meas_voltage.T).flatten()
    )


    return transform(meas, transform_funcs)


def transform(x: np.ndarray, transform_func: list) -> np.ndarray:
    """_summary_

    Args:
        x (np.ndarray): _description_
        transform_funcs (list): _description_

    Raises:
        Exception: _description_

    Returns:
        np.ndarray: _description_
    """
    if len(transform_func) != 2:
        raise Exception()

    for func in transform_func:
        if func is not None:
            x = func(x)
    x = np.reshape(x, (x.shape[0], 1))
    return x


class Imaging(ABC):

    transform_funcs = [identity, identity]
    label_imaging: str = ""
    label_meas = None
    get_channel: bool = False

    def __init__(self, transform: str, show_abs: bool) -> None:
        super().__init__()

        if transform not in DATA_TRANSFORMATIONS:
            raise Exception(f"The transformation {transform} unknown")

        transform_funcs = [
            DATA_TRANSFORMATIONS[transform],
            DATA_TRANSFORMATIONS["Abs"]
            if show_abs
            else DATA_TRANSFORMATIONS["Identity"],
        ]

        self.transform_funcs = transform_funcs

        self._post_init_()

    @abstractmethod
    def _post_init_(self):
        """Custom initialization"""
        # label_imaging: str = ""

    def process_data(
        self,
        v_ref: np.ndarray = None,
        v_meas: np.ndarray = None,
        labels=None,
        eit_model: EITModel = None,
    ) -> Tuple[EITData, dict[EITPlotsType, CustomLabels]]:

        self.get_metadata(labels)
        meas_voltages = self.transform_voltages(v_ref, v_meas, eit_model)
        return EITData(meas_voltages), self.make_labels()

    # @abstractmethod
    def transform_voltages(
        self, v_ref: np.ndarray, v_frame: np.ndarray, eit_model: EITModel
    ) -> List[np.ndarray]:
        """"""
        return np.hstack(
            (
                make_voltage_vector(
                    eit_model, self.transform_funcs, v_ref, self.get_channel
                ),
                make_voltage_vector(
                    eit_model, self.transform_funcs, v_frame, self.get_channel
                ),
                make_voltage_vector(
                    eit_model, self.transform_funcs, v_frame, self.get_channel
                )
                - make_voltage_vector(
                    eit_model, self.transform_funcs, v_ref, self.get_channel
                ),
            )
        )

    def get_metadata(self, labels):
        """provide all posible metadata for ploting"""

        self.lab_ref_idx = labels[0][0]
        self.lab_ref_freq = labels[0][1]
        self.lab_frm_idx = labels[1][0]
        self.lab_frm_freq = labels[1][1]

        for (
            key,
            func,
        ) in DATA_TRANSFORMATIONS.items():
            if func == self.transform_funcs[0]:
                trans_label = key

        self.label_meas = [f"{trans_label}(U)", f"{trans_label}({self.label_imaging})"]
        if DATA_TRANSFORMATIONS["Abs"] == self.transform_funcs[1]:
            self.label_meas = [f"||{lab}||" for lab in self.label_meas]

    @abstractmethod
    def make_labels(self) -> dict[EITPlotsType, CustomLabels]:

        """"""

    # def check_data(self, idx_frames_len, freqs_val_len):
    #     if len(self.idx_frames) != idx_frames_len:
    #         raise Exception(
    #             f"should be {idx_frames_len} frame idx idx_frames:{self.idx_frames}"
    #         )
    #     if len(self.freqs_val) != freqs_val_len:
    #         raise Exception(
    #             f"should be {freqs_val_len} freqences values freqs_val:{self.freqs_val}"
    #         )


class AbsoluteImaging(Imaging):
    def _post_init_(self):
        """Custom initialization"""
        self.label_imaging = "U"

    def transform_voltages(
        self, v_ref: np.ndarray, v_frame: np.ndarray, eit_model: EITModel
    ) -> List[np.ndarray]:
        """"""
        return np.hstack(
            (
                make_voltage_vector(
                    eit_model, self.transform_funcs, v_ref, self.get_channel
                ),
                make_voltage_vector(
                    eit_model, self.transform_funcs, v_frame, self.get_channel
                ),
                make_voltage_vector(
                    eit_model, self.transform_funcs, v_frame, self.get_channel
                ),
            )
        )

    def make_labels(self) -> dict[EITPlotsType, CustomLabels]:

        # self.check_data(1, 1)

        t = f"({self.label_meas[1]});"
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

    # def transform_voltages(self, v_ref:np.ndarray, v_frame:np.ndarray, eit_model: EITModel) -> List[np.ndarray]:
    #     """"""
    #     return np.hstack(
    #         (
    #             make_voltage_vector(eit_model, self.transform_funcs, v_ref, self.get_channel),
    #             make_voltage_vector(eit_model, self.transform_funcs, v_frame, self.get_channel),
    #             make_voltage_vector(eit_model, self.transform_funcs, v_frame, self.get_channel) - make_voltage_vector(eit_model, self.transform_funcs, v_ref, self.get_channel)

    #         )
    #     )
    def make_labels(self) -> dict[EITPlotsType, CustomLabels]:

        # self.check_data(2, 1)

        t = f"({self.label_meas[1]}); {self.lab_ref_idx} - {self.lab_frm_idx} ({self.lab_frm_freq})"

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

    # def transform_voltages(self, v_ref:np.ndarray, v_frame:np.ndarray, eit_model: EITModel) -> List[np.ndarray]:
    #     """"""
    #     return np.hstack(
    #         (
    #             make_voltage_vector(eit_model, self.transform_funcs, v_ref, self.get_channel),
    #             make_voltage_vector(eit_model, self.transform_funcs, v_frame, self.get_channel),
    #             make_voltage_vector(eit_model, self.transform_funcs, v_frame, self.get_channel) - make_voltage_vector(eit_model, self.transform_funcs, v_ref, self.get_channel)

    #         )
    #     )
    def make_labels(self) -> dict[EITPlotsType, CustomLabels]:

        # self.check_data(1, 2)

        t = (
            f" ({self.label_meas[1]}); {self.lab_ref_freq} - {self.lab_frm_freq} ({self.lab_frm_idx})",
        )

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
        self.get_channel = True
    def make_labels(self) -> dict[EITPlotsType, CustomLabels]:

        # self.check_data(1, 2)

        t = (f" ({self.label_meas[1]})",)

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


IMAGING_TYPE = {
    "Absolute imaging": AbsoluteImaging,
    "Time difference imaging": TimeDifferenceImaging,
    "Frequence difference imaging": FrequenceDifferenceImaging,
}


def eit_imaging_types()->list[str]:
    return list(IMAGING_TYPE.keys())

def eit_data_transformations()->list[str]:
    return list(DATA_TRANSFORMATIONS.keys())[:4]

if __name__ == "__main__":
    """"""
