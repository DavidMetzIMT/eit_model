import logging
from typing import Any, Tuple

import numpy as np
from eit_model.data import (
    EITData,
    EITImage,
    EITMeasMonitoringData,
    EITMeasVoltage,
    EITReconstructionData,
)

from eit_model.imaging import ChannelVoltageMonitoring, EITImaging
from eit_model.model import EITModel
from eit_model.plot import CustomLabels, EITPlotsType
from eit_model.solver_abc import Solver


logger = logging.getLogger(__name__)


class EITReconstruction:

    imaging: EITImaging = None
    monitoring: EITImaging = None
    eit_model: EITModel = None
    solver: Solver = None
    _last_eit_data: EITData = None
    _last_eit_image: EITImage = None
    _calibration: np.ndarray = (
        None  # diag matrix shape (n_exc,n_exc) with calibration coef
    )
    _enable_calibration: str = False

    def __init__(self):
        """The reconstruction agent is responsible to compute EIT image .

        for that data of type EITReconstructionData should be added its input buffer
        directly by calling the "add_data2compute"-method or using it
        SignalReciever functionality by passing the data though a signal

        The images ar then directly send to the plottingagent responsible of
        plotting the image , voltages graphs, ...

        """

        self.rec_enable = False
        self._calibration = None
        self._enable_calibration = False
        self.reset_monitoring_data()
        self.set_monitoring()

    def enable_calibration(self, val: bool = True):
        """
        Enable the calibration porcess on next reconstruction

        Args:
            val (bool, optional): _description_. Defaults to True.
        """
        self._enable_calibration = val
        self._calibration = None

    def compute_calibration(self, data: EITReconstructionData) -> None:
        if not self._enable_calibration:
            return

        v = self.imaging.transformer.run(data.ref_frame.volt, False)
        v_exc_max = np.max(v, axis=1)
        v_max = max(v_exc_max)
        coef = np.reciprocal(v_exc_max.astype(float)) * v_max
        self._calibration = np.diag(coef)

        title = "Calibration result"
        msg = f"\
Calibration done\n\
method : coef(exc)= max(v)/max(v(exc,:))\n\r\
v from {data.ref_frame.name},{data.ref_frame.freq}\n\
and transformed {self.imaging.transformer.transform}, {self.imaging.transformer.show_abs}\n\r\
Corrections coeffs: {coef}"
        logger.info(msg)
        # self.to_gui.emit(EvtPopMsgBox(title, msg, 'info'))
        self._enable_calibration = False

    def rec_process(self, data: EITReconstructionData) -> None:
        """Main reconstrcustion process
        - get eit_data for reconstruction
        - reconstruct eit image

        Args:
            data (EITReconstructionData): data for reconstruction
        """
        self._is_processing = True

        self._data = data
        _name = data.meas_frame.name

        data = self._preprocess_calibration(data)

        self._meas_v_ref, self._meas_v_frame = self._preprocess_meas_voltage(data)
        self.monitoring_data.add(self._meas_v_frame.volt, _name)
        logger.info(f"{_name} - Voltages Monitoring preproccessed")

        (
            self._eit_data,
            self._lab_data,
            self._plot_labels,
        ) = self.imaging.compute_imaging_data(self._meas_v_ref, self._meas_v_frame)
        (
            self._ch_data,
            self._lab_ch,
            self._ch_labels,
        ) = self.monitoring.compute_imaging_data(self._meas_v_ref, self._meas_v_frame)

        logger.info(f"{_name} - Voltages preproccessed")

        self._eit_image = self._process_rec_image(self._eit_data)
        logger.info(f"{_name} - Image rec")

        self._is_processing = False

    def imaging_results(
        self,
    ) -> Tuple[EITImage, EITData, dict[EITPlotsType, CustomLabels]]:
        return self._eit_image, self._eit_data, self._plot_labels

    def monitoring_results(
        self,
    ) -> Tuple[EITMeasMonitoringData, EITData, dict[EITPlotsType, CustomLabels]]:
        return self.monitoring_data, self._ch_data, self._ch_labels

    def _preprocess_calibration(
        self, data: EITReconstructionData
    ) -> EITReconstructionData:
        """"""
        self.compute_calibration(data)

        if self._calibration is None:
            return data

        data.ref_frame.volt = np.matmul(self._calibration, data.ref_frame.volt)
        data.meas_frame.volt = np.matmul(self._calibration, data.meas_frame.volt)

        return data

    def _preprocess_meas_voltage(
        self, data: EITReconstructionData
    ) -> Tuple[EITMeasVoltage, EITMeasVoltage]:
        """Prepocee the data for the monitoring of the voltages. During
        this method the voltages values are send for ploting
        """
        volt_ref, meas_ref = self.eit_model.get_meas_voltages(data.ref_frame.volt)
        v_ref = EITMeasVoltage(
            volt_ref, meas_ref, data.ref_frame.name, data.ref_frame.freq
        )
        volt_meas, meas_meas = self.eit_model.get_meas_voltages(data.meas_frame.volt)
        v_meas = EITMeasVoltage(
            volt_meas, meas_meas, data.meas_frame.name, data.meas_frame.freq
        )
        return v_ref, v_meas

    def _process_rec_image(self, eit_data: EITData) -> EITImage:
        """Reconstruct EIT image

        Args:
            eit_data (EITData):
        """
        if not self.rec_enable:
            return
        if not self.solver or not self.solver.ready.is_set():  #
            logger.warning("Solver not set")
            return

        return self.solver.rec(eit_data)

    def enable_rec(self, enable: bool = True):
        """Enable the EIT image reconstruction. if set to `False` only
        preprocessing of data to compute is done. (voltage meas. will be plot)
        """
        self.rec_enable = enable

    def init_eit_model(self, eit_model: EITModel) -> None:
        """Initialize internal eit_model
        """
        self.eit_model: EITModel = eit_model

    def init_solver(self, solver: Solver, params: Any) -> tuple[EITImage, EITData]:
        """Initialize internal solver, optionaly new solver or reconstruction
        parameters can be set before
        """
        if self.eit_model is None:
            return logger.warning('please set first eit_model')
        self.solver: Solver = solver(self.eit_model)
        logger.info(f"Reconstructions solver selected: {self.solver}")
        return self.solver.prepare_rec(params)

    def set_monitoring(self, transform: str = "Real", show_abs: bool = False):
        """Set voltage channel imaging mode for data visualisation"""
        self.monitoring = ChannelVoltageMonitoring(transform, show_abs)

    def reset_monitoring_data(self):
        """Clear the Eit monitoring data for visualization"""
        self.monitoring_data = EITMeasMonitoringData()


if __name__ == "__main__":
    """"""
