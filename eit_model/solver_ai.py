from posixpath import dirname
from matplotlib.pyplot import plot
import numpy as np
from eit_model.model import EITModel
from eit_model.solver_abc import Solver,RecParams
from typing import Any
from eit_model.data import EITData, EITImage, build_EITImage
from eit_ai.train_utils.workspace import AiWorkspace
from eit_ai.train_utils.metadata import MetaData, reload_metadata
from eit_ai.raw_data.matlab import MatlabSamples
from eit_ai.raw_data.raw_samples import reload_samples
from eit_ai.train_utils.select_workspace import select_workspace
from dataclasses import dataclass

from logging import getLogger

logger = getLogger(__name__)


@dataclass
class AiRecParams(RecParams):
    model_dirpath: str = ''
    normalize: bool = False


class SolverAi(Solver):
    def __post_init__(self) -> None:

        self.metadata: MetaData = None
        self.workspace: AiWorkspace = None
        self.fwd_model: dict = None
        self.params = AiRecParams()

    def _custom_preparation(self, params: Any = None) -> tuple[EITImage, EITData]:
        """Custom preparation of the solver to be ready for reconstruction

        Returns:
            tuple[EITImage, EITData]: a reconstructed EIT image and the
            corresponding EIT data used for it (random generated, simulated
            or loaded...)
            params[Any]: Reconstruction parameters
        """
        logger.info("Preparation of Ai reconstruction: Start...")

        sim_data = self.initialize(params=self.params)
        img_rec = self.rec(sim_data)
        logger.info("Preparation of Ai reconstruction: Done")
        return img_rec, sim_data

    def _custom_rec(self, data: EITData) -> EITImage:
        """Custom reconstruction of an EIT image using EIT data/measurements

        Args:
            data (EITData): EIT data/measurements for the reconstruction

        Returns:
            EITImage: a reconstructed EIT image corresponding to the EIT
            data/measurements
        """
        return self._solve_rec(data)

    def initialize(self, params: AiRecParams = None) -> EITData:
        """initialize the reconstruction method

        Args:
            model_dirpath (str, optional): Ai model path. Defaults to ''.

        Returns:
            EITData: _description_
        """
        self.ready.clear()

        self.metadata = reload_metadata(dir_path=params.model_dirpath)
        raw_samples = reload_samples(MatlabSamples(), self.metadata)
        self.workspace = select_workspace(self.metadata)
        self.workspace.load_model(self.metadata)
        self.workspace.build_dataset(raw_samples, self.metadata)
        self.fwd_model = self.workspace.getattr_dataset("fwd_model")
        voltages, _ = self.workspace.extract_samples(
            dataset_part="test", idx_samples="all"
        )
        logger.debug(f"{voltages.shape}")
        perm_real = self.workspace.get_prediction(
            metadata=self.metadata, single_X=voltages[2], preprocess=False
        )
        logger.debug(f"{perm_real= }, {perm_real.shape}")

        # perm = format_inputs(self.fwd_model, perm_real)

        # logger.debug(f"perm shape = {perm.shape}")
        init_data = EITData(voltages[2], voltages[2], voltages[2]-voltages[2], "solved data" )
        self.ready.set()

        return init_data

    def _solve_rec(self, data: EITData) -> EITImage:
        """Reconstruction of an EIT image
        using EIT data/measurements

        Args:
            data (EITData): EIT data/measurements for the reconstruction

        Returns:
            EITImage: a reconstructed EIT image corresponding to the EIT
            data/measurements
        """

        X = self.preprocess(data)

        logger.debug(f"{X=}\n, {data =}")
        perm_real = self.workspace.get_prediction(
            metadata=self.metadata, single_X=X, preprocess=True
        )

        return build_EITImage(data=perm_real, label="rec image", model=self.eit_mdl)

    def preprocess(self, data: EITData) -> np.ndarray:

        return data.ds / data.ref_frame if self.params.normalize else data.ds
    
    def set_params(self, params: AiRecParams = None) -> None:
        """Set the reconstrustions parameters for each inverse solver type

        Args:
            rec_params (PyEitRecParams, optional): reconstruction parameters
            used to set the inverse solver. If `None` default or precedents
            values will be used. Defaults to `None`.
        """
        if isinstance(params, AiRecParams):
            # test if new parameters have been transmitted if not  and if
            # inv_solver already ready quit
            isnew_params = self.params.__dict__ != params.__dict__
            if self.ready.is_set() and not isnew_params:
                return
            # otherwise set new parameters
            self.params = params

        self.ready.set()


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from eit_model.plot import EITImage2DPlot
    import glob_utils.log.log

    glob_utils.log.log.main_log()

    file_path = r'C:\Users\ryanzzzjw\Desktop\eit_model\eit_model\default\\2D_Dataset_infos2py.mat'
    eit_mdl = EITModel()
    eit_mdl.load_matfile(file_path=file_path)

    solver = SolverAi(eit_mdl)
    img_rec, data_sim = solver.prepare_rec()

    ref = np.random.randn(256, 1)
    frame = np.random.randn(256, 1)
    ds = ref - frame
    data_array = np.hstack([ref, frame, ds])
    v = EITData(data_array)
    # print(v)
    # v = eit_mdl.build_meas_data(ref, frame)
    # img = eit_mdl.build_img(v, 'rec')
    # print(v)
    rec = solver.rec(v)

    logger.debug(f'rec shape = {rec.data.shape}')
    p = EITImage2DPlot()
    fig, ax = plt.subplots(1, 1)
    p.plot(fig, ax, rec)
    plt.show()
