from dataclasses import dataclass
import imp
import logging
from typing import Any

import glob_utils.flags.flag
import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.greit as greit
import pyeit.eit.jac as jac
import pyeit.mesh
import pyeit.mesh.shape
from eit_model.data import EITData, EITImage

from eit_model.model import EITModel
from eit_model.solver_abc import Solver, RecParams
from pyeit.eit.base import EitBase
from pyeit.eit.fem import Forward

logger = logging.getLogger(__name__)

INV_SOLVER_PYEIT = {"JAC": jac.JAC, "BP": bp.BP, "GREIT": greit.GREIT}

def used_solver()->list[str]:
    return list(INV_SOLVER_PYEIT.keys())


@dataclass
class PyEitRecParams(RecParams):
    solver_type: str = next(iter(INV_SOLVER_PYEIT))
    p: float = 0.5
    lamb: float = 0.001
    n: int = 64
    normalize: bool = False
    method: str = "kotre"
    weight: str = "none"
    parser: str = "meas_current"
    background:float = 1.0


class InvSolverNotReadyError(BaseException):
    """"""


class FwdSolverNotReadyError(BaseException):
    """"""


class SolverPyEIT(Solver):

    fwd_solver: Forward
    inv_solver: EitBase
    params: PyEitRecParams

    ############################################################################
    # Abstract Methods
    ############################################################################

    # @abc.abstractmethod
    def __post_init__(self) -> None:
        """Custom post initialization"""
        self.fwd_solver: Forward = None
        self.inv_solver: EitBase = None
        self.params = PyEitRecParams()

    # @abc.abstractmethod
    def _custom_preparation(self, params: Any = None) -> tuple[EITImage, EITData]:
        """Custom preparation of the solver to be ready for reconstruction

        Returns:
            tuple[EITImage, EITData]: a reconstructed EIT image and the
            corresponding EIT data used for it (random generated, simulated
            or loaded...)
            params[Any]: Reconstruction parameters
        """
        logger.info("Preparation of PYEIT solver: Start...")
        self._build_mesh_from_pyeit(import_design=True)
        # solver.init_fwd() # already set in inv less time of computation...
        self.init_inv(params=params, import_fwd=True)
        sim_data, img_h, img_ih = self.simulate()
        img_rec = self.solve_inv(sim_data)
        logger.info("Preparation of PYEIT solver: Done")
        return img_rec, sim_data

    # @abc.abstractmethod
    def _custom_rec(self, data: EITData) -> EITImage:
        """Custom reconstruction of an EIT image using EIT data/measurements

        Args:
            data (EITData): EIT data/measurements for the reconstruction

        Returns:
            EITImage: a reconstructed EIT image corresponding to the EIT
            data/measurements
        """
        return self.solve_inv(data)

    ############################################################################
    # Custom Methods
    ############################################################################

    def init_fwd(self) -> None:  # not really used....
        """Initialize the foward solver from `PyEIT`

        this foward solver use the mesh contained in eit_model
        !!! be sure that the mesh is compatible with PyEIT solver
        otherwise `ValueError` will be raised!!!

        """
        mesh, indx_elec = self._get_mesh_idx_elec_for()
        self.fwd_solver = Forward(mesh, indx_elec)

    def _get_mesh_idx_elec_for(self):
        """Get and check compatility of mesh fro Pyeit

        Raises:
            ValueError: verify it mesh contained in eit_model is compatible

        """

        # get mesh from eit_model
        mesh = self.eit_model.pyeit_mesh()
        # get index of mesh nodes which correspond to the electrodes position
        indx_elec = np.arange(self.eit_model.n_elec)

        # verify if the mesh contains the electrodes positions only 2D compatible
        e_pos_mesh = mesh["node"][indx_elec][:, :2]
        e_pos_model = self.eit_model.elec_pos()[:, :2]
        mesh_compatible = np.all(np.isclose(e_pos_mesh, e_pos_model))

        if not mesh_compatible:
            msg = f"Tried to set solver from PyEIT with incompatible mesh {e_pos_mesh=} {e_pos_model=}"
            logger.error(msg)
            raise ValueError(msg)

        return mesh, indx_elec

    def solve_fwd(self, image: EITImage) -> EITData:
        """Solve the forward problem

        Simulate measurements based on the EIT Image (conductivity) and
        the excitation pattern from eit_model.

        Args:
            image (EITImage, optional): EIT Image contains the conductivity or
            permitivity of each mesh elements.

        Raises:
            FwdSolverNotReadyError: if self.fwd_solver has not been
            initializated
            TypeError: if image is not an `EITImage`

        Returns:
            EITData: simulated EIT data/measurements
        """
        if not isinstance(self.fwd_solver, Forward):
            raise FwdSolverNotReadyError("set first fwd_solver")

        if not isinstance(image, EITImage):
            raise TypeError("EITImage expected")

        ex_mat = self.eit_model.get_pyeit_ex_mat()

        f = self.fwd_solver.solve_eit(
            ex_mat, step=1, perm=image.data, parser=self.params.parser
        )

        return self.eit_model.build_meas_data(f.v, f.v, "solved data")

    def simulate(
        self, image: EITImage = None, homogenious_conduct: float = 1.0
    ) -> tuple[EITData, EITImage, EITImage]:
        """Run a simulation of a EIT measurement and a reference measurement

        Args:
            image (EITImage, optional): EIT_image to simulate EIT measurement.
            If `None` a dummy EIT image is generated. Defaults to `None`.
            homogenious_conduct (float, optional): homogenious conductivity of
            the mesh used for simulation of the reference measurement, with it
            an homogenious image img_h will be generated.Defaults to 1.0.

        Returns:
            tuple[EITData, EITImage, EITImage]: EIT data/measurement containing
            the simulated ref and measurement,
            and both EIT images used img_h and img_ih
        """
        img_h = self.eit_model.build_img(data=homogenious_conduct, label="homogenious")

        img_ih = image
        if img_ih is None:  # create dummy image
            mesh = self.eit_model.pyeit_mesh()
            anomaly = [{"x": 0.5, "y": 0.5, "d": 0.1, "perm": 10}]
            pyeit_mesh_ih = pyeit.mesh.set_perm(mesh, anomaly=anomaly, background=1.0)
            img_ih = self.eit_model.build_img(
                data=pyeit_mesh_ih["perm"], label="inhomogenious"
            )

        data_h = self.solve_fwd(img_h)
        data_ih = self.solve_fwd(img_ih)

        sim_data = self.eit_model.build_meas_data(
            data_ih.frame, data_h.frame, "simulated data"
        )

        return sim_data, img_h, img_ih

    def init_inv(self, params: PyEitRecParams = None, import_fwd: bool = False) -> None:
        """Initialize the inverse and forward solver from `PyEIT` using
        the passed reconstruction parameters

        Args:
            rec_params (PyEitRecParams, optional): reconstruction parameters
            used to set the inverse solver. If `None` default or precedents
            values will be used. Defaults to `None`.
            import_fwd[bool]= Set to `True` to import the fwd_model out of the
            computed one present in the inv_solver. Defaults to `False`
        """

        if isinstance(params, PyEitRecParams):
            # test if new parameters have been transmetted if not  and if
            # inv_solver already ready quit
            isnew_params = self.params.__dict__ != params.__dict__
            if self.ready.is_set() and not isnew_params:
                return
            # otherwise set new parameters
            self.params = params

        self.ready.clear()  # deactivate the solver

        eit_solver_cls = INV_SOLVER_PYEIT[self.params.solver_type]

        mesh, indx_elec = self._get_mesh_idx_elec_for()
        par_tmp = {
            "mesh": mesh,
            "el_pos": indx_elec,
            "ex_mat": self.eit_model.get_pyeit_ex_mat(),
            "step": 1,
            "perm": 1.0,
            "jac_normalized": False,
            "parser": self.params.parser,
        }

        self.inv_solver: EitBase = eit_solver_cls(**par_tmp)
        # during the setting of the inverse solver a fwd model is build...
        # so we use it instead of calling _init_forward
        if import_fwd:
            self.fwd_solver = self.inv_solver.fwd

        self.set_params(self.params)
        self.ready.set()  # activate the solver

    def solve_inv(self, data: EITData) -> EITImage:
        """Solve of the inverse problem or Reconstruction of an EIT image
        using EIT data/measurements

        Args:
            data (EITData): EIT data/measurements for the reconstruction

        Returns:
            EITImage: a reconstructed EIT image corresponding to the EIT
            data/measurements
        """

        ds = self.inv_solver.solve(data.frame, data.ref_frame, self.params.normalize)

        if isinstance(self.inv_solver, greit.GREIT):
            _, _, ds = self.inv_solver.mask_value(ds, mask_value=np.NAN)
        
        return self.eit_model.build_img(data=ds, label=f"PyEIT image: {data.label}")

    def set_params(self, params: PyEitRecParams = None) -> None:
        """Set the reconstrustions parameters for each inverse solver type

        Args:
            rec_params (PyEitRecParams, optional): reconstruction parameters
            used to set the inverse solver. If `None` default or precedents
            values will be used. Defaults to `None`.
        """
        if isinstance(params, PyEitRecParams):
            # test if new parameters have been transmetted if not  and if
            # inv_solver already ready quit
            isnew_params = self.params.__dict__ != params.__dict__
            if self.ready.is_set() and not isnew_params:
                return
            # otherwise set new parameters
            self.params = params

        self.ready.clear()  # deactivate the solver

        # setup for each inv_solve type
        if isinstance(self.inv_solver, bp.BP):
            self.inv_solver.setup(self.params.weight)

        elif isinstance(self.inv_solver, jac.JAC):
            self.inv_solver.setup(self.params.p, self.params.lamb, self.params.method)

        elif isinstance(self.inv_solver, greit.GREIT):
            self.inv_solver.setup(self.params.p, self.params.lamb, self.params.n)

        self.ready.set()  # activate the solver

    def _build_mesh_from_pyeit(self, import_design: bool = False) -> None:
        """To use `PyEIT` solvers (fwd and inv) a special mesh is needed
        >> the first nodes correspond to the position of the Point electrodes

        here a default 2D mesh will be generated corresponding of a 2D
        Circle chamber (radius 1.0, centerd in (0,0)) with 16 point electrodes
        arranged in ring on its contours

        It is also possible to import the chamber design from
        eit_model which comprise
         - elecl number
         - the electrode positions,
         - fem refinement
         - chamber limits
         (- chamber form TODO)

        Args:
            import_design (bool, optional): Set to `True` if you want to import
            the eltrode position . Defaults to False.
        """

        # set box and circle fucntion for a 2D cirle design fro pyEIT
        bbox = self.eit_model.bbox[:, :2] if import_design else None
        # TODO cicl rect depending on the chamber type
        def circ(pts):
            r = np.max(bbox) if bbox is not None else 1.0
            return pyeit.mesh.shape.circle(pts=pts, r=r)

        par_tmp = {
            "n_el": self.eit_model.n_elec if import_design else 16,
            "fd": circ,
            "h0": self.eit_model.refinement if import_design else None,
            "bbox": bbox,
            "p_fix": self.eit_model.elec_pos()[:, :2] if import_design else None,
        }

        pyeit_mesh, indx_elec = pyeit.mesh.create(**par_tmp)

        self.eit_model.update_mesh(pyeit_mesh, indx_elec)  # set the mesh in the model


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files
    import glob_utils.log.log

    glob_utils.log.log.main_log()

    file_path = "E:/Software_dev/Matlab_datasets/20220307_093210_Dataset_name/Dataset_name_infos2py.mat"
    var_dict = glob_utils.files.files.load_mat(file_path)

    eit_mdl = EITModel()
    eit_mdl.load_defaultmatfile()

    solver = SolverPyEIT(eit_mdl)
    img_rec, data_sim = solver.prepare_rec()

    # solver._build_mesh_from_pyeit(import_design=True)
    # # solver.init_fwd() # already set in inv less time of computation...
    # solver.init_inv(import_fwd=True)
    # sim_data, img_h, img_ih= solver.simulate()
    # img_rec= solver.solve_inv(sim_data)

    # fig, ax = plt.subplots(1,1)
    # plot_EIT_image(fig, ax, img_rec)
    # plt.show()
