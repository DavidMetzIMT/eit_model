


from dataclasses import dataclass

import glob_utils.flags.flag
import numpy as np
import pyeit.eit.bp as bp
import pyeit.eit.greit as greit
import pyeit.eit.jac as jac
import pyeit.mesh
import pyeit.mesh.shape
from eit_model.data import EITData
from eit_model.image import EITImage
from eit_model.model import EITModel
from eit_model.plot.mesh_2D import plot_EIT_image
from eit_model.solver.solver_abc import SolverAbc
from pyeit.eit.base import EitBase
from pyeit.eit.fem import Forward
from pyeit.eit.interp2d import pts2sim, sim2pts


@dataclass
class RecParams():
    p: float= 0.5
    lamb: float =0.001
    n: int =64
    normalize:bool=False
    
class InvSolverNotReadyError(BaseException):
    """"""

class FwdSolverNotReadyError(BaseException):
    """"""

class SolverPyEIT(SolverAbc):

    def __init__(self) -> None:
        self.fwd_solver:Forward=None
        self.inv_solver:EitBase=None
        self.rec_params:RecParams=RecParams()
        # self.normalize:bool=False
        self.eit_model:EITModel=None
        self.ready:glob_utils.flags.flag.CustomFlag=glob_utils.flags.flag.CustomFlag()

    def rec(self, data:EITData)-> EITImage:
        """"""
        if not self.ready.is_set():
            raise InvSolverNotReadyError('PyEIT Solver not ready')
        # ds = self.inv_solver.solve(data.frame, data.ref_frame, self.normalize )
        # print(data.frame, data.ref_frame)
        ds = self.inv_solver.solve(data.frame, data.ref_frame, self.rec_params.normalize )

        if isinstance(self.inv_solver, greit.GREIT):
            _, _, ds = self.inv_solver.mask_value(ds, mask_value=np.NAN)
        
        return self.eit_model.build_img(data= ds, label='rec image')

    
    
    def init_sim(self):
        """"""

        
        mesh= self.eit_model.pyeit_mesh()
        el_pos= np.arange(self.eit_model.n_elec)

        self.fwd_solver=Forward_all_meas(mesh, el_pos)
    
    
    def sim(self, image:EITImage=None)-> tuple[EITData, EITImage, EITImage]:
        """"""
        if not isinstance(self.fwd_solver, Forward):
            raise FwdSolverNotReadyError('set first fwd_solver')

        img_h=self.eit_model.build_img(data=1, label='homogenious')
        img_ih=image
        if img_ih is None:
            mesh= self.eit_model.pyeit_mesh()
            # print(mesh)
            anomaly = [{"x": 0.5, "y": 0.5, "d": 0.1, "perm": 10}]
            pyeit_mesh_ih = pyeit.mesh.set_perm(mesh, anomaly=anomaly,background=1.0)
            # print(pyeit_mesh_ih)
            img_ih= self.eit_model.build_img(
                data= pyeit_mesh_ih["perm"], label='inhomogenious')
            # print(img_ih.data)

        ex_mat = self.eit_model.excitation_mat()

        step = 1
        f0 = self.fwd_solver.solve_eit(ex_mat, step, img_h.data)
        f1 = self.fwd_solver.solve_eit(ex_mat, step, img_ih.data)
        sim_data= self.eit_model.build_meas_data(f0.v, f1.v, 'simulated data')

        return sim_data, img_h, img_ih

    
    def init_rec(self, solver_type, rec_params:RecParams=None)->None:

        self.ready.clear()
        eit:EitBase= SOLVER_PYEIT[solver_type]

        mesh= self.eit_model.pyeit_mesh()
        el_pos= np.arange(self.eit_model.n_elec)
        ex_mat= self.eit_model.excitation_mat()
        step= 1
        perm= 1.0
        jac_normalized= False
        parser= 'std'
        self.inv_solver= eit(mesh, el_pos, ex_mat, step, perm, jac_normalized, parser)
        self.set_rec_params(rec_params)
        self.ready.set()

    def set_rec_params(self, rec_params:RecParams=None)->None:
        self.ready.clear()
        if isinstance(rec_params, RecParams):
            self.rec_params=rec_params
        

        if isinstance(self.inv_solver, bp.BP):
            self.inv_solver.setup(
                weight="none"
            )         
        elif isinstance(self.inv_solver, jac.JAC):
            self.inv_solver.setup(
                self.rec_params.p, self.rec_params.lamb, method="kotre"
            )
        elif isinstance(self.inv_solver, greit.GREIT):
            self.inv_solver.setup(
                self.rec_params.p, self.rec_params.lamb,self.rec_params.n
            )
        self.ready.set()

    def build_mesh_from_pyeit(self, import_electrode:bool=False):

        n_el=self.eit_model.n_elec if import_electrode else 16
        h0=self.eit_model.refinement
        bbox= np.array(self.eit_model.setup.chamber.get_chamber_limit())
        p_fix=self.eit_model.elec_pos()[:,:2] if import_electrode else None

        def circ(pts, pc=None):
            return pyeit.mesh.shape.circle(pts, pc,r=np.max(bbox))

        pyeit_mesh, indx_elec = pyeit.mesh.create(
            n_el = n_el,
            fd=circ,
            h0 = h0,
            bbox = bbox,
            p_fix = p_fix)
        
        print(pyeit_mesh['node'][indx_elec])

        self.eit_model.update_mesh(pyeit_mesh) # set the mesh in the model

        if not import_electrode:
            # set
            self.eit_model.update_elec_from_pyeit(indx_elec)





SOLVER_PYEIT={
    'JAC':jac.JAC,
    'BP':bp.BP,
    'GREIT':greit.GREIT
}

class Forward_all_meas(Forward):
    def __init__(self, mesh, el_pos):
        super().__init__(mesh, el_pos)

    def voltage_meter(ex_line, n_el=16, step=1, parser=None):
        """
        extract subtract_row-voltage measurements on boundary electrodes.
        we direct operate on measurements or Jacobian on electrodes,
        so, we can use LOCAL index in this module, do not require el_pos.

        Notes
        -----
        ABMN Model.
        A: current driving electrode,
        B: current sink,
        M, N: boundary electrodes, where v_diff = v_n - v_m.

        'no_meas_current': (EIDORS3D)
        mesurements on current carrying electrodes are discarded.

        Parameters
        ----------
        ex_line: NDArray
            2x1 array, [positive electrode, negative electrode].
        n_el: int
            number of total electrodes.
        step: int
            measurement method (two adjacent electrodes are used for measuring).
        parser: str
            if parser is 'fmmu', or 'rotate_meas' then data are trimmed,
            boundary voltage measurements are re-indexed and rotated,
            start from the positive stimulus electrodestart index 'A'.
            if parser is 'std', or 'no_rotate_meas' then data are trimmed,
            the start index (i) of boundary voltage measurements is always 0.

        Returns
        -------
        v: NDArray
            (N-1)*2 arrays of subtract_row pairs
        """
        # local node
        drv_a = ex_line[0]
        drv_b = ex_line[1]
        i0 = drv_a if parser in ("fmmu", "rotate_meas") else 0

        # build differential pairs
        v = []
        for a in range(i0, i0 + n_el):
            m = a % n_el
            n = (m + step) % n_el
            # if any of the electrodes is the stimulation electrodes
            # if not (m == drv_a or m == drv_b or n == drv_a or n == drv_b) or True:
                # the order of m, n matters
            v.append([n, m])

        return np.array(v)


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files

    file_path='E:/Software_dev/Matlab_datasets/20220307_093210_Dataset_name/Dataset_name_infos2py.mat'
    var_dict= glob_utils.files.files.load_mat(file_path)

    eit_mdl= EITModel()
    eit_mdl.import_matlab_env(var_dict)
    eit_mdl.set_refinement(0.2)

    solver= SolverPyEIT()
    solver.eit_model=eit_mdl
    solver.build_mesh_from_pyeit(import_electrode=True)
    solver.init_sim()
    img= eit_mdl.build_img(data=1, label='homogenious')
    fig, ax = plt.subplots(1,1)
    # plot_EIT_image(fig, ax, img)

    sim_data, img_h, img_ih= solver.sim()
    solver.init_rec(next(iter(SOLVER_PYEIT)))

    img_rec= solver.rec(sim_data)

    plot_EIT_image(fig, ax, img_rec)
    plt.show()

