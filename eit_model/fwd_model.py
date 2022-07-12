from dataclasses import dataclass, field
import logging
from typing import Any
import numpy as np
from scipy.sparse import spmatrix
from scipy.linalg import block_diag
from pyeit.mesh import PyEITMesh

logger = logging.getLogger(__name__)

@dataclass
class Stimulation:
    stimulation: str = "Amperes"
    stim_pattern: spmatrix = None
    meas_pattern: spmatrix = None


@dataclass
class Electrode:
    nodes: np.ndarray = None  # 1D array
    z_contact: float = None
    pos: np.ndarray = None  # 1Darray x,y,z,nx,ny,nz
    shape: float = None  # ????
    obj: str = None  # to which it belongs


def mk_list_from_struct(struct: dict, cls) -> list:
    """_summary_

    the structure mimic a structure array from matlab
    struct= {
        '000':{
            'nodes':1, # 1D array
            'z_contact':2,
            'pos':3, #1Darray x,y,z,nx,ny,nz
        },
        '001':{
            'nodes':1, # 1D array
            'z_contact':2,
            'pos':3, #1Darray x,y,z,nx,ny,nz
        }
    }

    @dataclass
    class cls():
        nodes:np.ndarray=None # 1D array
        z_contact:float=None
        pos:np.ndarray=None #1Darray x,y,z,nx,ny,nz

    Args:
        struct (dict): dictionary which mimics a struct array from matlab
        cls: is a data classe with the same attributes name as the keys

    Returns:
        list: _description_
    """
    l = []
    n = len(list(struct.keys()))
    l = ["" for _ in range(n)]
    for k, val in struct.items():
        l[int(k)] = cls(**val)
    return l


@dataclass
class FwdModel:
    """"""

    type: str = "fwd_model"
    nodes: np.ndarray = None
    elems: np.ndarray = None
    boundary: np.ndarray = None
    boundary_numbers: np.ndarray = None
    gnd_node: np.ndarray = None
    np_fwd_solve: dict = None
    name: str = "ng"
    electrode: list[Electrode] = None
    solve: str = "eidors_default"
    jacobian: str = "eidors_default"
    system_mat: str = "eidors_default"
    mat_idx: np.ndarray = None
    normalize_measurements: int = 0
    misc: dict = None
    get_all_meas: int = 1
    stimulation: list[Stimulation] = None
    meas_select: np.ndarray = None
    initialized: int = 2
    SOLVER: list = field(
        default_factory=lambda: ["eidors_default" "fwd_solve_1st_order" "aa_fwd_solve"]
    )
    JACOBIAN: list = field(
        default_factory=lambda: ["eidors_default" "jacobian_adjoint" "aa_calc_jacobian"]
    )
    SYS_MAT: list = field(
        default_factory=lambda: [
            "eidors_default" "system_mat_1st_order" "aa_calc_system_mat"
        ]
    )
    PERM_SYM: list = field(default_factory=lambda: ["n"])

    def __post_init__(self):
        self._create_meas_pattern()
        self._create_meas_pattern_4_pyeit()

    def _create_meas_pattern(self):
        if self.stimulation is None:
            return
        
        l= [stim.meas_pattern.toarray() for stim in self.stimulation]
        self._meas_pattern= block_diag(*l)
        # logger.debug(f"{self._meas_pattern=}, {self._meas_pattern.shape=}")
    
    def _create_meas_pattern_4_pyeit(self):
        """
        Build the measurement pattern (subtract_row-voltage pairs [N, M])
        for all excitations on boundary electrodes.
        Notes
        -----
        ABMN Model.
        A: current driving electrode,
        B: current sink,
        M, N: boundary electrodes, where v_diff = v_n - v_m.
        """        

        if self.stimulation is None:
            return
        
        pattern = np.zeros((len(self.stimulation), self.stimulation[0].meas_pattern.shape[0], 2))
        for i, stim in enumerate(self.stimulation):
            e_min = np.argmin(stim.meas_pattern,axis=1) # TODO verify the order!
            e_plus = np.argmax(stim.meas_pattern,axis=1)
            t= np.hstack((e_min, e_plus)) 
            pattern[i] = t[np.newaxis,:,:]
        self._meas_pattern_4_pyeit=np.int_(pattern)
        # logger.debug(f"{self._meas_pattern_4_pyeit=}, {self._meas_pattern_4_pyeit.shape=}")

    def for_FEModel(self) -> dict:

        return {
            "nodes": self.nodes,
            "elems": self.elems-1, #python is 0 based indexing
            "boundary": self.boundary, 
            "gnd_node": self.gnd_node,
            "electrode": self.electrode,
        }

    def ex_mat(self) -> np.ndarray:
        """Return the excitaion matrix

           ex_mat[i,:]=[elec#IN, elec#OUT]
           electrode numbering with 1 based indexing

        Returns:
            np.ndarray: array like of shape (n_elec, 2)
        """
        ex_mat = np.zeros((len(self.stimulation), 2))
        for i, stim in enumerate(self.stimulation):
            e_in = np.argmin(stim.stim_pattern)
            e_out = np.argmax(stim.stim_pattern)
            ex_mat[i, :] = [e_in, e_out]
        return np.int_(ex_mat) + 1 
    
    @property
    def meas_pattern(self)->np.ndarray:
        return self._meas_pattern
    @property
    def meas_pattern_4_pyeit(self)->np.ndarray:
        return self._meas_pattern_4_pyeit


@dataclass
class FEModel:
    nodes: np.ndarray= None
    elems: np.ndarray= None
    elems_data: np.ndarray = None
    boundary: np.ndarray = None
    gnd_node: int = 0
    electrode: list[Electrode] = None
    refinement: float = 0.1

    def format_perm(self, perm: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            perm (np.ndarray): _description_

        Returns:
            np.ndarray: perm with shape(n_nodes, 1)
        """
        if np.isscalar(perm):
            perm = np.ones(self.elems.shape[0], dtype=np.float) * perm

        if perm.ndim == 2:
            data_s1 = perm.shape[1]
            nodes_s0 = self.nodes.shape[0]
            elems_s0 = self.elems.shape[0]
            if data_s1 in [nodes_s0, elems_s0]:
                perm = perm.T

        return perm

    def set_mesh(self, nodes:np.ndarray, elems:np.ndarray, perm:np.ndarray)->None:
        self.nodes = nodes
        self.elems = elems
        self.elems_data = self.format_perm(perm)
    
    def get_elems_data(self):
        """Return the elements data
        if they are not defined a one vector will be returned

        Returns:
            _type_: _description_
        """
        return self.elems_data if self.elems_data is not None else np.ones((self.elems.shape[0],1))

    # def build_mesh_from_matlab(self, fwd_model:dict, perm:np.ndarray):
    #     perm=format_inputs(fwd_model, perm)
    #     tri, pts, data= get_elem_nodal_data(fwd_model, perm)
    #     # model.fem.set_mesh(pts, tri, data['elems_data'])
    #     self.set_mesh(pts, tri, data['elems_data'])
    #     # self.nodes= fwd_model['nodes']
    #     # self.elems= fwd_model['elems']
    #     # self.set_perm(perm)

    def get_pyeit_mesh(self)-> PyEITMesh:
        """
        Return mesh needed for pyeit package

        Returns:
            PyEITMesh: mesh object
        """
        return PyEITMesh(
            node= self.nodes,
            element= self.elems,
            perm= self.elems_data,
            el_pos=np.arange(self.n_elec), 
        )
    
    def update_mesh(self, mesh: Any, update_elec:bool= False) -> None:
        if isinstance(mesh, PyEITMesh):
            self.update_from_pyeit(mesh,update_elec)
    
    def update_from_pyeit(self, mesh: PyEITMesh, update_elec:bool= False) -> None:
        self.set_mesh(mesh.node, mesh.element, mesh.perm)
        if update_elec:
            self.update_elec_from_pyeit(mesh.el_pos)

    def get_data_for_plots(self):
        return self.nodes, self.elems, self.elems_data

    def elec_pos_orient(self) -> np.ndarray:
        """Return the electrode positions vector and orientation

           pos[i,:]=[posx, posy, poyz, nx, ny, nz]

        Returns:
            np.ndarray: array like of shape (n_elec, 6)
        """
        pos = np.zeros((len(self.electrode), len(self.electrode[0].pos)))
        for i, e in enumerate(self.electrode):
            pos[i, :] = np.reshape(e.pos, (1, -1))
        return pos

    def update_elec_from_pyeit(self, indx_elec: np.ndarray) -> None:
        """Update the electrode object using the actual nodes (from pyeit)

        Args:
            indx_elec (np.ndarray): The nodes index in the pyeit mesh
            corresponding to the electrodes
        """

        self.electrode = [None for _ in list(np.int_(indx_elec))]
        for i in list(np.int_(indx_elec)):
            nodes = i
            pos = self.nodes[i, :]
            self.electrode[i] = Electrode(nodes=nodes, pos=pos, shape=0.0)

    @property
    def n_elec(self):
        return len(self.electrode)
    
    @property
    def is_3D(self):
        return self.elems.shape[1] == 4

    @property
    def is_2D(self):
        return self.elems.shape[1] == 3


if __name__ == "__main__":

    import glob_utils.log.log

    glob_utils.log.log.main_log()

    import glob_utils.file.mat_utils


    e= list(range(16))
    print(e)
    e_1 = [15-e_i for e_i in e]
    print(e_1)


    # file_path = "E:/Software_dev/Matlab_datasets/20220307_093210_Dataset_name/Dataset_name_infos2py.mat"
    # var_dict = glob_utils.file.mat_utils.load_mat(file_path)
    # m = glob_utils.file.mat_utils.MatFileStruct()
    # struct = m._extract_matfile(var_dict)
    # f = struct["fwd_model"]
    # f["electrode"] = mk_list_from_struct(f["electrode"], Electrode)
    # f["stimulation"] = mk_list_from_struct(f["stimulation"], Stimulation)

    # fmdl = FwdModel(**f)
    # print(fmdl.__dict__)
    # print("STIMMMMMMM", fmdl.stimulation[1])
    # print("STIMMMMMMM", fmdl.electrode[1])

    # print(fmdl.elec_pos_orient())
    # print(fmdl.ex_mat())

    # d = {
    #     "000": {
    #         "nodes": 1,  # 1D array
    #         "z_contact": 2,
    #         "pos": 3,  # 1Darray x,y,z,nx,ny,nz
    #     },
    #     "001": {
    #         "nodes": 1,  # 1D array
    #         "z_contact": 2,
    #         "pos": 3,  # 1Darray x,y,z,nx,ny,nz
    #     },
    # }

    # e = mk_list_from_struct(d, Electrode)
    # print(e)
    # print(e[1].nodes)

    
