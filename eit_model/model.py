import logging
import os
from typing import Any, Tuple
import numpy as np
import eit_model.setup
import eit_model.fwd_model
import glob_utils.file.mat_utils
import glob_utils.file.utils
import glob_utils.args.check_type
from scipy.sparse import csr_matrix
from pyeit.mesh import PyEITMesh

## ======================================================================================================================================================
##
## ======================================================================================================================================================
logger = logging.getLogger(__name__)

class ChipTranslatePins(object):
 
    # chip_trans_mat:np.ndarray # shape (n_elec, 2)
    _elec_to_ch:np.ndarray
    _ch_to_elec:np.ndarray
    _elec_num:np.ndarray # model elec # shape (n_elec, 1)
    _ch_num:np.ndarray # corresponding chip pad/channnel # shape (n_elec, 1)
    _file:str #file name 

    def __init__(self) -> None:
        """
        _summary_
        """
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "default", "Chip_Ring_e16_17-32.txt")
        self.load(path)            
    
    def load(self, path):

        tmp = np.loadtxt(path, dtype=int)
        glob_utils.file.utils.logging_file_loaded(path)
        # TODO verify the fiste colum schould be 1-N
        self._elec_num = tmp[:, 0]
        self._ch_num = tmp[:, 1] 
        logger.debug(f"{self._elec_num=},{self._ch_num=}")
        self.build_trans_matrices()   
        self._file= os.path.split(path)[1]

    def transform_exc(self, exc_pattern:np.ndarray )->np.ndarray:
        """transform the pattern given by the eit model with electrode 
        numbering into a corresponding pattern for the selected chip
        
        basically the electrode #elec_num in the model is connected to 
        the channel #ch_num

        exc_pattern[i,:]=[elec_num#IN, elec_num#OUT]
        >> new_pattern[i,:]=[ch_num#IN, ch_num#OUT]

        """
        new_pattern = np.array(exc_pattern)
        old = np.array(exc_pattern)
        for n in range(self._elec_num.size):
            new_pattern[old == self._elec_num[n]] = self._ch_num[n]
        return new_pattern

    def trans_elec_to_ch(self, volt:np.ndarray)->np.ndarray:
        """_summary_

        Args:
            volt (np.ndarray): volt(:, n_elec)

        Returns:
            np.ndarray: volt(:, n_channel)
        """
        return volt.dot(self._elec_to_ch)

    def trans_ch_to_elec(self, volt:np.ndarray)->np.ndarray:
        """_summary_

        Args:
            volt (np.ndarray): volt(:, n_channel)

        Returns:
            np.ndarray: volt(:, n_elec)
        """
        return volt.dot(self._ch_to_elec)


    def build_trans_matrices(self):
        """Build the transformation matrices

        _elec_to_ch:np.ndarray vol(:, n_elec) -> vol(:, n_channel)
        _ch_to_elec:np.ndarray vol(:, n_channel) -> vol(:, n_elec)
        
        """
        n_elec=self._elec_num.size
        self._elec_to_ch=np.zeros((n_elec,32))
        elec_idx = np.array(self._elec_num.flatten() -1) # 0 based indexing 
        ch_idx = np.array(self._ch_num.flatten() -1) # 0 based indexing
        data = np.ones(n_elec)

        a= csr_matrix((data,(elec_idx, ch_idx)), dtype=int).toarray()
        self._elec_to_ch[:a.shape[0],:a.shape[1]]= a

        self._ch_to_elec= self._elec_to_ch.T
        # logger.debug(f"{self._elec_to_ch=}, {self._ch_to_elec=}")
        logger.debug(f"{self._elec_to_ch.shape=}, {self._ch_to_elec.shape=}")




class EITModel(object):
    """Class regrouping all information about the virtual model
    of the measuremnet chamber used for the reconstruction:
    - chamber
    - mesh
    -
    """

    name: str = "EITModel_defaultName"
    setup:eit_model.setup.EITSetup
    fwd_model:eit_model.fwd_model.FwdModel
    fem:eit_model.fwd_model.FEModel
    sim:dict
    

    def __init__(self):
        self.setup = eit_model.setup.EITSetup()
        self.fwd_model = eit_model.fwd_model.FwdModel()
        self.fem = eit_model.fwd_model.FEModel()
        self.chip= ChipTranslatePins()
        self.load_default_chip_trans()

    def set_solver(self, solver_type):
        self.SolverType = solver_type
    
    def load_chip_trans(self, path:str):
        self.chip.load(path)

    def load_default_chip_trans(self):
        dirname = os.path.dirname(__file__)
        path = os.path.join(dirname, "default", "Chip_Ring_e16_1-16.txt")
        self.chip.load(path)

    def load_defaultmatfile(self):
        dirname = os.path.dirname(__file__)
        file_path = os.path.join(dirname, "default", "default_eit_model.mat")
        self.load_matfile(file_path)

    def load_matfile(self, file_path=None):
        if file_path is None:
            return
        var_dict = glob_utils.file.mat_utils.load_mat(file_path, logging=False)

        self.file_path=file_path
        self.import_matlab_env(var_dict)

    def import_matlab_env(self, var_dict):

        m = glob_utils.file.mat_utils.MatFileStruct()
        struct = m._extract_matfile(var_dict,verbose=False)

        fmdl = struct["fwd_model"]
        fmdl["electrode"] = eit_model.fwd_model.mk_list_from_struct(
            fmdl["electrode"], eit_model.fwd_model.Electrode
        )
        fmdl["stimulation"] = eit_model.fwd_model.mk_list_from_struct(
            fmdl["stimulation"], eit_model.fwd_model.Stimulation
        )
        self.fwd_model = eit_model.fwd_model.FwdModel(**fmdl)

        setup = struct["setup"]
        self.setup = eit_model.setup.EITSetup(**setup)

        self.fem = eit_model.fwd_model.FEModel(
            **self.fwd_model.for_FEModel(), **self.setup.for_FEModel()
        )
        self.sim= struct["sim"]
        # set name of eit_model
        for k in struct.keys():
            if "eit_" in k:
                self.name= struct[k]['name']
                break
    @property
    def refinement(self):
        return self.fem.refinement

    def set_refinement(self, value: float):
        glob_utils.args.check_type.isfloat(value, raise_error=True)
        if value >= 1:
            raise ValueError("Value of FEM refinement have to be < 1.0")

        self.fem.refinement = value

    @property
    def n_elec(self):
        return self.fem.n_elec

    def pyeit_mesh(self) -> PyEITMesh:
        """
        Return mesh needed for pyeit package

        Returns:
            PyEITMesh: mesh object
        """
        return self.fem.get_pyeit_mesh()

    def elec_pos(self) -> np.ndarray:
        """Return the electrode positions

            pos[i,:]= [posx, posy, posz]

        Returns:
            np.ndarray: array like of shape (n_elec, 3)
        """
        return self.fem.elec_pos_orient()[:, :3]

    def excitation_mat(self) -> np.ndarray:
        """Return the excitaion matrix

           ex_mat[i,:]=[elec#IN, elec#OUT]
           electrode numbering with 1 based indexing

        Returns:
            np.ndarray: array like of shape (n_elec, 2)
        """
        return self.fwd_model.ex_mat()

    def excitation_mat_chip(self) -> np.ndarray:
        """Return the excitaion matrix for the chip selected
        
        the pins will be corrected as defined in the chip design txt file

           ex_mat[i,:]=[elec#IN, elec#OUT]
           electrode numbering with 1 based indexing

        Returns:
            np.ndarray: array like of shape (n_elec, 2)
        """
        return self.chip.transform_exc(self.fwd_model.ex_mat())

    def get_pyeit_ex_mat(self)-> np.ndarray:
        """Return the excitaion matrix for pyeit which has to be 
        0 based indexing"""
        return self.excitation_mat()-1

    def get_pyeit_meas_pattern(self)-> np.ndarray:
        """Return the meas_pattern for pyeit which is 
        0 based indexed, and of shape (n_exc, n_meas_per_exc, 2) """
        return self.fwd_model.meas_pattern_4_pyeit

    @property
    def bbox(self) -> np.ndarray:
        """Return the mesh /chamber limits as ndarray

        limits= [
            [xmin, ymin (, zmin)]
            [xmax, ymax (, zmax)]
        ]

        if the height of the chamber is zero a 2D box limit is returned

        Returns:
            np.ndarray: box limit
        """
        # TODO
        # add a chekcing if chmaber and mesh are compatible
        return self.setup.chamber.box_limit()

    def set_bbox(self, val: np.ndarray) -> None:
        self.setup.chamber.set_box_size(val)

    def single_meas_pattern(self, exc_idx:int) -> np.ndarray:
        """Return the meas_pattern

            used to build the measurement vector
            measU = meas_pattern.dot(meas_ch)

        Returns:
            np.ndarray: array like of shape (n_meas, n_elec)
        """
        return self.fwd_model.stimulation[exc_idx].meas_pattern.toarray()
    
    def meas_pattern(self) -> np.ndarray:
        """Return the meas_pattern

            used to build the measurement vector
            measU = meas_pattern.dot(meas_ch)

        Returns:
            np.ndarray: array like of shape (n_meas*n_exc, n_elec*n_exc)
        """
        return self.fwd_model.meas_pattern

    def update_mesh(self, mesh: Any, update_elec:bool= False) -> None:
        """Update FEM Mesh

        Args:
            mesh_data (Any): can be a mesh from Pyeit
        """
        self.fem.update_mesh(mesh, update_elec)
        # update chamber setups to fit the new mesh...
        m = np.max(self.fem.nodes, axis=0)
        n = np.min(self.fem.nodes, axis=0)
        self.set_bbox(np.round(m - n, 1))
    
    def get_meas_voltages(self, volt:np.ndarray)-> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            voltages (np.ndarray): shape(n_exc, n_channel)

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
            - meas_voltage shape(n_exc, n_elec)
            - meas_data of shape(n_meas*n_exc, )
        """
        if volt is None:
            return np.array([])
        # get only the voltages of used electrode (0-n_el)
        meas_voltage = self.chip.trans_ch_to_elec(volt)
        # get the volgate corresponding to the meas_pattern and flatten
        meas_data= self.meas_pattern().dot(meas_voltage.flatten())
        return meas_data, meas_voltage 
    
    def get_protocol_info(self)->list[str]:
        """
        Return a list of string containing informaton saved in 
        the analysis protocol

        Returns:
            list[str]: lines of informationss
        """        
        return [
            f'Name: {self.name}',
            f'FEM Refinement: {self.refinement}',
            f'Chip config: {self.chip._file}',
        ]



if __name__ == "__main__":

    import glob_utils.log.log
    glob_utils.log.log.main_log()
    a = np.array([[[1,2], [3,4], [3,4]]])

    print(a.shape)
    print(a.shape[::2] != (1,2))
    a= a.flatten()
    print(a)


    def test2(ex_mat : np.ndarray,
        meas_pattern: np.ndarray= None):
        print(ex_mat, meas_pattern)


    def test1(ex_mat, **kwargs):
        test2(ex_mat, **kwargs)

    test1('ex_mat')



    dirname = os.path.dirname(__file__)
    # path = os.path.join(dirname, "default", "Chip_Ring_e16_1-16.txt")
    # p=  np.loadtxt(path)


    # path = os.path.join(dirname, "default", "Chip_Ring_e16_17-32.txt")
    # path = os.path.join(dirname, "default", "Chip_Ring_e16_1-16.txt")

    # eit = EITModel()
    # eit.load_defaultmatfile()
    # eit.load_chip_trans(path)
    # # print("pattern", eit.chip.transform_exc(p))soli
    # volt = np.array([list(range(32)) for _ in range(16)])+1
    # a, b =eit.get_meas_voltages(volt)

 





    
    path = os.path.join(dirname, "default", "test_adop_infos2py.mat")
    eit = EITModel()
    eit.load_matfile(path)

    # m = np.max(eit.fem.nodes, axis=0)
    # n = np.min(eit.fem.nodes, axis=0)
    # print(m, n, np.round(m - n, 1))
    # print(eit.fwd_model.electrode[1])
    # print(eit.fwd_model.electrode[1])
    # print(eit.refinement)
