


import os
from typing import Any
import numpy as np
from eit_model.data import EITData ,EITImage
import eit_model.setup
import eit_model.fwd_model
import glob_utils.files.matlabfile
import glob_utils.args.check_type

## ======================================================================================================================================================
##  
## ======================================================================================================================================================


class EITModel(object):
    """ Class regrouping all information about the virtual model 
    of the measuremnet chamber used for the reconstruction:
    - chamber
    - mesh
    - 
    """
    name:str= 'EITModel_defaultName'


    def __init__(self):
        # self.Name = 'EITModel_defaultName'
        # # self.InjPattern = [[0,0], [0,0]]
        # self.Amplitude= float(1)
        # # self.meas_pattern=[[0,0], [0,0]]
        # self.n_el=16
        # self.p=0.5
        # self.lamb=0.01
        # self.n=64

        # pattern='ad'
        # path= os.path.join(DEFAULT_DIR,DEFAULT_INJECTIONS[pattern])
        # self.InjPattern=np.loadtxt(path, dtype=int)
        # path= os.path.join(DEFAULT_DIR,DEFAULT_MEASUREMENTS[pattern])
        # self.meas_pattern=np.loadtxt(path)

        # self.SolverType= 'none'
        # self.FEMRefinement=0.1
        # self.translate_inj_pattern_4_chip()

        self.setup= eit_model.setup.EITSetup()
        self.fwd_model=eit_model.fwd_model.FwdModel()
        self.fem= eit_model.fwd_model.FEModel()


    
    def set_solver(self, solver_type):
        self.SolverType= solver_type

    def load_defaultmatfile(self):
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, 'default','default_eit_model.mat')
        self.load_matfile(filename)

    def load_matfile(self, file_path=None):
        if file_path is None:
            return
        var_dict= glob_utils.files.files.load_mat(file_path)
        self.import_matlab_env(var_dict)
    
    def import_matlab_env(self, var_dict):
        
        m= glob_utils.files.matlabfile.MatFileStruct()
        struct= m._extract_matfile(var_dict,True)

        fmdl= struct['fwd_model']
        fmdl['electrode']= eit_model.fwd_model.mk_list_from_struct(fmdl['electrode'], eit_model.fwd_model.Electrode)
        fmdl['stimulation']= eit_model.fwd_model.mk_list_from_struct(fmdl['stimulation'], eit_model.fwd_model.Stimulation)
        self.fwd_model= eit_model.fwd_model.FwdModel(**fmdl)

        setup= struct['setup']
        self.setup=eit_model.setup.EITSetup(**setup)

        self.fem= eit_model.fwd_model.FEModel(
            **self.fwd_model.for_FEModel(), **self.setup.for_FEModel())


    def translate_inj_pattern_4_chip(self, path=None):
        if path:
            self.ChipPins=np.loadtxt(path)
        else:
            # path= os.path.join(DEFAULT_DIR,DEFAULT_ELECTRODES_CHIP_RING)
            # self.ChipPins=np.loadtxt(path)
            """"""
        
        # test if load data are compatible...
        #todo..
        
        o_num=self.ChipPins[:,0] # Channel number
        n_num=self.ChipPins[:,1] # corresonpint chip pads
        new=np.array(self.InjPattern)
        old=np.array(self.InjPattern)
        for n in range(o_num.size):
            new[old==o_num[n]]= n_num[n]
            
        self.InjPattern= new # to list???
    
    @property    
    def refinement(self):
        return self.fem.refinement
    
    def set_refinement(self, value:float):
        glob_utils.args.check_type.isfloat(value,raise_error=True)
        if value >=1:
            raise ValueError('Value of FEM refinement have to be < 1.0')

        self.fem.refinement= value

    @property
    def n_elec(self, all:bool=True):
        return len(self.fem.electrode)

    # def set_n_elec(self,value:int):
        
    #     glob_utils.args.check_type.isint(value,raise_error=True)
    #     if value <=0:
    #         raise ValueError('Value of FEM refinement have to be > 0')
        
    #     self.setup.elec_layout.elecNb=value


    def pyeit_mesh(self, image:EITImage=None)->dict[str, np.ndarray]:
        """Return mesh needed for pyeit package

        mesh ={
            'node':np.ndarray shape(n_nodes, 2) for 2D , shape(n_nodes, 3) for 3D ,
            'element':np.ndarray shape(n_elems, 3) for 2D shape(n_elems, 4) for 3D,
            'perm':np.ndarray shape(n_elems,1),
        }

        Returns:
            dict: mesh dictionary
        """
        if image is not None and isinstance(image, EITImage):
            return {
                'node':image.fem['nodes'],
                'element':image.fem['elems'],
                'perm':image.data,
            }

        return self.fem.get_pyeit_mesh()

    def elec_pos(self)->np.ndarray:
        """Return the electrode positions 

            pos[i,:]= [posx, posy, posz]

        Returns:
            np.ndarray: array like of shape (n_elec, 3)
        """
        return self.fem.elec_pos_orient()[:,:3]

    def excitation_mat(self)->np.ndarray:
        """Return the excitaion matrix

           ex_mat[i,:]=[elec#IN, elec#OUT]

        Returns:
            np.ndarray: array like of shape (n_elec, 2)
        """
        return self.fwd_model.ex_mat()
    @property    
    def bbox(self)->np.ndarray:
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

    def set_bbox(self, val:np.ndarray)->None:

        self.setup.chamber.set_box_size(val)


    def meas_pattern(self, exc_idx)->np.ndarray:
        """Return the meas_pattern

            used to build the measurement vector
            measU = meas_pattern.dot(meas_ch)

        Returns:
            np.ndarray: array like of shape (n_measU, n_meas_ch*exitation)
        """
        return self.fwd_model.stimulation[exc_idx].meas_pattern

    def build_img(self, data:np.ndarray=None, label:str='image')-> EITImage:
        
        return EITImage(data, label, self.fem)
    
    def update_mesh(self, mesh_data:Any, indx_elec:np.ndarray)->None:
        """Update FEM Mesh

        Args:
            mesh_data (Any): can be a mesh dict from Pyeit 
        """

        if isinstance(mesh_data, dict):
            self.fem.update_from_pyeit(mesh_data, indx_elec)
            # update chamber setups to fit the new mesh...
            m= np.max(self.fem.nodes, axis=0)
            n= np.min(self.fem.nodes, axis=0)
            self.set_bbox(np.round(m-n,1))


    # def update_elec_from_pyeit(self,indx_elec:np.ndarray)->None:
    #     """Update the electrode object contained in the fem

    #     Args:
    #         indx_elec (np.ndarray): The nodes index in the pyeit mesh 
    #         corresponding to the electrodes
    #     """
    #     self.fem.update_elec_from_pyeit(indx_elec)
        

    def build_meas_data(self, ref:np.ndarray, frame:np.ndarray, label:str= '')->EITData:
        """"""
        #TODO  mk som test on the shape of the inputs
        meas= np.hstack((np.reshape(ref,(-1,1)), np.reshape(frame,(-1,1))))
        return EITData(meas, label)

if __name__ == '__main__':

    import glob_utils.files.matlabfile
    import glob_utils.files.files

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files
    import glob_utils.log.log
    glob_utils.log.log.main_log()

    eit= EITModel()
    eit.load_defaultmatfile()

    m= np.max(eit.fem.nodes, axis=0)
    n= np.min(eit.fem.nodes, axis=0)
    print(m, n, np.round(m-n,1))
    print(eit.fwd_model.electrode[1])
    print(eit.fwd_model.electrode[1])
    print(eit.refinement)

