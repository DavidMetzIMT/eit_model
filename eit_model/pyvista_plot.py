import logging
import math
from operator import mod
import sys

# Setting the Qt bindings for QtPy
import os

from matplotlib import pyplot as plt
from eit_model.model import EITModel
from eit_model.plot import format_inputs, get_elem_nodal_data
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets

import numpy as np

import pyvista as pv
from PyQt5 import QtCore, QtWidgets
# import tetgen
from pyvistaqt import QtInteractor, MainWindow

from eit_model.pyvista_gui import Ui_MainWindow


import vtk

logger = logging.getLogger(__name__)

class PyVistaPlotWidget(MainWindow):

    def __init__(self, parent=None, show=True):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # create the frame
        # add the pyvista interactor object
        pv.global_theme.multi_rendering_splitting_position = 0.30
        # vlayout = QtWidgets.QVBoxLayout()

        self.plotter = QtInteractor(self)#, shape= "3|1")

        self.ui.main_layout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)


        self.slicer = [QtInteractor(self) for _ in range(3)]
        for i,s in enumerate(self.slicer):
            self.ui.slice_layout.addWidget(s.interactor,i,0)
            self.signal_close.connect(s.close)

        # simple menu to demo functions
        self.ui.action_exit.triggered.connect(self.close)

        # allow adding a sphere
        self.ui.action_mesh_reset.triggered.connect(self.mesh_reset)
        self.ui.action_mesh_show_electrodes.triggered.connect(self.show_electrodes)
        self.ui.action_mesh_dynamic_slicing_x.triggered.connect(lambda : self.mesh_dynamic_slicing(0))
        self.ui.action_mesh_dynamic_slicing_y.triggered.connect(lambda : self.mesh_dynamic_slicing(1))
        self.ui.action_mesh_dynamic_slicing_z.triggered.connect(lambda : self.mesh_dynamic_slicing(2))

        # self.ui.action_slicing_set_slices.triggered.connect(lambda : self.set_slice(2))
        # self.ui.action_slicing_z_slicing.triggered.connect(self.ortho_slice)
        
        self.ui.action_view_xy_plane.triggered.connect(self.plotter.view_xy)
        self.ui.action_view_xz_plane.triggered.connect(self.plotter.view_xz)
        self.ui.action_view_yz_plane.triggered.connect(self.plotter.view_yz)
        self.ui.action_onoff_parallel_projection.triggered.connect(self.toggle_parallel_projection)

        self.ui.slider_x.valueChanged[int].connect(lambda val: self.update_slice_origin(val, 0))
        self.ui.slider_x.installEventFilter(self)
        self.ui.slider_y.valueChanged[int].connect(lambda val: self.update_slice_origin(val, 1))
        self.ui.slider_y.installEventFilter(self)
        self.ui.slider_z.valueChanged[int].connect(lambda val: self.update_slice_origin(val, 2))
        self.ui.slider_z.installEventFilter(self)

        self.ui.action_slicing_reset_origin.triggered.connect(self.reset_slice_origin)
        
        self.ui.action_new_data.triggered.connect(self.new_data)
        self.ui.menuColormap.clear()

        for i,(k,v) in enumerate(CMAP.items()):
            name= f'action_cmap_{k}'
            setattr(self, name, QtWidgets.QAction(self))
            action :QtWidgets.QAction = getattr(self, name)
            action.setObjectName(name)
            action.setText(k)
            if i==0:
                action.setChecked(True)
            action.triggered.connect(lambda b, cmap=k: self.set_cmap(cmap))
            self.ui.menuColormap.addAction(action)


        if show:
            self.show()

        self.slice_origin=[0,0,0]
        for s in self.slicer:
            s.parallel_projection= not s.parallel_projection 
        self.actors= {}
        self.cmap="viridis"
    
    def set_cmap(self,val):
        self.cmap=CMAP[val]
        self.plot_eit()
        self.plot_ortho_slice()

    
    def reset_slice_origin(self):
        self.ui.slider_x.setValue(50)
        self.ui.slider_y.setValue(50)
        self.ui.slider_z.setValue(50)

    
    def update_slice_origin(self, val, slice):
    
        bounds = np.array(self.chamber.bounds).reshape((3,2))
        min= bounds[slice,0]
        interval= abs(bounds[slice,0]- bounds[slice,1])
        # logger.debug(f'{bounds=}, {val=}, {slice=}')
        self.slice_origin[slice]=val/100*interval + min
        # logger.debug(f'{self.slice_origin=}')
        self.plot_ortho_slice(idx=[slice])


    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:
        
        #disable MouseWheel event on slider
        if isinstance(source, QtWidgets.QSlider) and event.type() ==QtCore.QEvent.Wheel:
            return True

        return super().eventFilter(source, event)




    def mesh_reset(self, name:str):
        for k in self.actors.keys():
            self.plotter.remove_actor(k)
        self.plotter.clear_plane_widgets()
        self.plotter.add_text(text="Mesh", font_size=10, name='text_main')



    def toggle_parallel_projection(self):
        text_2_display = {
            True : "Disable parallel projection",
            False : "Enable parallel projection"}
        # logger.debug(f'{self.plotter.parallel_projection=}')

        self.plotter.parallel_projection= not self.plotter.parallel_projection
        
        
        self.ui.action_onoff_parallel_projection.setText(
            text_2_display[self.plotter.parallel_projection]
        )

    def show_electrodes(self):
        elec_pos=self.eit_model.fem.elec_pos_orient()[:, :3]
        elec_label=[str(x+1) for x in range(elec_pos.shape[0])]
        for i in range (elec_pos.shape[0]):
            elec_mesh = pv.Sphere(0.25, elec_pos[i])
            single_electrode = elec_mesh.slice(normal='z',)
            # electrodes.append(single_electrode)
            self.plotter.add_mesh(single_electrode, color='green', line_width=3, pickable=True,)
            self.plotter.add_point_labels(elec_pos, elec_label,font_size=15)
            self.plotter.reset_camera()

    def new_data(self, data=None):
        

        self.chamber.cell_data['Conductivity'] = (
            data
            if isinstance(data, np.ndarray)
            else np.random.random_sample(self.data.shape) * 3
        )

        # self.plotter.update_scalars(data,)
        range= self.plotter.mesh.get_data_range()
        self.plotter.update_scalar_bar_range(range)
        self.plot_ortho_slice()

    def set_eitmodel(self, eit_model:EITModel):

        self.eit_model=eit_model
        self.ui.eit_model_name.setText(
            f"eit_model name : {self.eit_model.name}; file= {os.path.split(self.eit_model.file_path)[1]}"
        )
        pts, tri = self.eit_model.fem.nodes, self.eit_model.fem.elems
        # data= np.random.rand(tri.shape[0], 1)*10
        self.data=self.eit_model.sim["img_ih"]["elem_data"]
        self.data= np.random.random_sample(self.data.shape)*3
        # logger.debug(f"{tri.shape= }")

        # cell must contain padding indicating the number of points in the cell
        padding = np.ones((tri.shape[0],1))*tri.shape[1]
        # logger.debug(f"{padding= }{padding.shape= }")
        _cells = np.hstack((padding, tri))
        cells= _cells.astype(np.int64).flatten()
        cell_type = np.array([vtk.VTK_TETRA]*tri.shape[0], np.int8)

        self.chamber = pv.UnstructuredGrid(cells, cell_type, pts)
        self.chamber.cell_data['Conductivity']=self.data
        self.plot_eit()
        self.plot_ortho_slice()


    def plot_eit(self): 

        self.plotter.add_mesh(self.chamber, show_edges=True,name='chamber', opacity=0.1, cmap=self.cmap)
        self.plotter.add_text(text="Mesh", font_size=10, name='text_main')

        # logger.debug(f'{self.plotter.__dict__}')
        # self.plotter.add_scalar_bar('color', interactive=True, vertical=False)
        # self.ortho_slice()

        self.plotter.add_axes()
        self.plotter.reset_camera()

        return self.chamber
        
    def mesh_dynamic_slicing(self, idx:int):

        name=f"slice_{idx}"
        self.plotter.add_mesh_slice(self.chamber, assign_to_axis=idx, name=name, cmap= self.cmap)
        self.plotter.add_text(text="Mesh dynamic slicing", font_size=10, name='text_main')
        self.actors[name]=None
        outline_actor = f'{name}outline'
        self.actors[outline_actor]=None
        # self.plotter.add_mesh_slice(self.chamber, assign_to_axis='y')
        # self.plotter.add_mesh_slice(self.chamber, assign_to_axis='x')
    
        # # self.plotter.add_mesh_slice(object, assign_to_axis='z', widget_color= 'blue')
        # self.plotter.add_axes()
        # self.plotter.reset_camera()
        
        # return self.plotter.plane_sliced_meshes # List of slices



    # def set_slice(self, idx):
    #     # 

    #     if self.plotter.plane_sliced_meshes:
    #         slices = pv.MultiBlock(self.plotter.plane_sliced_meshes)
    #         self._activate_slice_subplot(idx)
    #         self.plotter.add_mesh(slices)
         

    # def show_slices(self):
        
    #     # slice_list = self.create_slice()
    #     # for i in slice_list:
    #     #     print(i)
    #     #     print(i.cell_data)
    #     if self.plotter.plane_sliced_meshes:
    #         slices = pv.MultiBlock(self.plotter.plane_sliced_meshes)
    #         self.plotter.subplot(2)
    #         self.plotter.add_mesh(self.plotter.plane_sliced_meshes[0])
    #         # slices.plot()
    
    def plot_ortho_slice(self, origin = None, idx:list[int]=None):

        if origin is None:
            origin = self.slice_origin
        if idx is None:
            idx = [0,1,2]

        text=[
            "Slice X",
            "Slice Y",
            "Slice Z",
        ]
        normal=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

        for i, (s, t, n) in enumerate(zip(self.slicer, text, normal)):
            if i not in idx:
                continue
            slice = self.chamber.slice(normal=n, origin=origin)
            s.add_mesh(self.chamber.outline(), name='outline')
            s.add_mesh(slice, cmap= self.cmap, name='slice')
            s.view_vector(n)
            s.add_axes()

            s.add_text(text=f'{t} = {origin[i]:.3f}', font_size=10, name="text")


CMAP={
    #https://docs.pyvista.org/examples/02-plot/cmap.html
    #https://matplotlib.org/stable/tutorials/colors/colormaps.html

    "viridis":"viridis",
    "viridis5": plt.cm.get_cmap("viridis", 5),
    "jet":"jet",
    'RdYlGn':'RdYlGn',
    'seismic':'seismic',
    'plasma':'plasma',
    'inferno':'inferno',
    'magma':'magma',
    'cividis':'cividis',
    }
       






if __name__ == '__main__':#
    import glob_utils.log.log
    glob_utils.log.log.main_log()


    # from pyvista import examples
    # import pyvista as pv
    # bolt_nut = examples.download_bolt_nut()
    # pl = pv.Plotter()
    # _ = pl.add_volume(bolt_nut, cmap="coolwarm")
    # pl.show()


    
    # pts= np.array([[i ,i, i ] for i in range(12)])
    # logger.debug(f'{pts=}, {pts.shape=}')

    # tri= np.array([i for i in range(12)]).reshape((4,3))

    # logger.debug(f'{tri=}, {tri.shape=}')

    # center= np.mean(pts[tri], axis=1)
    # logger.debug(f'{center=}, {center.shape=}')

    # n_pos=4
    # X= np.array([[i+j for j in range(10)] for i in range(10)])
    # Y= np.array([[i*j for j in range(4) ] for i in range(10)])
    # logger.debug(f'{X=}, {X.shape=}')
    # logger.debug(f'{Y=}, {Y.shape=}')

    # idx= np.array([ 1, 2, 3, 12, 13, 14, 20, 30])

    # m_pos= mod(idx, n_pos)
    # n_samples= idx//n_pos

    # logger.debug(f'{m_pos=}, {m_pos.shape=}')
    # logger.debug(f'{n_samples=}, {n_samples.shape=}')

    # new_y= Y[n_samples, m_pos]
    # logger.debug(f'{new_y=}, {new_y.shape=}')
    # new_x= np.hstack((X[n_samples, :], center[m_pos,:]))
    # logger.debug(f'{new_x=}, {new_x.shape=}')


    eit_model = EITModel()
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, "default", "default_eit_model_new.mat")
    eit_model.load_matfile(path)
    app = QtWidgets.QApplication(sys.argv)
    window = PyVistaPlotWidget()
    window.set_eitmodel(eit_model)
    sys.exit(app.exec_())
