import logging
import math
import sys

# Setting the Qt bindings for QtPy
import os
from eit_model.model import EITModel
from eit_model.plot import format_inputs, get_elem_nodal_data
os.environ["QT_API"] = "pyqt5"

from qtpy import QtWidgets

import numpy as np

import pyvista as pv
# import tetgen
from pyvistaqt import QtInteractor, MainWindow


import vtk

logger = logging.getLogger(__name__)

class PyVistaPlotWidget(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)


        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QtWidgets.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)
        
        self.add_electrodes_action = QtWidgets.QAction('Add Electrodes', self)
        self.add_electrodes_action.triggered.connect(self.add_electrodes)
        meshMenu.addAction(self.add_electrodes_action)

        self.plot_eit_action = QtWidgets.QAction('plotEIt', self)
        self.plot_eit_action.triggered.connect(self.plot_eit)
        meshMenu.addAction(self.plot_eit_action)
        
        self.slicing_action = QtWidgets.QAction('Slicing', self)
        self.slicing_action.triggered.connect(self.create_slice)
        meshMenu.addAction(self.slicing_action)
        
        self.review_slice_action = QtWidgets.QAction('Slicing review', self)
        self.review_slice_action.triggered.connect(self.show_slices)
        meshMenu.addAction(self.review_slice_action)

        if show:
            self.show()
        
        self.eit = EITModel()
        self.eit.load_defaultmatfile()
        
        self.plot_eit()


    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.plotter.clear()
        self.plotter.add_mesh(sphere, show_edges=True)
        self.plotter.reset_camera()
        
    def add_electrodes(self):
        elec_pos=self.eit.fem.elec_pos_orient()[:, :3]
        elec_label=[str(x+1) for x in range(elec_pos.shape[0])]
        for i in range (elec_pos.shape[0]):
            elec_mesh = pv.Sphere(0.25, elec_pos[i])
            single_electrode = elec_mesh.slice(normal='z',)
            # electrodes.append(single_electrode)
            self.plotter.add_mesh(single_electrode, color='green', line_width=10, pickable=True,)
            self.plotter.add_point_labels(elec_pos, elec_label,font_size=15)
            self.plotter.reset_camera
        
    
    def plot_eit(self):

        pts, tri = self.eit.fem.nodes, self.eit.fem.elems
        # data= np.random.rand(tri.shape[0], 1)*10
        self.data=self.eit.sim["img_ih"]["elem_data"]
        logger.debug(f"{tri.shape= }")

        # cell must contain padding indicating the number of points in the cell
        padding = np.ones((tri.shape[0],1))*tri.shape[1]
        logger.debug(f"{padding= }{padding.shape= }")
        _cells = np.hstack((padding, tri))
        cells= _cells.astype(np.int64).flatten()
        cell_type = np.array([vtk.VTK_TETRA]*tri.shape[0], np.int8)
        chamber = pv.UnstructuredGrid(cells, cell_type, pts)
        
        # find the elements of object
        idx = np.arange(self.data.shape[0])
        chamber_idx = np.where(self.data == self.data.max())
        cell_indices = np.delete(idx, chamber_idx)
        logger.debug(f'{cell_indices=}')
        object = chamber.extract_cells(cell_indices)
        
        # whole = chamber.merge(object, merge_points=True, main_has_priority=True)
        
        # merge = pv.MultiBlock([chamber, object]).combine(merge_points=True)
        merged = object.merge(chamber)
        logger.debug(f'{merged.n_cells=}')
        
        self.plotter.clear()
        colors = np.real(self.data)   
        
        self.plotter.add_mesh(object, color= 'blue')
        self.plotter.add_mesh(chamber,style='wireframe')
        self.plotter.add_mesh(merged, opacity=0.1, show_scalar_bar=True)
        
        # self.plotter.add_scalar_bar('color', interactive=True, vertical=False)

        self.plotter.add_axes()
        self.plotter.reset_camera()

        return chamber
        
    def create_slice(self):
        
        chamber = self.plot_eit()
        
        
        self.plotter.clear()
        colors = np.real(self.data)     
        # self.plotter.add_mesh(chamber, scalars=colors) 
        self.plotter.add_mesh(chamber, show_edges=True,scalars=colors, show_scalar_bar=False, name='chamber', opacity=0.1)
        # self.plotter.add_mesh(object_slc, show_edges=True, color='blue',name='obj_slice')

        self.plotter.add_mesh_slice(chamber, assign_to_axis='z',)
    
        # self.plotter.add_mesh_slice(object, assign_to_axis='z', widget_color= 'blue')
        self.plotter.add_axes()
        self.plotter.reset_camera()
        
        slice_list = self.plotter.plane_sliced_meshes # List of slices
        
        return slice_list 
        
        
    def show_slices(self):
        
        slice_list = self.create_slice()
        
        for i in slice_list:
            print(i)
            print(i.cell_data)
        slices = pv.MultiBlock(slice_list)
        slices.plot()






if __name__ == '__main__':#
    import glob_utils.log.log
    glob_utils.log.log.main_log()
    app = QtWidgets.QApplication(sys.argv)
    window = PyVistaPlotWidget()
    sys.exit(app.exec_())