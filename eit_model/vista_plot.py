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

        self.plot_eit_action = QtWidgets.QAction('plotEIt', self)
        self.plot_eit_action.triggered.connect(self.plot_eit)
        meshMenu.addAction(self.plot_eit_action)

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
    
    def plot_eit(self):

        pts, tri = self.eit.fem.nodes, self.eit.fem.elems
        data= np.random.rand(tri.shape[0], 1)*10
        self.data=self.eit.sim["img_ih"]["elem_data"]

        # cell must contain padding indicating the number of points in the cell
        padding = np.ones((tri.shape[0],1))*tri.shape[1]
        logger.debug(f"{padding= }{padding.shape= }")
        _cells = np.hstack((padding, tri))
        cells= _cells.astype(np.int64).flatten()
        cell_type = np.array([vtk.VTK_TETRA]*tri.shape[0], np.int8)
        mesh = pv.UnstructuredGrid(cells, cell_type, pts)

        colors = np.real(self.data)
        # colors[colors == 1] = math.nan
        self.plotter.clear()
        self.plotter.add_mesh(mesh, show_edges=True, scalars=colors)
        self.plotter.reset_camera()
        # self.plotter.background_color='white'






if __name__ == '__main__':#
    import glob_utils.log.log
    glob_utils.log.log.main_log()
    app = QtWidgets.QApplication(sys.argv)
    window = PyVistaPlotWidget()
    sys.exit(app.exec_())