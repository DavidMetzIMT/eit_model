



import sys
from matplotlib import pyplot as plt
from eit_model.model import EITModel
from eit_model.plot import EITImage2DPlot, EITUPlot
from eit_model.pyvista_plot import PyVistaPlotWidget
from eit_model.solver_pyeit import PyEitRecParams, SolverPyEIT
from PyQt5 import QtCore, QtWidgets

def main():
    """
    _summary_
    """
    p= PyEitRecParams(
        method= ["kotre", "lm"],
        mesh_generation_mode_2D=False,
        
    )
    

    eit_mdl = EITModel()
    eit_mdl.load_defaultmatfile()
    eit_mdl.set_refinement(0.6)

    solver = SolverPyEIT(eit_mdl)
    img_rec, data_sim = solver.prepare_rec(p)
    # solver._build_mesh_from_pyeit(p, True)

    # fig, ax = plt.subplots(1,1)
    # u_graph= EITUPlot()
    # u_graph.plot(fig,ax,data_sim)
    # plt.show()

    app = QtWidgets.QApplication(sys.argv)
    w= PyVistaPlotWidget(eit_mdl, show=True)
    w.plot_eit_image(img_rec)
    sys.exit(app.exec_())  


if __name__ == "__main__":
    import glob_utils.log.log
    glob_utils.log.log.main_log()
    main()