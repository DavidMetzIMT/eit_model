



from matplotlib import pyplot as plt
from eit_model.model import EITModel
from eit_model.plot import EITImage2DPlot, EITUPlot
from eit_model.solver_pyeit import PyEitRecParams, SolverPyEIT


def main():
    """
    _summary_
    """
    p= PyEitRecParams(
        method= ["kotre", "lm"],
        mesh_generation_mode_2D=True,
    )
    

    eit_mdl = EITModel()
    eit_mdl.load_defaultmatfile()

    solver = SolverPyEIT(eit_mdl)
    img_rec, data_sim = solver.prepare_rec(p)

    fig, ax = plt.subplots(1,1)
    u_graph= EITUPlot()
    u_graph.plot(fig,ax,data_sim)
    plt.show()

    fig, ax = plt.subplots(1,1)
    img_graph= EITImage2DPlot()
    img_graph.plot(fig,ax,img_rec)
    plt.show()


if __name__ == "__main__":
    main()