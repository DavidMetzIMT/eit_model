### this code is called by EVAL.py

import logging
from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.interp2d
import pyeit.mesh.utils
from matplotlib import axes, figure
from abc import ABC, abstractmethod
from enum import Enum
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from eit_model.data import EITData, EITImage, EITMeasMonitoring
import logging
import numpy as np
import pandas as pd

# from eit_model.model import EITModel
import seaborn as sns
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True


def get_elem_nodal_data(fwd_model, perm):
    """check mesh (tri, pts) in fwd_model and provide elems_data and nodes_data"""

    tri = np.array(fwd_model["elems"])
    pts = np.array(fwd_model["nodes"])

    return check_plot_data(tri, pts, perm)


def check_plot_data(tri, pts, data):
    """check mesh (tri, pts) in fwd_model and provide elems_data and nodes_data"""

    # perm= fwd_model['un2']
    perm = np.reshape(data, (data.shape[0],))

    # tri = tri-1 # matlab count from 1 python from 0
    tri = pyeit.mesh.utils.check_order(pts, tri)

    if perm.shape[0] == pts.shape[0]:
        data = {"elems_data": pyeit.eit.interp2d.pts2sim(tri, perm), "nodes_data": perm}
    elif perm.shape[0] == tri.shape[0]:
        data = {
            "elems_data": perm,
            "nodes_data": pyeit.eit.interp2d.sim2pts(pts, tri, perm),
        }

    for key in data.keys():
        data[key] = np.reshape(data[key], (data[key].shape[0],))
    return tri, pts, data


def format_inputs(fwd_model, data):
    if data.ndim == 2:
        tri = np.array(fwd_model["elems"])
        pts = np.array(fwd_model["nodes"])
        if data.shape[1] in [pts.shape[0], tri.shape[0]]:
            data = data.T
    return data


@dataclass
class CustomLabels(object):
    """Organize the labels utilized by a"""

    title: str = "Default title"
    legend: list[str] = field(
        default_factory=lambda: ["signal_1", "signal_2"]
    )  # ["signal_1", "signal_2"]
    axis: list[str] = field(default_factory=lambda: ["x", "y", "z"])  # ["x", "y", "z"]


class EITPlotsType(Enum):
    Image_2D = "Image_2D"
    Image_3D = "Image_3D"
    U_plot = "U_plot"
    Ch_plot = "Ch_plot"
    U_plot_diff = "U_plot_diff"
    MeasErrorPlot = "MeasErrorPlot"


class EITCustomPlots(ABC):
    """descripe a sort of plot"""

    type: EITPlotsType = None

    def __init__(self) -> None:
        super().__init__()
        self._post_init_()

    @abstractmethod
    def _post_init_(self):
        """Custom initialization"""
        # self.type=PlotType.Image_2D

    @abstractmethod
    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        data: Any,
        labels: CustomLabels = None,
        options: Any = None,
    ) -> Tuple[figure.Figure, axes.Axes]:
        """Plot"""


class EITUPlot(EITCustomPlots):
    """TODO"""

    def _post_init_(self):
        """Custom initialization"""
        self.type = EITPlotsType.U_plot

    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        data: EITData,
        labels: CustomLabels = None,
        options: Any = None,
    ) -> Tuple[figure.Figure, axes.Axes]:
        """Plot"""

        if labels is None or not isinstance(labels, CustomLabels):
            labels = CustomLabels(
                "Voltages", ["ref", "frame"], ["Measurement #", "Voltages in [V]"]
            )

        ax.plot(data.ref_frame, "-b", label=labels.legend[0])
        if labels.legend[1]:
            ax.plot(data.frame, "-r", label=labels.legend[1])

        ax.set_title(labels.title)
        ax.set_xlabel(labels.axis[0])
        ax.set_ylabel(labels.axis[1])
        legend = ax.legend(loc="upper left")

        return fig, ax


class EITUPlotDiff(EITCustomPlots):
    """TODO"""

    def _post_init_(self):
        """Custom initialization"""
        self.type = EITPlotsType.U_plot_diff

    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        data: EITData,
        labels: CustomLabels = None,
        options: Any = None,
    ) -> Tuple[figure.Figure, axes.Axes]:
        """Plot"""

        if labels is None or not isinstance(labels, CustomLabels):
            labels = CustomLabels(
                "Voltages", ["frame-ref", ""], ["Measurement #", "Voltages in [V]"]
            )

        ax.plot(data.frame - data.ref_frame, "-g", label=labels.legend[0])

        ax.set_title(labels.title)
        ax.set_xlabel(labels.axis[0])
        ax.set_ylabel(labels.axis[1])
        if any(l != "" for l in labels.legend):
            legend = ax.legend(loc="upper left")

        return fig, ax


class MeasErrorPlot(EITCustomPlots):
    """This function is used to detect the small values which occurs during the real measurement."""

    def _post_init_(self):
        """Custom initialization"""
        self.type = EITPlotsType.MeasErrorPlot

    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        data: EITMeasMonitoring,
        labels: CustomLabels = None,
        options: Any = None,
    ) -> Tuple[figure.Figure, axes.Axes]:
        """Plot"""

        if labels is None or not isinstance(labels, CustomLabels):
            labels = CustomLabels(
                "Voltages", ["frame-ref", ""], ["Measurement #", "Voltages in [V]"]
            )

        first= list(data.volt_frame.keys())[0]
        ch = data.volt_frame[first].shape[0] #channel numbers, 16

        volt_frame = {k: v.flatten().real for k, v in data.volt_frame.items()}
        
        df = pd.DataFrame.from_dict(
            volt_frame, orient='columns'
        )
        df = df.applymap(filter_value)
        dfm = df.reset_index().melt("index", var_name="frames", value_name="vals")
        dfm["index"] = dfm["index"].apply(lambda x: x % ch + 1)

        df_plot = dfm.loc[dfm["vals"] ==1]
        # fig, ax = plt.subplots()
        ax = sns.histplot(x="frames", y="index", data=df_plot, bins=100, cbar=True, cmap = 'coolwarm')
        # ax.set_xlabel("frame")
        ax.set_ylabel("channel voltage")
        return fig, ax


def filter_value(x):
        # x = np.linalg.norm(x)   # mag
    # return 1 if np.abs(x) < 0.5 else 0    #this line is used for testing
    return 1 if np.abs(x) < 0.00001 else 0


class EITImage2DPlot(EITCustomPlots):
    """TODO"""

    def _post_init_(self):
        """Custom initialization"""
        self.type = EITPlotsType.Image_2D

    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        image: EITImage,
        labels: CustomLabels = None,
        options: Any = None,
        show: list[bool] = [True] * 4,
        colorbar_range: list[int] = None,
    ) -> Tuple[figure.Figure, axes.Axes]:
        """[summary]

        Args:
            fig (figure): [description]
            ax (axes): [description]
            image (ImageEIT): [description]
            show (list[bool], optional): [description]. Defaults to [True*4].
        """
        if labels is None or not isinstance(labels, CustomLabels):
            labels = CustomLabels("Voltages", ["", ""], ["X axis", "Y axis"])

        if colorbar_range is None:
            colorbar_range = [None, None]

        pts, tri, data = image.get_data_for_plot()
        # tri, pts, data= check_plot_data(pts, tri, data)
        logger.debug(f'pts shape = {pts.shape}, tri shape = {tri.shape}')

        key = "elems_data"  # plot only Element data
        perm = np.real(data)
        logger.debug(f'perm shape = {perm.shape}')

        if np.all(perm <= 1) and np.all(perm > 0):
            colorbar_range = [0, 1]
            title = image.label + "\nNorm conduct"
        else:
            title = image.label + "\nConduct"
        im = ax.tripcolor(
            pts[:, 0],
            pts[:, 1],
            tri,
            perm,
            shading="flat",
            vmin=colorbar_range[0],
            vmax=colorbar_range[1],
        )

        fig, ax = add_elec_numbers(fig, ax, image)

        # ax.axis("equal")
        # fig.set_tight_layout(True)
        # ax.margins(x=0.0, y=0.0)
        ax.set_aspect("equal", "box")
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        # ax.axis('off')
        if show[0]:
            ax.set_title(labels.title)
        if show[1]:
            ax.axis("on")
            ax.set_xlabel(labels.axis[0])
        if show[2]:
            ax.set_ylabel(labels.axis[1])
        if show[3]:
            fig.colorbar(im, ax=ax)
        return fig, ax


# def plot_EIT_data(fig:figure.Figure, ax:axes.Axes, data:EITData, labels=None):

#     ax.plot(data.ref_frame, "-b", label='label["legend"][0]')
#     ax.plot(data.frame, "-r", label='label["legend"][1]')

#     ax.set_title("Voltages")
#     ax.set_xlabel("Measurement #")
#     ax.set_ylabel("Voltages in [V]")


#     # ax.set_title(label["title"])
#     # ax.set_xlabel(label["xylabel"][0])
#     # if len(label["xylabel"]) == 2:
#     #     ax.set_ylabel(label["xylabel"][1])
#     # if self.y_axis_log:
#     #     ax.set_yscale("log")
#     # # legend = ax[graph_indx].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
#     # if label["legend"][0] != "":
#     #     legend = ax.legend(loc="upper left")

#     return fig, ax


# def plot_EIT_data_diff(fig:figure.Figure, ax:axes.Axes, data:EITData, labels=None):

#     ax.plot(data.frame - data.ref_frame, "-g", label='label["legend"][0]')

#     ax.set_title("Voltages")
#     ax.set_xlabel("Measurement #")
#     ax.set_ylabel("Voltages in [V]")


#     # ax.set_title(label["title"])
#     # ax.set_xlabel(label["xylabel"][0])
#     # if len(label["xylabel"]) == 2:
#     #     ax.set_ylabel(label["xylabel"][1])
#     # if self.y_axis_log:
#     #     ax.set_yscale("log")
#     # # legend = ax[graph_indx].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
#     # if label["legend"][0] != "":
#     #     legend = ax.legend(loc="upper left")

#     return fig, ax


# def plot_EIT_samples(fwd_model, perm, U):

#     perm=format_inputs(fwd_model, perm)
#     U=format_inputs(fwd_model, U)

#     tri, pts, data= get_elem_nodal_data(fwd_model, perm)

#     key = 'nodes_data' if perm.shape[0]==pts.shape[0] else 'elems_data'
#     fig, ax = plt.subplots(1,2)
#     im = ax[0].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[key]),shading='flat', vmin=None,vmax=None)
#     title = key + (
#         '\nNormalized conductivity distribution'
#         if np.all(perm <= 1)
#         else '\nConductivity distribution'
#     )

#     ax[0].set_title(title)
#     ax[0].set_xlabel("X axis")
#     ax[0].set_ylabel("Y axis")

#     ax[0].axis("equal")
#     fig.colorbar(im,ax=ax[0])

#     ax[1].plot(U.T)

#     plt.show(block=False)

# def plot_real_NN_EIDORS(fwd_model, perm_real,*argv):

#     _perm = [perm_real]
#     for arg in argv:
#         if _perm[0].shape==arg.shape:
#             _perm.append(arg)

#     perm = []
#     if perm_real.ndim > 1:
#         n_row=  perm_real.shape[1]
#         perm.extend(iter(_perm))
#     else:
#         perm.extend(p.reshape((p.shape[0],1)) for p in _perm)
#         n_row= 1
#     n_col = len(perm)

#     fig, ax = plt.subplots(n_row,n_col)
#     if ax.ndim==1:
#         ax=ax.reshape((ax.shape[0],1)).T

#     key= 'elems_data'
#     for row in range(ax.shape[0]):

#         data= [dict() for _ in range(n_col)]
#         for i, p in enumerate(perm):
#             tri, pts, data[i]= get_elem_nodal_data(fwd_model, p[:, row])
#         for col in range(n_col):
#             print(row, col)
#             im = ax[row, col].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[col][key]),shading='flat', vmin=None,vmax=None)
#             title = f'{key}#{row}'

#             # if np.all(perm <= 1):
#             #     title= title +'\nNormalized conductivity distribution'
#             # else:
#             #     title= title +'\nConductivity distribution'
#             ax[row, col].set_title(title)
#             ax[row, col].set_xlabel("X axis")
#             ax[row, col].set_ylabel("Y axis")

#             ax[row, col].axis("equal")
#             fig.colorbar(im,ax=ax[row, col])

#     plt.show(block=False)


# def plot_EIT_image(fig:figure.Figure, ax:axes.Axes, image:EITImage, show:list[bool]=[True] * 4, colorbar_range:list[int]=[0,1])-> None:
#     """[summary]

#     Args:
#         fig (figure): [description]
#         ax (axes): [description]
#         image (ImageEIT): [description]
#         show (list[bool], optional): [description]. Defaults to [True*4].
# #     """

# #     tri, pts, data= get_elem_nodal_data(image.fem, image.data)

# #     key= 'elems_data'
# #     perm=np.real(data[key])
# #     if np.all(perm <= 1) and np.all(perm > 0):
# #         colorbar_range=[0,1]
# #         title= image.label +'\nNorm conduct'
# #     else:
# #         title= image.label +'\nConduct'
# #     im = ax.tripcolor(pts[:,0], pts[:,1], tri, perm, shading='flat', vmin=colorbar_range[0],vmax=colorbar_range[1])
# #     # ax.axis("equal")
# #     # fig.set_tight_layout(True)
# #     # ax.margins(x=0.0, y=0.0)
# #     ax.set_aspect('equal', 'box')
# #     # ax.set_xlim(-1, 1)
# #     # ax.set_ylim(-1, 1)
# #     # ax.axis('off')
# #     if show[0]:
# #         ax.set_title(title)
# #     if show[1]:
# #         ax.axis('on')
# #         ax.set_xlabel("X axis")
# #     if show[2]:
# #         ax.set_ylabel("Y axis")
# #     if show[3]:
# #         fig.colorbar(im,ax=ax)
# #     return fig, ax, im

# def plot_EIT_image(fig:figure.Figure, ax:axes.Axes, image:EITImage, show:list[bool]=[True] * 4, colorbar_range:list[int]=None)-> None:
#     """[summary]

#     Args:
#         fig (figure): [description]
#         ax (axes): [description]
#         image (ImageEIT): [description]
#         show (list[bool], optional): [description]. Defaults to [True*4].
#     """
#     if colorbar_range is None:
#         colorbar_range=[None, None]


#     tri, pts, data= get_elem_nodal_data(image.fem, image.data)

#     key= 'elems_data'
#     perm=np.real(data[key])
#     if np.all(perm <= 1) and np.all(perm > 0):
#         colorbar_range=[0,1]
#         title= image.label +'\nNorm conduct'
#     else:
#         title= image.label +'\nConduct'
#     im = ax.tripcolor(pts[:,0], pts[:,1], tri, perm, shading='flat', vmin=colorbar_range[0],vmax=colorbar_range[1])

#     fig, ax= add_elec_numbers(fig, ax, image)


#     # ax.axis("equal")
#     # fig.set_tight_layout(True)
#     # ax.margins(x=0.0, y=0.0)
#     ax.set_aspect('equal', 'box')
#     # ax.set_xlim(-1, 1)
#     # ax.set_ylim(-1, 1)
#     # ax.axis('off')
#     if show[0]:
#         ax.set_title(title)
#     if show[1]:
#         ax.axis('on')
#         ax.set_xlabel("X axis")
#     if show[2]:
#         ax.set_ylabel("Y axis")
#     if show[3]:
#         fig.colorbar(im,ax=ax)
#     return fig, ax, im


# def plot_2D_EIT_image(fig:figure.Figure, ax:axes.Axes, image:EITImage, show:list[bool]=[True] * 4, colorbar_range:list[int]=None)-> Tuple[figure.Figure,axes.Axes] :
#     """[summary]

#     Args:
#         fig (figure): [description]
#         ax (axes): [description]
#         image (ImageEIT): [description]
#         show (list[bool], optional): [description]. Defaults to [True*4].
#     """
#     if colorbar_range is None:
#         colorbar_range=[None, None]

#     pts, tri, data = image.get_data_for_plot()
#     # tri, pts, data= check_plot_data(pts, tri, data)

#     key= 'elems_data' # plot only Element data
#     perm=np.real(data)

#     if np.all(perm <= 1) and np.all(perm > 0):
#         colorbar_range=[0,1]
#         title= image.label +'\nNorm conduct'
#     else:
#         title= image.label +'\nConduct'
#     im = ax.tripcolor(pts[:,0], pts[:,1], tri, perm, shading='flat', vmin=colorbar_range[0],vmax=colorbar_range[1])

#     fig, ax= add_elec_numbers(fig, ax, image)


#     # ax.axis("equal")
#     # fig.set_tight_layout(True)
#     # ax.margins(x=0.0, y=0.0)
#     ax.set_aspect('equal', 'box')
#     # ax.set_xlim(-1, 1)
#     # ax.set_ylim(-1, 1)
#     # ax.axis('off')
#     if show[0]:
#         ax.set_title(title)
#     if show[1]:
#         ax.axis('on')
#         ax.set_xlabel("X axis")
#     if show[2]:
#         ax.set_ylabel("Y axis")
#     if show[3]:
#         fig.colorbar(im,ax=ax)
#     return fig, ax

# def set_plot_labels(fig:figure.Figure, ax:axes.Axes, label):


#     return fig , ax


def add_elec_numbers(fig: figure.Figure, ax: axes.Axes, image: EITImage):

    elec_x = image.fem["elec_pos"][:, 0]
    elec_y = image.fem["elec_pos"][:, 1]

    ax.plot(elec_x, elec_y, "ok")
    for i, (x, y) in enumerate(zip(elec_x, elec_y)):
        ax.text(x, y, i + 1, color="red", fontsize=12)

    return fig, ax


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files
    import glob_utils.log.log

    glob_utils.log.log.main_log()

    print()
    print([True for _ in range(4)])  #

    p = MeasErrorPlot()
    # v = np.random.randn(256, 9)
    v1 = np.random.randn(16, 16) + np.random.randn(16, 16) * 1j
    v2 = np.random.randn(16, 16) + np.random.randn(16, 16) * 1j
    v3 = np.random.randn(16, 16) + np.random.randn(16, 16) * 1j
    
    # d = EITMeasMonitoring(volt_frame={1:v1})
    d = EITMeasMonitoring(volt_frame={1:v1, 2:v2, 3:v3})

    fig, ax = plt.subplots(1, 1)
    p.plot(fig, ax, d)
    plt.show()
