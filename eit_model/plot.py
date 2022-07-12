### this code is called by EVAL.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyeit.eit.interp2d
import pyeit.mesh.utils

# from eit_model.model import EITModel
import seaborn as sns
from matplotlib import axes, figure

from eit_model.data import EITData, EITImage, EITMeasMonitoringData

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.colorbar").disabled = True


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
    Image_elem = "Image_elem"
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
    def set_options(self,**kwargs):
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
        self.set_options()

    def set_options(self,**kwargs):
        """Custom initialization"""

    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        data: EITData,
        labels: CustomLabels = None,
        
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
        self.set_options()

    def set_options(self,**kwargs):
        """Custom initialization"""
    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        data: EITData,
        labels: CustomLabels = None,
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
        self.set_options()

    def set_options(self,**kwargs):
        """Custom initialization"""

    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        data: EITMeasMonitoringData,
        labels: CustomLabels = None,
    ) -> Tuple[figure.Figure, axes.Axes]:
        """Plot"""

        if labels is None or not isinstance(labels, CustomLabels):
            labels = CustomLabels(
                "Voltages", ["frame-ref", ""], ["Measurement #", "Voltages in [V]"]
            )

        first = list(data.volt_frame.keys())[0]
        ch = data.volt_frame[first].shape[0]  # channel numbers, 16

        volt_frame = {k: v.flatten().real for k, v in data.volt_frame.items()}

        df = pd.DataFrame.from_dict(volt_frame, orient="columns")
        df = df.applymap(filter_value)
        dfm = df.reset_index().melt("index", var_name="frames", value_name="vals")
        dfm["index"] = dfm["index"].apply(lambda x: x % ch + 1)

        df_plot = dfm.loc[dfm["vals"] == 1]
        ax = sns.histplot(
            x="frames", y="index", data=df_plot, bins=100, cbar=True, cmap="coolwarm"
        )
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
        self.set_options()

    def set_options(self, show_electrode: bool= True,  show_axis: bool= True, show_colorbar: bool= True,
        colorbar_range: list[int] = None,
        cmap: str = "viridis",**kwargs):
        """Custom initialization"""
        self.show_electrode=show_electrode
        self.show_axis=show_axis
        self.show_colorbar=show_colorbar
        self.colorbar_range=colorbar_range
        self.cmap= cmap

    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        image: EITImage,
        labels: CustomLabels = None,
    ) -> Tuple[figure.Figure, axes.Axes]:
        """[summary]

        Args:
            fig (figure): [description]
            ax (axes): [description]
            image (ImageEIT): [description]
            show (list[bool], optional): [description]. Defaults to [True*4].
            cmap "viridis", "binary" https://matplotlib.org/stable/tutorials/colors/colormaps.html
        """

        if image.is_3D:
            return fig, ax

        if labels is None or not isinstance(labels, CustomLabels):
            labels = CustomLabels("Voltages", ["", ""], ["X axis", "Y axis"])

        if self.colorbar_range is None:
            self.colorbar_range = [None, None]

        pts, tri, data = image.get_data_for_plot()
        perm = np.real(data)
        perm = perm.flatten() # should be maybe made in EIT_image directly

        try:
            im = ax.tripcolor(
                pts[:, 0],
                pts[:, 1],
                tri,
                perm,
                shading="flat",
                vmin=self.colorbar_range[0],
                vmax=self.colorbar_range[1],
                cmap=self.cmap
            )
            if self.show_electrode:
                fig, ax = _add_elec_numbers(fig, ax, image)
            
        except ValueError:

            im = ax.imshow(perm, interpolation="none", cmap=self.cmap)
            ax.invert_yaxis()
            #TODO electrode for greit


        
        ax.set_aspect("equal", "box")

        if self.show_axis:
            ax.set_title(labels.title)
        # if self.show[1]:
            ax.axis("on")
            ax.set_xlabel(labels.axis[0])
        # if self.show[2]:
            ax.set_ylabel(labels.axis[1])
        else:
            ax.axis("off")
        if self.show_colorbar:
            fig.colorbar(
                im, ax=ax, location="right", shrink=0.6
            )  # label= labels.title)
        return fig, ax


def _add_elec_numbers(
    fig: figure.Figure, ax: axes.Axes, image: EITImage
) -> Tuple[figure.Figure, axes.Axes]:

    elec_x = image.elec_pos[:, 0]
    elec_y = image.elec_pos[:, 1]

    ax.plot(elec_x, elec_y, "ok")
    for i, (x, y) in enumerate(zip(elec_x, elec_y)):
        ax.text(x, y, i + 1, color="red", fontsize=12)

    return fig, ax


class EITElemsDataPlot(EITCustomPlots):
    """Plot the elem data from an EITImage"""

    def _post_init_(self):
        """Custom initialization"""
        self.type = EITPlotsType.Image_elem
        self.set_options()

    def set_options(self):
        """Custom initialization"""

    def plot(
        self,
        fig: figure.Figure,
        ax: axes.Axes,
        image: EITImage,
        labels: CustomLabels = None,
    ) -> Tuple[figure.Figure, axes.Axes]:
        """Plot"""

        if labels is None or not isinstance(labels, CustomLabels):
            labels = CustomLabels(
                "Elements data", ["", ""], ["Elems #", "Conductivity"]
            )

        ax.plot(image.data, "-b")

        ax.set_title(labels.title)
        ax.set_xlabel(labels.axis[0])
        ax.set_ylabel(labels.axis[1])
        # legend = ax.legend(loc="upper left")

        return fig, ax


if __name__ == "__main__":

    import glob_utils.log.log
    from matplotlib import pyplot as plt

    glob_utils.log.log.main_log()

    print()
    print([True for _ in range(4)])  #

    p = MeasErrorPlot()
    # v = np.random.randn(256, 9)
    v1 = np.random.randn(16, 16) + np.random.randn(16, 16) * 1j
    v2 = np.random.randn(16, 16) + np.random.randn(16, 16) * 1j
    v3 = np.random.randn(16, 16) + np.random.randn(16, 16) * 1j

    # d = EITMeasMonitoring(volt_frame={1:v1})
    d = EITMeasMonitoringData(volt_frame={1: v1, 2: v2, 3: v3})

    fig, ax = plt.subplots(1, 1)
    p.plot(fig, ax, d)
    plt.show()
