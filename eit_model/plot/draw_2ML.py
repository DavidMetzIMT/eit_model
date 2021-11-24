### this code is called by EVAL.py

import random
from enum import Enum, auto
from logging import getLogger
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.interp2d
import pyeit.mesh.utils
from matplotlib import axes, figure
from scipy.io import loadmat


from eit_model.plot.mesh_2D import ImageDataset, plot_EIT_mesh

logger = getLogger(__name__)


class Orientation(Enum):
    Portrait=auto()
    Landscape=auto()

def plot_compare_samples(
        image_data:list[ImageDataset],
        nb_samples:int=0,
        rand:bool=False,
        orient:Orientation=Orientation.Portrait)-> None:

    if not len(image_data):
        logger.warning(f'no ImageData {image_data}')
        return
    
    idx_list= generate_nb_samples2plot(image_data, nb_samples, rand)
    img2plot= [ImageDataset(id.data[idx_list,:], id.label, id.fwd_model) for id in image_data]

    n_img= len(img2plot)
    n_samples= len(idx_list)

    n_row, n_col= orient_swap(orient, n_samples, n_img)

    fig, ax = plt.subplots(n_row,n_col)

    for row in range(n_row):
        for col in range(n_col):
            idx_sample, idx_image= orient_swap(orient, row, col)
            image=img2plot[idx_image].get_single(idx_sample)
            show= [False] * 4
            if idx_sample==0:
                show[0]= True #title
            if col==0 and row==n_row-1:
                show[1]= True #x axis
                show[2]= True #y axis
            fig, ax[row, col], im= plot_EIT_mesh(fig, ax[row, col], image, show)   
            if idx_sample== n_samples-1:
                if orient==Orientation.Landscape:
                    fig.colorbar(im, ax=ax[idx_image, :], location='right', shrink=0.6)
                elif orient==Orientation.Portrait:
                    fig.colorbar(im, ax=ax[:,idx_image], location='bottom', shrink=0.6)
    # fig.set_tight_layout(True)        
    plt.show(block=False)

def orient_swap(orient:Orientation, a:Any, b:Any)-> tuple[Any, Any]:#
    if orient==Orientation.Landscape:
        return b, a
    elif orient==Orientation.Portrait:
        return a, b
    else:
        logger.error(f'wrong orientation type {orient}')
        return a, b

def generate_nb_samples2plot(
        image_data:list[ImageDataset],
        nb_samples:Union[int,list[int]]=3,
        rand:bool=False) -> list[int]:
    """ """
    nb_samples_total=image_data[0].data.shape[0]
    if nb_samples_total==0:
        logger.error(f'image data do not contain any data!!!!)')
        return None

    if isinstance(nb_samples, list):
        if max(nb_samples)>nb_samples_total:
            logger.error(f'List of indexes : {nb_samples} is not correc')
            logger.info(f'first image will be plot')
            return [0]
        return nb_samples
    elif isinstance(nb_samples, int):
        if nb_samples==0:
            nb_samples=1
        if nb_samples>nb_samples_total:
            return None
        if rand:
            return random.sample(range(nb_samples_total), nb_samples)
        return range(nb_samples_total)
    
if __name__ == "__main__":
    from eit_tf_workspace.utils.log import change_level, main_log
    import logging
    main_log()
    change_level(logging.DEBUG)

    print()
    print([True for _ in range(4)])

    
    

