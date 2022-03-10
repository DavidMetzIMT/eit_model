### this code is called by EVAL.py

from logging import getLogger
import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.interp2d
import pyeit.mesh.utils
from matplotlib import axes, figure

from eit_model.image import EITImage


logger = getLogger(__name__)




def get_elem_nodal_data(fwd_model, perm):
    """ check mesh (tri, pts) in fwd_model and provide elems_data and nodes_data """

    tri = np.array(fwd_model['elems'])
    pts = np.array(fwd_model['nodes'])

    # perm= fwd_model['un2']    
    perm= np.reshape(perm, (perm.shape[0],))

    # tri = tri-1 # matlab count from 1 python from 0
    tri= pyeit.mesh.utils.check_order(pts, tri)

    if perm.shape[0]==pts.shape[0]:
        data = {
            'elems_data': pyeit.eit.interp2d.pts2sim(tri, perm),
            'nodes_data': perm
        }
    elif perm.shape[0]==tri.shape[0]:
        data = {
            'elems_data': perm,
            'nodes_data': pyeit.eit.interp2d.sim2pts(pts, tri, perm)
        }

    for key in data.keys():
        data[key]= np.reshape(data[key], (data[key].shape[0],))
    return tri, pts, data

def format_inputs(fwd_model, data):
    if data.ndim==2:
        tri = np.array(fwd_model['elems'])
        pts = np.array(fwd_model['nodes'])
        if data.shape[1] in [pts.shape[0], tri.shape[0]]:
            data= data.T
    return data

def plot_EIT_samples(fwd_model, perm, U):

    perm=format_inputs(fwd_model, perm)
    U=format_inputs(fwd_model, U)

    tri, pts, data= get_elem_nodal_data(fwd_model, perm)

    key = 'nodes_data' if perm.shape[0]==pts.shape[0] else 'elems_data'
    fig, ax = plt.subplots(1,2)
    im = ax[0].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[key]),shading='flat', vmin=None,vmax=None)
    title= key

    if np.all(perm <= 1):
        title += '\nNormalized conductivity distribution'
    else:
        title += '\nConductivity distribution'
    ax[0].set_title(title)
    ax[0].set_xlabel("X axis")
    ax[0].set_ylabel("Y axis")

    ax[0].axis("equal")
    fig.colorbar(im,ax=ax[0])

    ax[1].plot(U.T)

    plt.show(block=False)

def plot_real_NN_EIDORS(fwd_model, perm_real,*argv):

    _perm = [perm_real]
    for arg in argv:
        if _perm[0].shape==arg.shape:
            _perm.append(arg)

    perm = []
    if perm_real.ndim > 1:
        n_row=  perm_real.shape[1]
        for p in _perm:
            perm.append(p)
    else:
        for p in _perm:
            perm.append(p.reshape((p.shape[0],1)))
        n_row= 1
    n_col = len(perm)

    fig, ax = plt.subplots(n_row,n_col)
    if ax.ndim==1:
        ax=ax.reshape((ax.shape[0],1)).T

    key= 'elems_data'
    for row in range(ax.shape[0]):

        data= [dict() for _ in range(n_col)]
        for i, p in enumerate(perm):
            tri, pts, data[i]= get_elem_nodal_data(fwd_model, p[:, row])
        for col in range(n_col):
            print(row, col)
            im = ax[row, col].tripcolor(pts[:,0], pts[:,1], tri, np.real(data[col][key]),shading='flat', vmin=None,vmax=None)
            title= key + f'#{row}'

            # if np.all(perm <= 1):
            #     title= title +'\nNormalized conductivity distribution'
            # else:
            #     title= title +'\nConductivity distribution'
            ax[row, col].set_title(title)
            ax[row, col].set_xlabel("X axis")
            ax[row, col].set_ylabel("Y axis")

            ax[row, col].axis("equal")
            fig.colorbar(im,ax=ax[row, col])

    plt.show(block=False)


# def plot_EIT_image(fig:figure.Figure, ax:axes.Axes, image:EITImage, show:list[bool]=[True] * 4, colorbar_range:list[int]=[0,1])-> None:
#     """[summary]

#     Args:
#         fig (figure): [description]
#         ax (axes): [description]
#         image (ImageEIT): [description]
#         show (list[bool], optional): [description]. Defaults to [True*4].
#     """    
    
#     tri, pts, data= get_elem_nodal_data(image.fem, image.data)

#     key= 'elems_data'
#     perm=np.real(data[key])
#     if np.all(perm <= 1) and np.all(perm > 0):
#         colorbar_range=[0,1]
#         title= image.label +'\nNorm conduct'
#     else:
#         title= image.label +'\nConduct'
#     im = ax.tripcolor(pts[:,0], pts[:,1], tri, perm, shading='flat', vmin=colorbar_range[0],vmax=colorbar_range[1])
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
    
def plot_EIT_image(fig:figure.Figure, ax:axes.Axes, image:EITImage, show:list[bool]=[True] * 4, colorbar_range:list[int]=None)-> None:
    """[summary]

    Args:
        fig (figure): [description]
        ax (axes): [description]
        image (ImageEIT): [description]
        show (list[bool], optional): [description]. Defaults to [True*4].
    """    
    if colorbar_range is None:
        colorbar_range=[None, None]
    tri, pts, data= get_elem_nodal_data(image.fem, image.data)


    key= 'elems_data'
    perm=np.real(data[key])
    if np.all(perm <= 1) and np.all(perm > 0):
        colorbar_range=[0,1]
        title= image.label +'\nNorm conduct'
    else:
        title= image.label +'\nConduct'
    im = ax.tripcolor(pts[:,0], pts[:,1], tri, perm, shading='flat', vmin=colorbar_range[0],vmax=colorbar_range[1])
    
    fig, ax= add_elec_numbers(fig, ax, image)

    
    # ax.axis("equal")
    # fig.set_tight_layout(True)
    # ax.margins(x=0.0, y=0.0)
    ax.set_aspect('equal', 'box')
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    # ax.axis('off')
    if show[0]:
        ax.set_title(title)
    if show[1]:
        ax.axis('on')
        ax.set_xlabel("X axis")
    if show[2]:
        ax.set_ylabel("Y axis")
    if show[3]:    
        fig.colorbar(im,ax=ax)
    return fig, ax, im

def set_plot_labels(fig:figure.Figure, ax:axes.Axes, label):



    return fig , ax


def add_elec_numbers(fig:figure.Figure, ax:axes.Axes, image:EITImage):

    elec_x=image.fem['elec_pos'][:,0]
    elec_y=image.fem['elec_pos'][:,1]

    ax.plot(elec_x,elec_y, "ok")
    for i, (x, y ) in enumerate(zip(elec_x,elec_y)):
        ax.text(x, y, i+1, color="red", fontsize=12)

    return fig , ax
    
    

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    import glob_utils.files.matlabfile

    import glob_utils.files.files
    import glob_utils.log.log
    glob_utils.log.log.main_log()

    print()
    print([True for _ in range(4)])

    
    

