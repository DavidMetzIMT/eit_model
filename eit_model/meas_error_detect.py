import logging
import numpy as np
import pandas as pd
# from eit_model.model import EITModel
import seaborn as sns
import matplotlib.pyplot as plt


from eit_model.data import EITData

logger = logging.getLogger(__name__)

def Meas_error_plot(data: np.ndarray):
    """detect the error values during measurement

    Args:
        data (np.ndarray): should be (256, frames)

    Returns:
        
    """
    df = pd.DataFrame(data)
    df = df.applymap(filter_value)
    dfm = df.reset_index().melt('index', var_name='frames',  value_name='vals')
    dfm['index'] = dfm['index'].apply(lambda x: x % 16 + 1)
    
    df_plot = dfm.loc[dfm['vals'] == 1]
    # fig, ax = plt.subplots()
    fig = sns.histplot(x="frames", y="index", data=df_plot, bins=100,cbar=True)
    return fig
    

def filter_value(x):
    if np.abs(x) < 0.5:
    # if np.abs(x) < 0.00001:
        x = 1
    else:
        x = 0
    return x


if __name__ == '__main__':

    data = np.random.randn(256, 9)
    
    Meas_error_plot(data)
    plt.show()
    
