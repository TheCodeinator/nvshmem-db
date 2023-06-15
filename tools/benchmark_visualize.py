import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

'''
Definition of all plotting routines for benchmark visualization
'''

def plot_2d(X,Y,**kwargs):

    fig, axs = plt.subplots()
    
    directory = kwargs["directory"]
    fig_name = kwargs["fig_name"]
    filetype = kwargs["filetype"]


    axs.plot(X,Y)

    plt.savefig(f"{directory}/{fig_name}.{filetype}")
