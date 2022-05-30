
from ast import Call
from subprocess import call
from typing import Callable
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict

DIST_PARAMS = {
    'normal' : ['loc', 'scale']
    , 'poisson' : ['lam']
    , 'exponential' : ['scale']
    , 'weibull' : ['a']
}

def create_sidebar()->Tuple[Callable, dict]:
    """
    Creates all the widgets needed to define the numpy random function from which the observations will be sampled

    Returns
    -------
    Tuple[Callable, dict]
        the numpy random function to use to sample the observations and the parameters of the distribution
    """    
    _ = st.sidebar.empty()
    selected_dist = st.sidebar.selectbox(
        'Distribution',
        list(DIST_PARAMS)
    )
    func = getattr(np.random, selected_dist)
    all_params = {param: st.sidebar.number_input(label=param) for param in DIST_PARAMS[selected_dist]}
    return func, all_params

def create_middle(func:Callable
                , all_params:dict)->None:
    col1, col2= st.columns([2,1])
    with col1:
        #this xolumn contains the parameters and the viz
        
        #the remaining inputs
        n_per_sample = st.number_input(label='#obs per sample', value=1000)
        n_samples = st.number_input(label='#samples', value=100)
        bins = st.number_input(label='nbins', value=50)
        
        #sample the data from the distributions
        all_data = func(**all_params, size=(n_samples, n_per_sample))
        summarized_data = pd.Series(np.mean(all_data, axis=1))
        
        #create the figure
        fig, axes = plt.subplots(nrows=6, ncols=1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 5]})

        #create the histogram (here, for simplicity I am capping it at 4 graphs)
        axs = (pd.DataFrame(all_data)
            .T
            .melt()
            .set_index('variable')
            .loc[:4].reset_index()
            .plot(kind='hist', by='variable', bins=bins, legend=False, ax=axes[:-1],title=['', '','','',''])
            
        )
        #remove the y label of thre subplots for a less cluttered view
        _ = [ax.axes.get_yaxis().set_visible(False) for ax in axs]

        #produce the histogram of the sample of means
        ax2=summarized_data.plot(kind='hist'
                            , bins=bins
                            , legend=False
                            , ax=axes[-1])
        sns.despine()

        ax2.set_ylabel('Frequency', rotation=0)
        ax2.yaxis.set_label_coords(-.1,1.05)
        _ = st.pyplot(fig=fig)
    with col2:
        #this column will display the data in pandas dataframes
        right_column(all_data=all_data, summarized_data=summarized_data)
    return 

def right_column(all_data:pd.DataFrame
                , summarized_data:pd.DataFrame
                )->None:
    """
    Creates the widgets to display the dataframes

    Parameters
    ----------
    all_data : pd.DataFrame
        The samples created
    summarized_data : pd.DataFrame
        A pd.DataFrame with the sampled means from the all_data dataframe
    """    
    st.text('Summary of individual samples')
    st.dataframe(pd.DataFrame(all_data).T.describe().T['mean std'.split()])
    st.text('Summary of the mean of samples')
    st.dataframe(summarized_data.describe().loc['mean std'.split()])

def main()->None:
    func, all_params = create_sidebar()
    create_middle(func=func, all_params=all_params)

if __name__=='__main__':
    main()