# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:39:17 2023

@author: Jake
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from RNN_XOR_class import RNN
plt.style.use(['ggplot','paper_plots']) #add 'paper_plots.mplstyle' to your style sheets folder if you want to use it

#%% Initialise the params of a nonlinear RNN
seed = None
rnn = RNN(noise_std = 0.255, reg = 0.02525)
params_init = rnn.initialise_params(seed=seed)
x,r,outputs = rnn.run_rnn_batch(params_init,seed=seed)

#%% Train rnn
params_trained = rnn.train_rnn(params_init,seed=seed,n_train_steps=1000)

#%% Plot training cost
rnn.plot_cost()

#%% Plot readouts of the trained network
x,r,outputs = rnn.run_rnn_batch(params_trained,seed=seed)
rnn.plot_outputs(outputs)

#%% Temporal decoding of task variables
decoding_init = rnn.temporal_decoding(params_init)
decoding_trained = rnn.temporal_decoding(params_trained)
rnn.plot_temporal_decoding(decoding_init,decoding_trained)

#%% Changes in stimulus coding
stim_coding_init = rnn.stimulus_coding(params_init)
stim_coding_trained = rnn.stimulus_coding(params_trained)

rnn.plot_changes_stim_coding(stim_coding_init,stim_coding_trained)

#%% Changes in firing rates
rates_init = rnn.norm_firing_rates(params_init)
rates_trained = rnn.norm_firing_rates(params_trained)

rnn.plot_changes_firing_rates(rates_init,rates_trained)
