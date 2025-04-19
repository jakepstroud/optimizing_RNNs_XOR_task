# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:37:36 2023

@author: Jake
"""

import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
from scipy.special import softmax
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable

#%% Create RNN class
class RNN:
    def __init__(self, N = 50,T = 2000, tau = 50, dt = 1, 
                 noise_std = 0.01, reg = 0.0005, init_input_weight_scale = 'small'):
        """
        Initialize an RNN model with given hyperparameters.

        Parameters:
        N (int): Number of neurons.
        T (int): Length of simulation in ms.
        tau (float): Time constant in ms.
        dt (float): Euler step size in ms.
        noise_std (float): Standard deviation of the Gaussian noise.
        reg (float): Regularization coefficient (metabolic cost).
        init_input_weight_scale (str): Scale of input weights ('0', 'small', or 'large').
        """
        self.N = N #number of neurons        
        self.T = T #length of simulation (ms)
        self.tau = tau #time constant (ms)
        self.dt = dt #euler step size (ms)
        self.dt_tau = dt/tau #dt/tau
        self.noise_std = noise_std #std of noise in neural activities
        self.reg = reg #firing rate regularization (metabolic cost)
        self.batch_size = 10
        
        self.color_on = 500
        self.shape_on = 1000
        self.stim_off = 1500
        
        if init_input_weight_scale == '0':
            self.scale = 0 #std of initial network weights
        elif init_input_weight_scale == 'small':
            self.scale = 1/np.sqrt(self.N) #std of initial network weights
        elif init_input_weight_scale == 'large':
            self.scale = 10/np.sqrt(self.N) #std of initial network weights
               
        self.shapes_stack = np.array([[1,0],[1,0],[0,1],[0,1],[0,1],[0,1],[1,0],[1,0]]).T #size 2x8
        self.widths_stack = np.tile(np.array([[1,0],[0,1]]),(4,1)).T #size 2x8
        self.colors_stack = np.array([[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1]]).T #size 2x8
        self.n_colors = 2
        self.n_conds = 8
        
        #Create one hot encoded target choices for training
        tgt_np = np.zeros((self.n_conds,2))
        idx = np.where(np.tile([1,1,0,0],self.n_colors))
        tgt_np[idx,0] = 1
        tgt_np[:,1] = 1 - tgt_np[:,0]
        tgt_np = np.transpose(np.stack([tgt_np]*self.batch_size),[0,2,1])
        self.tgt_np = tgt_np #shape: n_conds,2,batch_size
        
    #Create numpy relu activation function
    def act_fun(self,x):
        """
        Applies a ReLU activation function to the input.

        Parameters:
        x (ndarray): Input array.

        Returns:
        ndarray: Output after applying ReLU.
        """
        return np.maximum(0,x)
        
    #Simulate RNN dynamics in numpy
    def run_rnn(self,params, seed = None):
        """
        Simulates RNN dynamics over time for all task conditions (non-batched).

        Parameters:
        params (dict): Dictionary of RNN parameters (weights and biases).
        seed (int): Optional random seed.

        Returns:
        tuple: Raw neural activity (x), firing rates (r), and output probabilities.
        """
        np.random.seed(seed) #fix random seed if given

        color_weights = params['color_weights']
        shape_weights = params['shape_weights']
        width_weights = params['width_weights']
        W = params['W']
        b = params['b']
        w_out = params['w_out']
        b_out = params['b_out']
        
        x = np.zeros((self.T,self.N,self.n_conds))
        for t in range(self.T-1):
            x[t+1,:] = x[t,:] + (self.dt_tau)*(-x[t,:] + W @ self.act_fun(x[t,:]) + b) + np.sqrt(self.dt_tau)*self.noise_std*np.random.normal(0,1,(self.N,self.n_conds))
            
            if t >= self.color_on and t < self.shape_on: #color period
                x[t+1,:] += (self.dt_tau)*(color_weights @ self.colors_stack)
            elif t >= self.shape_on and t < self.stim_off: #shape period
                x[t+1,:] += (self.dt_tau)*(color_weights @ self.colors_stack + shape_weights @ self.shapes_stack + width_weights @ self.widths_stack)
            
        r = self.act_fun(x)
        outputs = softmax(w_out @ self.act_fun(x) + b_out,axis=-2)
        return x,r,outputs
        
    #Simulate RNN dynamics in a batch in numpy
    def run_rnn_batch(self,params,seed = None):
        """
        Simulates RNN dynamics over time using batch processing.

        Parameters:
        params (dict): Dictionary of RNN parameters.
        seed (int): Optional random seed.

        Returns:
        tuple: Batched raw activity (x), firing rates (r), and output probabilities.
        """
        np.random.seed(seed) #fix random seed if given
        
        color_weights = params['color_weights']
        shape_weights = params['shape_weights']
        width_weights = params['width_weights']
        W = params['W']
        b = np.expand_dims(params['b'],axis=0)
        w_out = params['w_out']
        b_out = np.expand_dims(params['b_out'],axis=1)
        
        x = np.zeros((self.T,self.batch_size,self.N,self.n_conds))
        for t in range(self.T-1):
            x[t+1,:] = x[t,:] + (self.dt_tau)*(-x[t,:] + W @ self.act_fun(x[t,:]) + b) + np.sqrt(self.dt_tau)*self.noise_std*np.random.normal(0,1,(self.batch_size,self.N,self.n_conds))
            
            if t >= self.color_on and t < self.shape_on: #color period
                x[t+1,:] += (self.dt_tau)*(color_weights @ self.colors_stack)
            elif t >= self.shape_on and t < self.stim_off: #shape period
                x[t+1,:] += (self.dt_tau)*(color_weights @ self.colors_stack + shape_weights @ self.shapes_stack + width_weights @ self.widths_stack) #go cue is just sum of all inputs - as in the monkey experiment
            
        r = self.act_fun(x)
        outputs = softmax(w_out @ self.act_fun(x) + b_out,axis=-2)
        return x,r,outputs
        
    #Initialise network parameters before training
    def initialise_params(self,seed = None):
        """
        Initializes network parameters with random values.

        Parameters:
        seed (int): Optional random seed.

        Returns:
        dict: Initialized RNN parameters.
        """
        np.random.seed(seed) #fix random seed if given

        color_weights = np.random.normal(0,self.scale,(self.N,2))
        shape_weights = np.random.normal(0,self.scale,(self.N,2))
        width_weights = np.random.normal(0,self.scale,(self.N,2))
        
        W = np.random.normal(0,1/np.sqrt(self.N),(self.N,self.N))
        b = np.random.normal(0,1/np.sqrt(self.N),(self.N,1))
        
        w_out = np.random.normal(0,1/np.sqrt(self.N),(2,self.N))
        b_out = np.random.normal(0,1/np.sqrt(self.N),(1,2,1))
        
        params_init = {
            "color_weights": color_weights,
            "shape_weights": shape_weights,
            "width_weights": width_weights,
            "W": W,
            "b": b,
            "w_out": w_out,
            "b_out": b_out}
        
        return params_init

    #%% Tensorflow stuff
    @tf.function
    def act_fun_tf(self,x_tf):
        """
        TensorFlow version of ReLU activation function.

        Parameters:
        x_tf (Tensor): Tensor input.

        Returns:
        Tensor: Output after applying ReLU.
        """
        return tf.nn.relu(x_tf)
    
    @tf.function
    def cost_fun(self,outputs_tf): #Cross-entropy loss
        """
        Computes cross-entropy loss between outputs and target.

        Parameters:
        outputs_tf (Tensor): Network output logits.

        Returns:
        Tensor: Computed loss.
        """
        return -tf.reduce_sum(self.tgt_tf*(tf.math.log(tf.nn.softmax(outputs_tf,axis=-2))))

    @tf.function
    def cond(self,x_tf,t,color_weights_tf,shape_weights_tf,width_weights_tf,
             W_tf,b_tf,w_out_tf,b_out_tf,cost):
        """
        Condition function for tf.while_loop.

        Parameters:
        x_tf (Tensor): RNN state.
        t (Tensor): Time step.

        Returns:
        Tensor: Boolean condition for while loop.
        """
        return tf.less(t,self.T)
    
    @tf.function
    def body(self,x_tf,t,color_weights_tf,shape_weights_tf,width_weights_tf,
             W_tf,b_tf,w_out_tf,b_out_tf,cost):
        """
        Body function for tf.while_loop that updates RNN state.

        Returns:
        tuple: Updated variables after one step.
        """
        #Run rnn dynamics
        x_tf = x_tf + self.dt_tau*(-x_tf + W_tf @ self.act_fun_tf(x_tf) + b_tf) + tf.math.sqrt(self.dt_tau_tf)*self.noise_std*tf.random.normal((self.batch_size,self.N,self.n_conds),dtype=tf.float64)
        
        #Color period
        x_tf = tf.cond(tf.logical_and(t >= self.color_on,t < self.shape_on),
                        lambda: x_tf + self.dt_tau*(color_weights_tf @ self.colors_stack_tf), 
                        lambda: x_tf) #apply the inputs during the stimulus period
        
        #Shape period
        x_tf = tf.cond(tf.logical_and(t >= self.shape_on,t < self.stim_off),
                        lambda: x_tf + self.dt_tau*(color_weights_tf @ self.colors_stack_tf + shape_weights_tf @ self.shapes_stack_tf + width_weights_tf @ self.widths_stack_tf), 
                        lambda: x_tf) #apply the inputs during the stimulus period
                    
        r_tf = self.act_fun_tf(x_tf)
        
        #Apply rate regularisation at all times
        cost += self.reg*tf.nn.l2_loss(r_tf)
                
        #Apply cross entropy loss using the provided cost type (either just_in_time, cue_delay, or after_go_time)
        cost = tf.cond(t > 1500,
                       lambda:cost + self.cost_fun(w_out_tf@r_tf + b_out_tf),
                       lambda:cost)
        
        return x_tf,t+1,color_weights_tf,shape_weights_tf,width_weights_tf,W_tf,b_tf,w_out_tf,b_out_tf,cost
    
    @tf.function
    def run_rnn_while(self,color_weights_tf,shape_weights_tf,width_weights_tf,
             W_tf,b_tf,w_out_tf,b_out_tf):
        """
        Executes the RNN forward pass and computes loss using TensorFlow while loop.

        Returns:
        Tensor: Total loss after simulation.
        """
        cost = tf.constant(0.0,dtype=tf.float64)
        t = tf.constant(0,dtype=tf.int64)
        x_tf = tf.zeros((self.batch_size,self.N,self.n_conds),dtype=tf.float64)
        
        b_tf = tf.expand_dims(b_tf,axis=0)
        b_out_tf = tf.expand_dims(b_out_tf,axis=1)
        
        self.dt_tau_tf = tf.constant(self.dt_tau,dtype=tf.float64)
        self.colors_stack_tf = tf.constant(self.colors_stack,dtype=tf.float64)
        self.shapes_stack_tf = tf.constant(self.shapes_stack,dtype=tf.float64)
        self.widths_stack_tf = tf.constant(self.widths_stack,dtype=tf.float64)
        self.tgt_tf = tf.constant(self.tgt_np,dtype=tf.float64)        
        
        x_tf,t,color_weights_tf,shape_weights_tf,width_weights_tf,W_tf,b_tf,w_out_tf,b_out_tf,cost = tf.while_loop(
            self.cond, self.body, (x_tf,t,color_weights_tf,shape_weights_tf,width_weights_tf,
             W_tf,b_tf,w_out_tf,b_out_tf,cost), parallel_iterations=10)
        return cost
    
    def initialise_tf_params(self,params_init):    
        """
        Converts NumPy parameter arrays to TensorFlow variables.

        Parameters:
        params_init (dict): Initialized NumPy parameters.

        Returns:
        tuple: TensorFlow variables for RNN parameters.
        """
        color_weights_tf = tf.Variable(params_init['color_weights'],dtype=tf.float64)
        shape_weights_tf = tf.Variable(params_init['shape_weights'],dtype=tf.float64)
        width_weights_tf = tf.Variable(params_init['width_weights'],dtype=tf.float64)
        
        W_tf = tf.Variable(params_init['W'],dtype=tf.float64)
        b_tf = tf.Variable(params_init['b'],dtype=tf.float64)
        
        w_out_tf = tf.Variable(params_init['w_out'],dtype=tf.float64)
        b_out_tf = tf.Variable(params_init['b_out'],dtype=tf.float64)   
        
        return color_weights_tf,shape_weights_tf,width_weights_tf,W_tf,b_tf,w_out_tf,b_out_tf
        
    def train_rnn(self,params_init,learning_rate = 0.001,
                  n_train_steps = 1000,seed = None):
        """
        Trains the RNN using TensorFlow and backpropagation through time.

        Parameters:
        params_init (dict): Initialized parameters.
        learning_rate (float): Learning rate for optimizer.
        n_train_steps (int): Number of training steps.
        seed (int): Optional seed for reproducibility.

        Returns:
        dict: Trained RNN parameters.
        """
        tf.random.set_seed(0)
        self.learning_rate = learning_rate #learning rate
        self.n_train_steps = n_train_steps #number of training steps (200 may be enough, but 1000 may be needed for complete convergence)
        cost_over_training = tf.TensorArray(size = self.n_train_steps,dtype=tf.float64)        

        color_weights_tf,shape_weights_tf,width_weights_tf,W_tf,b_tf,w_out_tf,b_out_tf = self.initialise_tf_params(params_init)
        
        #Force to CPU
        my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
        tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')        
        optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)      
        
        #Train
        for epoch in range(self.n_train_steps):
            with tf.GradientTape() as tape:
                cost = self.run_rnn_while(color_weights_tf,shape_weights_tf,width_weights_tf,
                                          W_tf,b_tf,w_out_tf,b_out_tf)
                
            gradients = tape.gradient(cost,[color_weights_tf,shape_weights_tf,width_weights_tf,
                                            W_tf,b_tf,w_out_tf,b_out_tf])    
            optimizer.apply_gradients(zip(gradients,[color_weights_tf,shape_weights_tf,width_weights_tf,
                                                     W_tf,b_tf,w_out_tf,b_out_tf]))        
            cost_over_training.write(epoch,cost) #Grab cost over time
    
            if epoch%10==0:
                print('Num. steps:',epoch,', cost:',int(cost.numpy()))
        
        cost_over_training = cost_over_training.stack()
        self.cost_over_training = cost_over_training.numpy()
        
        #Explicity convert to numpy arrays
        params_trained = {
            "color_weights": color_weights_tf.numpy(),
            "shape_weights": shape_weights_tf.numpy(),
            "width_weights": width_weights_tf.numpy(),
            "W": W_tf.numpy(),
            "b": b_tf.numpy(),
            "w_out": w_out_tf.numpy(),
            "b_out": b_out_tf.numpy()}
         
        return params_trained
    
    #%% Plotting    
    def plot_cost(self):
        """
        Plots cost over training iterations.
        """
        plt.plot(self.cost_over_training)
        plt.xlabel('training iterations')
        plt.ylabel('cost')

    def plot_outputs(self,outputs):
        """
        Plots RNN output probabilities over time for different condition types.

        Parameters:
        outputs (ndarray): Output probabilities over time.
        """
        rew_idx = [0,1,4,5]
        n_rew_idx = [2,3,6,7]
        
        outputs = np.mean(outputs,axis=1) #take mean over batches
        # outputs = outputs[:,0,:]
        plt.plot(np.mean(outputs[:,0,rew_idx],axis=-1),'-C0',clip_on=False,label='true conditions')
        plt.plot(np.mean(outputs[:,0,n_rew_idx],axis=-1),'-C1',clip_on=False,label='false conditions')
        
        # plt.plot(outputs[:,1,rew_idx],'-C0',clip_on=False,label='false conditions')
        # plt.plot(outputs[:,1,n_rew_idx],'-C1',clip_on=False,label='true conditions')

        plt.xticks([0,500,1000,1500,2000],['-0.5','0','0.5','1','1.5'])
        plt.xlabel('time (s)');plt.ylabel('readouts')
        plt.legend()
        plt.ylim([0,1])
        
    def plot_temporal_decoding(self,decoding_init,decoding_trained):
        """
        Plots temporal decoding performance before and after training.

        Parameters:
        decoding_init (ndarray): Initial decoding performance.
        decoding_trained (ndarray): Decoding after training.
        """
        fig, axs = plt.subplots(1,4,figsize=(4,1))
        plt.subplots_adjust(wspace=0.5)
        stimuli = ['XOR','shape','color','width']
        decoding_init = np.mean(decoding_init,axis=-1)
        decoding_trained = np.mean(decoding_trained,axis=-1)
        
        for i in range(4):
            axs[i].plot(decoding_init[i,:],'-',color=[0.5,0.5,0.5],clip_on=False)
            axs[i].plot(decoding_trained[i,:],'-k',clip_on=False)
            axs[i].plot([0,200],[0.5,0.5],':',color=[0.5,0.5,0.5])
            axs[i].set_xlabel('time (s)')
            axs[i].set_xticks([0,50,100,150,200])
            axs[i].set_xticklabels(['-0.5','0','0.5','1','1.5'])
            axs[i].set_yticks([0.5,0.75,1])
            axs[i].set_ylim([0.45,1])
            axs[i].set_title(stimuli[i],pad=15)
            
            if i == 0:
                axs[0].set_ylabel('decoding accuracy')
                axs[0].legend(['early learning','late learning'],handlelength=1,ncol=2,loc=(-0.75,1.05))
        
    def plot_changes_stim_coding(self,stim_coding_init,stim_coding_trained):
        """
        Plots change in stimulus representation and overlap before/after training.

        Parameters:
        stim_coding_init (ndarray): Initial stimulus coding values.
        stim_coding_trained (ndarray): Final stimulus coding values.
        """
        fig, axs = plt.subplots(1,2,figsize=(2,1))
        plt.subplots_adjust(wspace=1)
        red = '#c12b2b'
        
        axs[0].plot([stim_coding_init[0],stim_coding_trained[0]],'.-',color=red)
        axs[0].plot([stim_coding_init[1],stim_coding_trained[1]],'.-k')
        axs[1].plot([stim_coding_init[1],stim_coding_trained[1]],'.-k')
        
        axs[0].set_xticks([0,1])
        axs[1].set_xticks([0,1])
        axs[0].set_xticklabels(['early\nlearning','late\nlearning'])
        axs[1].set_xticklabels(['early\nlearning','late\nlearning'])
        
        axs[0].set_ylabel('stimulus coding')
        axs[1].set_ylabel('overlap')
        
        axs[0].legend(['rel.','irrel.'],handlelength=1)
        
        axs[0].set_ylim([0,None])
        axs[1].set_ylim([0,None])
    
    def plot_changes_firing_rates(self,rates_init,rates_trained):
        """
        Plots changes in average firing rates before and after training.

        Parameters:
        rates_init (ndarray): Initial firing rates.
        rates_trained (ndarray): Final firing rates.
        """
        plt.plot(rates_init,'.-',color=[0.5,0.5,0.5])
        plt.plot(rates_trained,'.-k')
        plt.xticks([0,1,2],['fixation\nperiod','color\nperiod','shape\nperiod'])
        # plt.ylabel(r'$\frac{||\mathbf{r}(t)||_2}{\sqrt{N}}$')
        plt.ylabel('metabolic cost')
        plt.legend(['early learning','late learning'],handlelength=1)
        plt.ylim([0,None])
        
    #%% Analysis
    def temporal_decoding(self,params_in):
        """
        Computes decoding accuracy of different task variables over time.

        Parameters:
        params_in (dict): Parameters of the RNN.

        Returns:
        ndarray: Decoding accuracy matrix of shape (4, time, runs).
        """
        lin_model = SVC(kernel='linear')
        targets_x = np.stack([[1,1,0,0,1,1,0,0]]*self.batch_size).flatten() #xor
        targets_s = np.stack([[0,0,1,1,1,1,0,0]]*self.batch_size).flatten() #shape        
        targets_c = np.stack([[1,1,1,1,0,0,0,0]]*self.batch_size).flatten() #color
        targets_w = np.stack([[1,0,1,0,1,0,1,0]]*self.batch_size).flatten() #width
        all_targets = [targets_x,targets_s,targets_c,targets_w]
        
        num_times = int(self.T/10)
        decoding = np.zeros((4,num_times,10))        
        
        for run in range(10):
            _,r_train,_ = self.run_rnn_batch(params_in)
            _,r_test,_ = self.run_rnn_batch(params_in)
    
            r_train = np.transpose(r_train,[0,1,3,2])
            r_test = np.transpose(r_test,[0,1,3,2])
            
            for t in range(num_times):            
                r_train_shaped = np.reshape(r_train[10*t,:],(self.batch_size*self.n_conds,self.N))
                r_test_shaped = np.reshape(r_test[10*t,:],(self.batch_size*self.n_conds,self.N))
                
                for i in range(4):
                    lin_model.fit(r_train_shaped,all_targets[i])
                    decoding[i,t,run] = lin_model.score(r_test_shaped,all_targets[i])
                
        return decoding
    
    def stimulus_coding(self,params_in):
        """
        Computes stimulus-related coding using regression on neural activity.

        Parameters:
        params_in (dict): Parameters of the RNN.

        Returns:
        tuple: Regression weights and overlaps for further analysis.
        """
        _,r,_ = self.run_rnn_batch(params_in)
        r = np.transpose(r[1400:1500,:],[0,1,3,2])
            
        lin_model = linear_model.LinearRegression(fit_intercept=True)
        X = 0.5*np.ones((self.n_conds,2))
        X[[2,3,6,7],0] = -0.5 #XOR
        X[[1,3,5,7],1] = -0.5 #width
        X = np.stack([X]*self.batch_size)
        X = np.reshape(X,(self.batch_size*self.n_conds,2))
        
        t_tot = r.shape[0]
        rel_coding = np.zeros((self.N,t_tot))
        irrel_coding = np.zeros((self.N,t_tot))            
        
        for t in range(t_tot):
            data = np.reshape(r[t,:],(self.batch_size*self.n_conds,self.N))
            result = lin_model.fit(X,data)
            rel_coding[:,t] = result.coef_[:,0]
            irrel_coding[:,t] = result.coef_[:,1]

        rel_coding = np.mean(rel_coding,axis=1) #average over time
        irrel_coding = np.mean(irrel_coding,axis=1)
        
        delta_rel = np.linalg.norm(rel_coding)/np.sqrt(self.N)
        delta_irrel = np.linalg.norm(irrel_coding)/np.sqrt(self.N)
        overlap = np.abs(np.sum(rel_coding*irrel_coding)/(np.linalg.norm(rel_coding)*np.linalg.norm(irrel_coding)))
    
        return [delta_rel,delta_irrel,overlap]
    
    def norm_firing_rates(self,params_in):
        
        _,r,_ = self.run_rnn_batch(params_in)
        fix_rates = np.mean(np.linalg.norm(r[400:500,:],axis=2))/np.sqrt(self.N)
        color_rates = np.mean(np.linalg.norm(r[900:1000,:],axis=2))/np.sqrt(self.N)
        shape_rates = np.mean(np.linalg.norm(r[1400:1500,:],axis=2))/np.sqrt(self.N)

        return [fix_rates,color_rates,shape_rates]
        
    
    
    
