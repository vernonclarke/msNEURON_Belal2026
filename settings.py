'''
default settings
'''

import os
from os import walk
from tqdm import tqdm
import pickle
import random 
import csv
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
import colorednoise as cn
import datetime 

import seaborn as sns
import matplotlib as mpl

from plotly.subplots import make_subplots
import copy

# # Neuron
os.environ['NEURON_MODULE_OPTIONS'] = '-nogui'
from neuron import h
import neuron as nrn

# if model = 0 then original cell; if model = 1 then paper 
# for model = x uses MSN_builderx and params_dMSNx

# this will speed up simulations considerably IF impedance calculations are off
# Disable variable-step solver (explicitly use fixed step)
cvode = h.CVode()
cvode.active(0)
cvode.cache_efficient(1)

# Enable parallel threading
pc = h.ParallelContext()
pc.nthread(1)  # Try 3–5 on the M4 Mini; 4 is usually best
print("Threads active:", int(pc.nthread()))

mechs = None # 'single Nylen'
# model = 3
# if model is 3 then neck_dynamics can be set to True; default is False

neck_dynamics = False

# check if 'mechs' should be updated
update_mechs = mechs != 'single Nylen'  # Only update mechs if it is not 'single Nylen'

# Update 'mechs' based on the value of 'model' and set NMDA to AMPA conductance ratio
glut = False
AMPA = False
NMDA = False
g_AMPA = 0.001
num_gluts = 0
dend_glut = []
glut_frequency = 1000 # every 1ms
ratio = 2.15
if update_mechs:
    if model in [0, 1, 2]:
        mechs = 'single'
    elif model == 3:
        mechs = 'single3'
        ratio = 2.5

# phasic gaba
gaba = False
g_GABA = 0.001
gaba_reversal = -60
gaba_frequency = None
num_gabas = 0
dend_gaba = []
gaba_locations = []
rel_gaba_onsets = None

gaba_tau1 = 0.9
gaba_tau2 = 18

# Update 'model' based on the original or updated 'mechs' value
if mechs == 'single Nylen':
    model = 1
elif mechs == 'single3':
    model = 3

if mechs == 'single3':
    nrn.load_mechanisms('mechanisms/single3')
elif mechs == 'single':
    nrn.load_mechanisms('mechanisms/single')
elif mechs == 'single Nylen':
    # single new contains versions of mod file updates as per Nylen et al., 2023
    nrn.load_mechanisms('mechanisms/single Nylen')

# import custom functions
from master_functions import *
#  Setup neuron
h.load_file('stdlib.hoc')
h.load_file('import3d.hoc')

method = 1                  # if 0 stimulate away from soma; if 1 then stimulate towards
physiological = True        # if False then uses original values else more physiological settings for phasic conductances 

# cell_type='dspn'
specs = {'dspn': {
                    'N': 71,
                    'lib': 'Libraries/D1_71bestFit_updRheob.pkl',
                    'model': 2,
                    'morph': 'Morphologies/WT-dMSN_P270-20_1.02_SGA1-m24.swc',
                    },
         'ispn': {
                    'N': 34,
                    'lib': 'Libraries/D2_34bestFit_updRheob.pkl',
                    'model': 2,
                    'morph': 'Morphologies/WT-iMSN_P270-09_1.01_SGA2-m1.swc'}
        }


specs[cell_type]['model'] = model
morphology = specs[cell_type]['morph']
if model == 0:
    method = 0
    v_init = -83.67719
elif model == 1:
    v_init = -83.67719
elif model == 2:
    v_init = -84
elif model == 3:
    v_init = -85.4087
    
frequency = 2000 # determines nseg
d_lambda = 0.05  # determines nseg
dend2remove=None # removes any dendrite and its children from the cell; dend2remove = ['dend[42]', 'dend[0]']

# as at least 1 spine is added per nseg > 30um, then if nseg > total nspines can only control spine number by controlling d_lambda
# this is only an issue when/if you want each spine in its own segment.

# for default nsegs  < nspines so each spine is in its own segment

# default settings
holding_current = 0
current_step = False
if current_step:
    sim_time = 1000 
    step_end = sim_time - 200
    step_duration = 500
    step_start = step_end - step_duration
    step_current = 250 # 150 is rheobase
else:
    sim_time = 400
    step_current = None
    step_duration = None
    step_start = None

add_ramp = False
ramp_amplitude = 400

stim_time = 150
baseline = 20

gaba_locations = None      # if None then default placement is the midpoint of the dendrite
vary_gaba_time = False

timing_range = [stim_time]
gaba_range = np.repeat(True, len(timing_range)) 
glut_range = np.repeat(True, len(timing_range)) 

# each sim will produce some plots; turn off if large number of sims
Nsim_save = False
Nsim_plot = False

add_noise = False
beta=0                     # if add_noise is True then 0 is white noise
B=1                        # if add_noise is True then scaling factor B=1

add_sine = False 
axoshaft = False

Cm = 1
Ra = 200                  # Ra=1.59e-10 for silver wire for perfect space clamp

# scale conductances using scale_factors
scale_factors=None        # must be Python dict of form scale_factors = {'naf': 1.1, 'kas': 0.8, 'cav32': 0.6}

spine_per_length = 1.61
spine_neck_diam = 0.1
spine_neck_len = 1
spine_head_diam = 0.5
soma_diameter = None      # if None then default i.e 12.199999809265137

space_clamp = False
record_dendrite = None 
record_location = None 
record_currents = False     # for synaptic currents GABA and glutamatergic
record_branch = False       # if True then also determine voltages in all unique sections of that branch
dend_glut2 = None

record_mechs = False
mechs2record = None
record_path_dend = False    # if True then calculates voltages and i mechanisms (if record_mechs = True) in unique sections of dendrites within pathlist
record_path_spines = False  # if True then calculates voltages and i mechanisms (if record_mechs = True) in unique spines within pathlist
record_all_spines = False
record_all_path = True      # record at all unique points in the path (including points beyond site of activation)
record_3D = False           # record voltages at all unique segments of every section
record_3D_impedance = False # record voltages at all unique segments of every section
freq = 10                   # impedance measures made at 10Hz
record_3D_mechs = False     # record mechanisms at all unique segments of every section
record_Ca = False
record_3D_Ca = False
tonic = False               # add tonic GABA conductance
gbar_gaba = None            # add gbar for tonic GABA
rectification = False       # if tonic GABA then choose whether it is rectified or not
distributed = False         # can specify the distribution of tonic GABA using GABA params
gaba_params = None
tonic_gaba_reversal = -60

dt = 0.025 # 1 
ds_imp = 40 # downsample to ds_imp * dt

space_clamp = False
show_figs = True 

voltage_clamp = False
holding_potential = -84
voltage_clamp_site = 'soma[0]'
voltage_clamp_spine=False
Rs = 1e-9 
voltage_clamp_loc = 0.5

downsample = True
ds = 10

variable_names = [
     'AMPA',
     'axoshaft',
     'axospine',
     'baseline',
     'cell_coordinates',
     'cell_type',
     'Cm',
     'dend2remove',
     'dt',
     'model',
     'physiological',
     'NMDA',
     'dend_gaba',
     'dend_glut',
     'dt',
     'ds_imp',
     'Ndend_gaba',
     'Nsim_plot',
     'Nsim_save',
     'Nsims',
     'add_noise',
     'burn_time',
     'current_step',
     'dend_gaba',
     'dend_glut',
     'freq',
     'glut',
     'glut_time',
     'g_AMPA',
     'g_GABA',
     'g_GABA_range',
     'gaba',
     'gaba_time',
     'gaba_tau1',
     'gaba_tau2',  
     'gbar_gaba',
     'gaba_frequency',
     'gaba_locations',
     'gaba_locs',
     'gaba_range',
     'gaba_reversal',
     'glut_frequency',
     'glutamate_locations',
     'glutamate_locs',
     'glut_range',
     'holding_current',
     'impedance',
     'num_gabas',
     'num_gluts',
     'Ra',
     'ratio',
     'record_dendrite',
     'record_dends',
     'record_dists',
     'record_location',
     'record_locs',
     'rel_gaba_onsets',
     'save',
     'scale_factors',
     'showplot',
     'sim',
     'sim_time',
     'space_clamp',
     'spine_per_length',
     'spine_neck_diam',
     'spine_neck_len',
     'spine_head_diam',
     'soma_diam',
     'start_time',
     'stim_time',
     'timing',
     'timing_range',
     'tonic',
     'vary_gaba_location',
     'vary_gaba_time',
     'vary_location',
     'voltage_clamp',
     'holding_potential',
     'voltage_clamp_site',
     'voltage_clamp_spine',
     'Rs',
     'voltage_clamp_loc',
     'downsample',
     'ds',
     'deltat'
]

# jupyter nbconvert --to script settings.ipynb

