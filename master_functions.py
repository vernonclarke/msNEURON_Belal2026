'''
functions for model
'''
from   neuron           import h
import pandas as pd
import math as math
import numpy as np
import random 
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib as mpl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy

def cell_build(cell_type='dspn', specs=None, addSpines=False, branch=False, spine_per_length=1.61, frequency=2000, d_lambda=0.05, soma_diameter=None, verbose=True, dend2remove=None, neck_dynamics=False):
    model = specs[cell_type]['model']
    morphology = specs[cell_type]['morph']
    if model == 0:
        import MSN_builder0 as build
        if cell_type == 'dspn':
            params='params_dMSN0.json'
        elif cell_type == 'ispn':
            params='params_iMSN0.json'       
        h('forall delete_section()')
        h.define_shape()
        cell = build.MSN(params=params, morphology=morphology, variables=None)
        dend_tree = get_root_branches(cell)
        if dend2remove is not None:
            dends2remove = dendrite_removal(cell=cell, dend_tree=dend_tree, dend2remove=dend2remove)
            for section in dends2remove:
                h.disconnect(sec=section)
                h.delete_section(sec=section)
            dend_tree = get_root_branches(cell)
        if soma_diameter is not None:
            soma_resize(cell=cell, diam=soma_diameter)            
        if branch: branch_groups = get_root_groups(cell)
        if addSpines: spines = build.add_spines(cell, spines_per_sec=30)
    elif model == 1:
        import MSN_builder1 as build
        if cell_type == 'dspn':
            params='params_dMSN1.json'
        elif cell_type == 'ispn':
            params='params_iMSN1.json'
        h('forall delete_section()')
        h.define_shape()
        cell = build.MSN(params=params, morphology=morphology, variables=None, freq=frequency, d_lambda=d_lambda)
        dend_tree = get_root_branches(cell)
        if dend2remove is not None:
            dends2remove = dendrite_removal(cell=cell, dend_tree=dend_tree, dend2remove=dend2remove)
            for section in dends2remove:
                h.disconnect(sec=section)
                h.delete_section(sec=section)
            dend_tree = get_root_branches(cell)
        if soma_diameter is not None:
            soma_resize(cell=cell, diam=soma_diameter)            
        if branch: branch_groups = get_root_groups(cell)
        if addSpines: spines = build.add_spines(cell=cell, spine_per_length=spine_per_length, verbose=verbose)
    elif model == 2:
        import MSN_builder2 as build
        if cell_type == 'dspn':
            params='params_dMSN2.json'
        elif cell_type == 'ispn':
            params='params_iMSN2.json'
        h('forall delete_section()')
        h.define_shape()
        cell = build.MSN(params=params, morphology=morphology, variables=None)
        dend_tree = get_root_branches(cell)
        if dend2remove is not None:
            dends2remove = dendrite_removal(cell=cell, dend_tree=dend_tree, dend2remove=dend2remove)
            for section in dends2remove:
                h.disconnect(sec=section)
                h.delete_section(sec=section)
            dend_tree = get_root_branches(cell)
        if soma_diameter is not None:
            soma_resize(cell=cell, diam=soma_diameter)            
        if branch: branch_groups = get_root_groups(cell)
        if addSpines: spines = build.add_spines(params=params, cell=cell, spine_per_length=spine_per_length, verbose=verbose)
            
    elif model == 3:
        if neck_dynamics:
            # MSN_builder3a assumes channels in spine head are also in neck 
            # and have same distribution as the neighboring dendrite
            # in this model ONLY the spine head is unique
            import MSN_builder3a as build
        else:
            import MSN_builder3 as build
        if cell_type == 'dspn':
            params='params_dMSN3.json'
        elif cell_type == 'ispn':
            params='params_iMSN3.json'
        h('forall delete_section()')
        h.define_shape()
        cell = build.MSN(params=params, morphology=morphology, variables=None, freq=frequency, d_lambda=d_lambda)
        dend_tree = get_root_branches(cell)
        if dend2remove is not None:
            dends2remove = dendrite_removal(cell=cell, dend_tree=dend_tree, dend2remove=dend2remove)
            for section in dends2remove:
                h.disconnect(sec=section)
                h.delete_section(sec=section)
            dend_tree = get_root_branches(cell)
        if branch: branch_groups = get_root_groups(cell)
        if soma_diameter is not None:
            soma_resize(cell=cell, diam=soma_diameter)            
        if addSpines: 
            if neck_dynamics:
                spines = build.add_spines(params=params, cell=cell, spine_per_length=spine_per_length, verbose=verbose)
            else:
                spines = build.add_spines(cell=cell, spine_per_length=spine_per_length, verbose=verbose)

    if addSpines and branch: return(cell, spines, dend_tree, branch_groups)
    elif addSpines and not branch: return(cell, spines, dend_tree)
    elif not addSpines and branch: return(cell, dend_tree, branch_groups)
    elif not addSpines and not branch: return(cell, dend_tree)   
    
def dendrite_removal(cell, dend_tree, dend2remove):
    target_list = []
    for target in dend2remove:
        for sec in cell.dendlist:
            if sec.name() == target:
                target_list.append(sec)
    
    objects_to_remove_set = set()
    for branch in dend_tree:
        for path in branch:
            if not isinstance(path, list):
                path = [path]  # Ensure path is always a list
            for target in target_list:
                if target in path:
                    # Find the index of the target
                    start_index = path.index(target)
                    # If not at the end, collect the target and all subsequent objects
                    if start_index < len(path) - 1:
                        objects_to_remove_set.update(path[start_index:])
                    else:
                        # If it's a terminal branch, simply remove it
                        objects_to_remove_set.add(target)

    # Convert the set to a list if needed
    dends2remove = list(objects_to_remove_set)
    return dends2remove

def soma_resize(cell, diam):
    """
    Resize the soma of the cell to a new diameter and reconnect initial connections.
    
    Parameters:
    cell : NEURON cell object
        The cell whose soma is to be resized.
    diam : float
        The new diameter for the soma.
    """
    
    # Step 1: Record the initial connections to the soma
    initial_connections = []
    for sec in cell.dendlist:
        if sec.parentseg() is not None and sec.parentseg().sec == cell.soma:
            initial_connections.append(sec)
    for sec in cell.axonlist:
        if sec.parentseg() is not None and sec.parentseg().sec == cell.soma:
            initial_connections.append(sec)

    # Step 2: Update soma dimensions
    radius = diam / 2
    for sec in cell.somalist:
        h('access ' + sec.name())
        h.pt3dchange(0, -radius, 0.0, 0.0, diam)  # Leftmost point
        h.pt3dchange(1, 0.0, 0.0, 0.0, diam)      # Center point
        h.pt3dchange(2, radius, 0.0, 0.0, diam)   # Rightmost point

    # Step 3: Disconnect only those sections initially connected to the soma
    for sec in initial_connections:
        sec.disconnect()

    # Step 4: Reconnect only sections in `initial_connections` to the end of the soma
    for sec in initial_connections:
        sec.connect(cell.soma(1))  # Connect to the end of the soma section (position 1)
    
    print(f"soma diameter: {round(cell.soma.diam)} μm")

# this function forces nseg to match that obtained AFTER record_all_3D is run
# this issue caused by this line inside record_all_3D():sec.nseg = int(h.n3d())
# in essence the model is reparameterised and more accurate after 3D
def set_nseg_from_n3d(cell, make_odd=True, include_soma=False, include_axon=False):
    def _n3d(sec):
        h('access ' + sec.name())
        return int(h.n3d())

    changed = {}
    if include_soma:
        for sec in cell.somalist:
            n = _n3d(sec)
            if make_odd and n % 2 == 0: n += 1
            if sec.nseg != n:
                changed[sec.name()] = (sec.nseg, n)
                sec.nseg = n

    if include_axon and hasattr(cell, 'axonlist'):
        for sec in cell.axonlist:
            n = _n3d(sec)
            if make_odd and n % 2 == 0: n += 1
            if sec.nseg != n:
                changed[sec.name()] = (sec.nseg, n)
                sec.nseg = n

    for sec in cell.dendlist:
        n = _n3d(sec)
        if make_odd and n % 2 == 0: n += 1
        if sec.nseg != n:
            changed[sec.name()] = (sec.nseg, n)
            sec.nseg = n
    return changed

# SimClamp 
def msNEURONsim(
            sim_time, 
            stim_time,
            baseline,
            glut, 
            glut_frequency, 
            glutamate_locations, 
            glut_reversals,
            glut_time, 
            num_gluts, 
            dend_glut, 
            g_AMPA,
            ratio,
            gaba, 
            gaba_frequency, 
            gaba_locations,
            gaba_reversals,
            gaba_time, 
            g_GABA, 
            num_gabas, 
            dend_gaba, 
            specs, 
            scale_factors=None,
            gaba_tau1=0.9,
            gaba_tau2=18,
            rel_gaba_onsets=None,
            rel_glut_onsets=None,
            frequency=2000,
            d_lambda=0.05,
            dend2remove=None,
            v_init=-84, 
            AMPA=True,
            NMDA=True,
            physiological=True,
            timing_range=None, 
            add_noise=False,
            beta=0,                    
            B=1,                       
            add_sine=False, 
            axoshaft=False,
            cell_type='dspn',
            current_step=False,
            step_current=-200,
            step_duration=500,
            step_start = 300,
            holding_current=0,
            add_ramp=False,
            ramp_amplitude = 200,
            Cm=1,
            Ra=200,
            spine_per_length=1.61,
            spine_neck_diam=0.1,
            spine_neck_len=1,
            spine_head_diam=0.5,
            soma_diameter=None,
            neck_dynamics=False,
            space_clamp=False,
            record_dendrite=None, 
            record_location=None, 
            record_currents=False,     
            record_branch=False,       
            dend_glut2=None,
            record_mechs=False,
            mechs2record=None,
            record_path_dend=False,    
            record_path_spines=False, 
            record_all_spines=False, 
            record_all_path=True,      
            record_3D=False,           
            record_3D_impedance=False, 
            freq=10,                   
            record_3D_mechs=False,     
            record_Ca=False,
            record_3D_Ca=False,
            tonic=False,
            gbar_gaba=None,
            rectification=False,       
            distributed=False,         
            gaba_params=None,
            tonic_gaba_reversal=-60,
            dt=0.025,
            ds_imp=40,
            voltage_clamp=False,
            holding_potential=-84,
            voltage_clamp_site = 'soma[0]',
            voltage_clamp_spine=False,
            voltage_clamp_loc=0.5,    
            Rs=0,
            downsample=True,
            ds=10
            ):

    # 1. initialize variables
    i_recording_site = []; v_recording_site = []; 
    vspine = []; vdend = []; vsoma = []; zdend = []; ztransfer = []
    v_dend_tree = {}; v_spine_tree = {}; i_mechs_dend = {}; i_mechs_dend_tree = {};
    v_spine_activated = {}; v_dend_activated = {};
    i_mechs_spine_tree = {}; i_mechs_spine_tree = {}; v_branch = {} 
    v_all_3D = {}; imp_all_3D = {}; Ca_all_3D = {}; i_mechs_3D = {}
    Ca_spine = []; Ca_dend = []; Ca_soma = []

    start_time = min(stim_time, *timing_range)
    burn_time = start_time - baseline  # time allowed to reach clamped potential

    # pA to nA for NEURON
    # if current_step: cs = np.array(step_current, ndmin=1) / 1e3  # cs = step_current/1e3       
    hc = holding_current/1e3

    if add_ramp: ramp_amplitude = ramp_amplitude/1e3  
        
    # 2. build cell 
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, dend2remove=dend2remove, neck_dynamics=neck_dynamics)

    # ensure 3D discretisation for all runs
    _ = set_nseg_from_n3d(cell, make_odd=False, include_soma=False, include_axon=False)

    # 3. set up tonic gaba
    if tonic:
        if distributed:
            tonic_gaba(cell=cell, gaba_reversal=tonic_gaba_reversal, gbar_gaba=gbar_gaba, d3=gaba_params['d3'], a4=gaba_params['a4'], a5=gaba_params['a5'], a6=gaba_params['a6'], a7=gaba_params['a7'], rectification=rectification)                
        else:
            tonic_gaba(cell=cell, gaba_reversal=tonic_gaba_reversal, gbar_gaba=gbar_gaba, rectification=rectification)

    # 4. change properties

    if Ra != 200:
        space_clamped(cell=cell, spines=spines, Ra=Ra)
        print(f"Ra is {Ra} Ωcm")
    
    if Cm != 1:
        cap(cell=cell, spines=spines, cm = Cm)
        print(f"Cm is {Cm} μFcm\u207B\u00B2")
  
    if scale_factors is not None:
        printed = set()  # track which mechanisms have already been reported
        for sec in h.allsec():
            for seg in sec:
                for mech, sf in scale_factors.items():
                    if hasattr(seg, mech):
                        getattr(seg, mech).sf = sf
                        if mech not in printed:
                            if mech in ['cal12', 'cal13', 'can', 'car', 'cav32', 'cav33']:
                                print(f"permeability (cm/s) scaled: '{mech}' scaling factor = {sf}")
                            else:
                                print(f"conductance (S/cm²) scaled: '{mech}' scaling factor = {sf}")
                            printed.add(mech)

    # change all spine neck diameters
    if spine_neck_diam != 0.1:
        spine_neck_diameter(cell=cell, spines=spines, diam=spine_neck_diam)
        print(f"spine neck diameter {spine_neck_diam} μm")
        
    # change all spine neck lengths
    if spine_neck_len != 1:
        spine_neck_length(cell=cell, spines=spines, length=spine_neck_len)
        print(f"spine neck length {spine_neck_len} μm")
        
    # change all spine head diameters
    if spine_head_diam != 0.5:
        spine_head_diameter(cell=cell, spines=spines, diam=spine_head_diam, length=spine_head_diam)
        print(f"spine_head_diameter {spine_head_diam} μm")

    capacitance = whole_cell_capacitance(cell, spines, Cm=Cm)
    
    # 5. prepare synaptic inputs
    if glut_frequency is not None:
        dstep1 = int(1 / glut_frequency * 1e3)
    else:
        dstep1 = 0

    if gaba_frequency is not None:
        dstep2 = int(1 / gaba_frequency * 1e3)

    glut_secs = [sec for target_dend in dend_glut for sec in cell.dendlist if sec.name() == target_dend]
    if num_gluts > 1 and len(glut_secs) == 1:
        glut_secs = glut_secs * num_gluts

    if gaba:
        gaba_secs = [sec for target_dend in dend_gaba for sec in cell.allseclist if sec.name() == target_dend]
    else:
        gaba_secs = []

    if num_gabas > 1 and len(gaba_secs) == 1:
        gaba_secs = gaba_secs * num_gabas

    # phasic glut
    if rel_glut_onsets:
        glut_onsets = [glut_time + x for x in rel_glut_onsets]
    else:
        if dstep1 > 0:
            glut_onsets = list(range(glut_time, glut_time + num_gluts * dstep1, dstep1))
        else:
            glut_onsets = [glut_time] * num_gluts  

    # phasic gaba
    # if rel_gaba_onsets is not None and rel_gaba_onsets.size > 0: # rel_gaba_onsets is numpy array

    if rel_gaba_onsets is not None and (
        (isinstance(rel_gaba_onsets, list) and len(rel_gaba_onsets) > 0) or
        (isinstance(rel_gaba_onsets, np.ndarray) and rel_gaba_onsets.size > 0)
    ):        
        gaba_onsets = [x + gaba_time for x in rel_gaba_onsets]
    else:
        if gaba_frequency is None:
            gaba_onsets = [gaba_time] * len(gaba_secs)
        else:
            gaba_onsets = list(range(gaba_time, gaba_time + num_gabas * dstep2, dstep2)) * len(gaba_secs)

    # 6a. place glutamatergic synapses
    if num_gluts > 1 and len(glutamate_locations) == 1:
        glutamate_locations = glutamate_locations*num_gluts

    glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs, final_spine_secs = glut_place4(
        cell = cell,
        spines = spines,
        physiological = physiological, 
        AMPA = AMPA, 
        g_AMPA = g_AMPA,
        NMDA = NMDA,
        ratio = ratio,
        glut_reversals = glut_reversals,
        glut = glut,
        glut_time = glut_time,
        glut_secs = glut_secs,
        glut_onsets = glut_onsets,
        glut_locs = glutamate_locations,
        num_gluts = num_gluts,
        return_currents = record_currents,
        axoshaft = axoshaft
    )
    glut_locs = []
    for spine in final_spines:
        glut_locs.append(spine.x)

    # 6b. place gabaergic synapses
    if num_gabas > 1 and len(gaba_locations) == 1:
        gaba_locations = gaba_locations*num_gabas

    gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances, gaba_locs = gaba_place3(
        physiological = physiological,
        gaba_reversals = gaba_reversals,
        gaba_weight = g_GABA,
        gaba_tau1=gaba_tau1,
        gaba_tau2=gaba_tau2,
        gaba_time = gaba_time,
        gaba_secs = gaba_secs,
        gaba_onsets = gaba_onsets,
        gaba_locations = gaba_locations,
        num_gabas = num_gabas,
        return_currents = record_currents
    )   
                            
    if not voltage_clamp:
        # 7. setup recording locations
        if record_location is None:
            if glut:
                # loc = sum(glut_locs) / len(glut_locs) # midpoint
                loc = glutamate_locations[0] # first location
            else:
                # loc = sum(gaba_locations) / len(gaba_locations) # midpoint
                loc = gaba_locations[0] # first location
        else:
            loc = record_location[0]
        spine_dist = []
        if record_dendrite is None: # will assume want 1st listed dendrite
            dendrite = glut_secs[0] if glut else glut_secs_orig[0] if gaba else None
            spine = final_spines[0] if glut else glut_secs_orig[0] if gaba else None
            spine_dist = h.distance(dendrite(glut_locs[0])) 
        else:
            for sec in cell.allseclist:
                if sec.name() == record_dendrite:
                    dendrite = sec
            # only record from spine head IF only 1 glutamatergic input
            if num_gluts == 1:
                spine = final_spines[0]
                spine_dist = h.distance(dendrite(glut_locs[0]))
            else: 
                spine = None

        print('recording at {} with location {}'.format(dendrite, round(loc,4)))

        # 8. configure basic current clamp
        t = h.Vector().record(h._ref_t)
        iclamp1 = h.IClamp(cell.soma(0.5))
        iclamp1.dur = sim_time
        if add_noise or add_sine or add_ramp:
            iclamp1.amp = 0 # nA
        else:
            iclamp1.amp = hc # nA

        # 9. add further types of stimulus
        # add coloured noise
        if add_noise:
            samples = int(sim_time/dt)  # number of samples to generate (time series extension)
            noise = B * cn.powerlaw_psd_gaussian(beta, samples) + hc
            noise_vector = h.Vector()
            noise_vector.from_python(noise)
            tvec = h.Vector(np.linspace(dt, sim_time, int(sim_time/dt)))
            noise_vector.play(iclamp1._ref_amp, tvec, True)

        # add sine wave
        if add_sine:
            time = np.linspace(dt, sim_time, int(sim_time/dt))
            tvec = h.Vector(time)
            sine_wave = amplitude/1e3 * np.sin(2*np.pi*frequency*time/1e3)  + hc
            sine_vector = h.Vector()
            sine_vector.from_python(sine_wave)
            sine_vector.play(iclamp1._ref_amp, tvec, True)

        # # add current step
        # if current_step:
        #     step_end = step_start + step_duration
        #     iclamp2 = h.IClamp(cell.soma(0.5))
        #     iclamp2.delay = step_start
        #     iclamp2.dur = step_duration
        #     iclamp2.amp = cs # nA
        #     print(f"step clamp parameters: {iclamp2.delay} ms, {iclamp2.dur} ms, {iclamp2.amp*1e3:.2f} pA")

        # add at least one current step
        if current_step:
            cs = np.array(step_current, ndmin=1) / 1e3  # convert to nA
            dur = np.array(step_duration, ndmin=1)
            ts = np.array(step_start, ndmin=1)
        
            n_steps = len(cs)
            iclamps = []
        
            for i in range(n_steps):
                iclamp = h.IClamp(cell.soma(0.5))
                iclamp.delay = float(ts[i])
                iclamp.dur   = float(dur[i])
                iclamp.amp   = float(cs[i])  # << use scaled current here
                iclamps.append(iclamp)
                print(f"Step {i+1}: {iclamp.delay}–{iclamp.delay+iclamp.dur} ms, {iclamp.amp*1e3:.1f} pA")
        

        # create ramp
        if add_ramp:
            time = np.linspace(dt, sim_time, int(sim_time / dt))
            tvec = h.Vector(time)
            ramp_wave = np.full_like(time, hc)  # holding current always on
        
            stim_index = int(stim_time / dt)
            ramp_wave[stim_index:] += ramp_amplitude * ((time[stim_index:] - stim_time) / (sim_time - stim_time))
        
            ramp_vector = h.Vector()
            ramp_vector.from_python(ramp_wave)
            ramp_vector.play(iclamp1._ref_amp, tvec, True)
    
    else:
        # 7. setup recording locations
        loc = voltage_clamp_loc
        if voltage_clamp_spine:
            spine = final_spines[0]
            section = spine.head
            loc = 0.5
        else:
            for sec in cell.allseclist:
                if sec.name() == voltage_clamp_site:
                    section = sec

        print('voltage clamp at {} with location {}'.format(section, round(loc,4)))

        # 8. setup voltage recording site for synaptic location (for voltage breakthrough)
        if glut:
            loc1 = glutamate_locations[0] # first location
        else:
            loc1 = gaba_locations[0] # first location

        spine_dist = []
        if record_dendrite is None: # will assume want 1st listed dendrite
            dendrite = glut_secs[0] if glut else glut_secs_orig[0] if gaba else None
            spine = final_spines[0] if glut else glut_secs_orig[0] if gaba else None
            spine_dist = h.distance(dendrite(glut_locs[0])) 
        else:
            for sec in cell.allseclist:
                if sec.name() == record_dendrite:
                    dendrite = sec
            # only record from spine head IF only 1 glutamatergic input
            if num_gluts == 1:
                spine = final_spines[0]
                spine_dist = h.distance(dendrite(glut_locs[0]))
            else: 
                spine = None

        # 9. configure basic voltage clamp

        t = h.Vector().record(h._ref_t)
        vclamp = h.SEClamp(section(loc))
        vclamp.dur1 = sim_time
        vclamp.amp1 = holding_potential # mV
        vclamp.rs = Rs # MOhm
        i_recording_site = h.Vector().record(vclamp._ref_i) 
    
    # 10. set up vectors
    vsoma = h.Vector()
    vdend = h.Vector()
    
    if voltage_clamp:
        v_recording_site = h.Vector()
        v_recording_site.record(section(loc)._ref_v)
        vdend.record(dendrite(loc1)._ref_v)
    else:
        vdend.record(dendrite(loc)._ref_v)
    vsoma.record(cell.soma(0.5)._ref_v)

    if num_gluts == 1:
        record_spine = True
        vspine = h.Vector()
        vspine.record(spine.head(0.5)._ref_v)
    else:
        record_spine = False 

    v_spine_activated, v_dend_activated, dists_spine_activated = record_all_activated_spine_v2(cell=cell, dendrite=dendrite, activated_spines=final_spines) 
    v_spine_activated = {
            'v': v_spine_activated,
            'dists': dists_spine_activated,
            'dendrites': final_spine_secs,
            'locs': final_spine_locs
        }  
    v_dend_activated = {
            'v': v_dend_activated,
            'dists': dists_spine_activated
        }
    if record_Ca:
        Ca_soma = h.Vector(); Ca_dend = h.Vector()
        Ca_soma.record(cell.soma(0.5)._ref_cai)        
        Ca_dend.record(dendrite(loc)._ref_cai)
        
        if record_spine:
            Ca_spine = h.Vector()
            Ca_spine.record(spine.head(0.5)._ref_cai)

    if record_path_dend:
        # all voltages in path in dendrite
        v_dend_tree, dists_tree, dends_v = record_all_path_secs_v2(cell=cell, dend_tree=dend_tree, dendrite=dendrite)
        v_dend_tree = {
            'v': v_dend_tree,
            'dists': dists_tree,
            'dendrites': dends_v
        }    

    if record_all_spines:
        # all voltages in every spine head across all dendrites
        _v, _dists, _dends, _locs = record_all_secs_spines(spines=spines)
        v_spine_tree = {
            'v': _v,
            'dists': _dists,
            'dendrites': _dends,
            'locs': _locs
        }
    elif record_path_spines:
        # all voltages in path in a spine
        v_spine_tree, dists_tree, dends_spine, locs_spine = record_all_path_secs_spine_v2(cell=cell, spines=spines, dend_tree=dend_tree, dendrite=dendrite, activated_spines=final_spines)
        v_spine_tree = {
            'v': v_spine_tree,
            'dists': dists_tree,
            'dendrites': dends_spine,
            'locs': locs_spine
        }

    if record_mechs:
        # i_mechs at recording site
        mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kcnq', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']
        i_mechs_dend = record_i_mechs(cell=cell, dend=dendrite, loc=loc, return_currents=record_currents, mechs=mechs)
        i_mechs_dend ={
            'i': i_mechs_dend,
            'mechs': mechs
        }
        if record_path_dend:
            # all i_mechs in path in dendrite
            i_mechs_dend_tree, dists_tree, dends_i = record_all_path_secs_i_mechs(cell=cell, dend_tree=dend_tree, dendrite=dendrite, mechs=mechs)
            i_mechs_dend_tree = {
                'i': i_mechs_dend_tree,
                'dists': dists_tree,
                'dendrites': dends_i,
                'mechs': mechs
            } 

        if record_path_spines:
            # all i_mechs in path in spines
            spine_mechs = ['pas', 'kir', 'cal12', 'cal13', 'car', 'cav32', 'cav33', 'sk']
            i_mechs_spine_tree, dists_tree, dends_i = record_all_path_secs_spine_i_mechs(cell=cell, spines=spines, dend_tree=dend_tree, dendrite=dendrite, activated_spines=final_spines, spine_mechs=spine_mechs)
            i_mechs_spine_tree = {
                'i': i_mechs_spine_tree,
                'dists': dists_tree,
                'dendrites': dends_i,
                'mechs': spine_mechs
            }
    
    if record_branch:
        for dend_name in dend_glut2:
            dendrite_branch = None
            # Find the dendrite in the cell's section list
            for dend in cell.allseclist:
                if dend.name() == dend_name:
                    dendrite_branch = dend
                    break
            # Collect the data
            v, _, _, dists = sec_all_v(section=dendrite_branch, all_v={}, i=0)
            v_branch[dend_name] = {
                'v': v,
                'dists': dists
            }

    # print('\n---- nseg before record_all_3D ----')
    # for sec in h.allsec():
    #     print(f"{sec.name():25s}  nseg = {sec.nseg}")

    
    # if want 3D heatmaps
    if record_3D:
        # all voltages at 3D coordinates
        all_v, cell_coordinates, dends3D, dists3D = record_all_3D(cell)
        v_all_3D = {
            'v': all_v,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists3D,
            'dendrites': dends3D
        }   

    # print('\n---- nseg after record_all_3D ----')
    # for sec in h.allsec():
    #     print(f"{sec.name():25s}  nseg = {sec.nseg}")
    
    # if want 3D heatmaps
    if record_3D_Ca:
        # all voltages at 3D coordinates
        all_Ca, cell_coordinates, dends3D, dists3D = record_Ca_3D(cell)
        Ca_all_3D = {
            'Ca': all_Ca,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists3D,
            'dendrites': dends3D
        }   

    if record_3D_mechs:
        # i_mechs at recording site
        if mechs2record == None:
            # mechs3D = ['cal12', 'cal13', 'can', 'car', 'cav32', 'cav33']
            mechs3D = ['pas', 'kdr', 'naf', 'kaf', 'kas', 'kcnq', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']
        else:
            mechs3D = mechs2record
        all_i_mechs, cell_coordinates, dends3D, dists3D = record_mechs_3D(cell, mechs=mechs3D)
        i_mechs_3D ={
            'i': all_i_mechs,
            'cell_coordinates': cell_coordinates,
            'dists': dists3D,
            'dendrites': dends3D,
            'mechs': mechs3D
        }        

    # 11. run simulation
    # try:
    #     h.cvode.use_fast_imem(1)
    # except:`1
    #     pass
    
    pc = h.ParallelContext()
    _nt = int(pc.nthread())
    
    # set threads BEFORE initialize
    if record_3D_impedance:
        pc.nthread(1)
    
    h.dt = dt
    h.finitialize(v_init)
    
    if record_3D_impedance:
        impedance_locations, cell_coordinates, dends, dists = setup_impedance_measurements(cell)
    #         impedance_transfer_locations, _, _, _ = setup_impedance_measurements(cell)
    
        # Initialize a dictionary to store impedance vectors for each location
        impedance_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_phase_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_transfer_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_transfer_phase_vectors = {loc: h.Vector() for loc in impedance_locations}
    
        if record_spine:
            imp_spine = h.Vector()
            imp_spine_phase = h.Vector()
        
        # Initialize Impedance object
        imp = h.Impedance()
        imp.loc(loc, sec=dendrite) # location of interest, impedance transfer is relative to this point 
        
        # Initialize a variable to track the next time to compute impedance
        next_impedance_time = 0 + burn_time # only start calculating impedance when burn_time is over
        # ds_imp = 40 # downsample to ds_imp * h.dt
        
        # Simulation loop
        while h.t < sim_time:
            if h.t >= round(next_impedance_time, 3):
                # Compute impedance at the specified frequency
                # second argument 1 calculates extended impedance; takes into account differential gating variables
                # of all membrane mechanisms 
                imp.compute(freq, 1)
    
                if num_gluts == 1:
                    # Record input impedance magnitude and phase at the spine head
                    imp_spine.append(imp.input(0.5, sec=spine.head))
                    imp_spine_phase.append(imp.input_phase(0.5, sec=spine.head))  # Phase in radians
    
                # Record impedance magnitude and phase at each location
                for (sec, loc) in impedance_locations:
                    # Append the impedance value to the corresponding vector
                    impedance_vectors[(sec, loc)].append(imp.input(loc, sec=sec))
                    impedance_phase_vectors[(sec, loc)].append(imp.input_phase(loc, sec=sec))  # Phase in radians
    
                    # Record transfer impedance magnitude and phase
                    impedance_transfer_vectors[(sec, loc)].append(imp.transfer(loc, sec=sec))
                    impedance_transfer_phase_vectors[(sec, loc)].append(imp.transfer_phase(loc, sec=sec))  # Transfer phase
    
                # Schedule the next impedance computation
                next_impedance_time += dt * ds_imp
    
            # Advance the simulation by one time step
            h.fadvance()
           
        imp_all_3D = {
            'imp': list(impedance_vectors.values()),
            'imp phase': list(impedance_phase_vectors.values()),
            'imp transfer': list(impedance_transfer_vectors.values()),
            'imp transfer phase': list(impedance_transfer_phase_vectors.values()),
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists,
            'dendrites': dends
        }
        pc.nthread(_nt)
    
    else:
        while h.t < sim_time:
            h.fadvance()

    record_dist = h.distance(dendrite(loc))
    
    # make outputs as numpy arrays
    if record_spine:
        vspine = np.array(vspine)
        if record_3D_impedance:
            imp_spine=np.array(imp_spine)
    
    if voltage_clamp:
        i_recording_site = np.array(i_recording_site); v_recording_site = np.array(v_recording_site)
    
    vdend = np.array(vdend); vsoma = np.array(vsoma)
    
    if record_Ca:  
        Ca_dend = np.array(Ca_dend); Ca_soma = np.array(Ca_soma)
        if record_spine:
            Ca_spine = np.array(Ca_spine)
            
    v_spine_activated['v'] = vec2np(v_spine_activated['v'])
    v_dend_activated['v'] = vec2np(v_dend_activated['v'])
    
    if record_path_dend:
        v_dend_tree['v'] = vec2np(v_dend_tree['v'])
    if record_all_spines or record_path_spines:
        v_spine_tree['v'] = vec2np(v_spine_tree['v'])
    if record_mechs:
        i_mechs_dend['i'] = vec2np(i_mechs_dend['i'])

        if record_path_dend:        
            out = {}
            for name, data in i_mechs_dend_tree['i'].items():
                # initialize a list to hold NumPy arrays for each dendrite
                out[name] = vec2np(data)
            i_mechs_dend_tree['i'] = out

        if record_path_spines:    
            out = {}
            for name, data in i_mechs_spine_tree['i'].items():
                out[name] = vec2np(data)
            i_mechs_spine_tree['i'] = out

    
    if record_branch:
        # Convert NEURON Vectors in v_branch to NumPy arrays
        for dend_name, data in v_branch.items():
            # Initialize a list to hold NumPy arrays for the current dendrite
            v_branch_np_arrays = []

            # Iterate through each Vector in 'v' and convert to NumPy array
            for i, vector in data['v'].items():
                np_array = np.array(vector)
                v_branch_np_arrays.append(np_array)

            # Replace the original 'v' data with the list of NumPy arrays
            v_branch[dend_name]['v'] = vec2np(v_branch[dend_name]['v'])        
    
    if record_3D:
        v_all_3D['v'] = vec2np(v_all_3D['v'])
    
    if record_3D_Ca:
        Ca_all_3D['Ca'] = vec2np(Ca_all_3D['Ca'])

    if record_3D_impedance:
        imp_all_3D['imp'] = vec2np(imp_all_3D['imp'])
        imp_all_3D['imp phase'] = vec2np(imp_all_3D['imp phase'])
        imp_all_3D['imp transfer'] = vec2np(imp_all_3D['imp transfer'])
        imp_all_3D['imp transfer phase'] = vec2np(imp_all_3D['imp transfer phase'])
        if record_spine:
            imp_all_3D['imp spine'] = imp_spine

    if record_3D_mechs:        
        out = {}
        for name, data in i_mechs_3D['i'].items():
            # initialize a list to hold NumPy arrays for each dendrite
            out[name] = vec2np(data)
        i_mechs_3D['i'] = out
    
    if record_currents:
        ampa_currents = vec2np(ampa_currents)
        nmda_currents = vec2np(nmda_currents)
        gaba_currents = vec2np(gaba_currents)
        gaba_conductances = vec2np(gaba_conductances)

    if downsample:
        def ds_array(x, ds):
            if isinstance(x, np.ndarray) and x.ndim > 0:
                return x[::ds]
            elif isinstance(x, list) and len(x) > 0:
                # Handle lists of arrays (e.g., nmda_currents, ampa_currents, gaba_currents)
                result = []
                for arr in x:
                    if isinstance(arr, np.ndarray) and arr.ndim > 0:
                        result.append(arr[::ds])
                    else:
                        # Keep 0-dimensional arrays or non-arrays unchanged
                        result.append(arr)
                return result
            else:
                return x
        
        # Downsample simple arrays
        i_recording_site = ds_array(i_recording_site, ds)
        v_recording_site = ds_array(v_recording_site, ds)
        vspine = ds_array(vspine, ds)
        vdend = ds_array(vdend, ds)
        vsoma = ds_array(vsoma, ds)
        Ca_spine = ds_array(Ca_spine, ds)
        Ca_dend = ds_array(Ca_dend, ds)
        Ca_soma = ds_array(Ca_soma, ds)

        ampa_currents = ds_array(ampa_currents, ds)
        nmda_currents = ds_array(nmda_currents, ds)
        gaba_currents = ds_array(gaba_currents, ds)
        gaba_conductances = ds_array(gaba_conductances, ds)


        # Dictionary-style outputs
        def ds_dict(d, ds):
            if not isinstance(d, dict):
                return d
            
            for k, v in d.items():
                if isinstance(v, np.ndarray) and v.ndim > 0:
                    d[k] = v[::ds]
                elif isinstance(v, dict):
                    ds_dict(v, ds)  # Recursive call for nested dicts
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.ndarray):
                    d[k] = [arr[::ds] if arr.ndim > 0 else arr for arr in v]
            
            return d
        
        # Apply to dictionary variables
        v_all_3D = ds_dict(v_all_3D, ds)
        Ca_all_3D = ds_dict(Ca_all_3D, ds)

        i_mechs_3D = ds_dict(i_mechs_3D, ds)
        v_dend_tree = ds_dict(v_dend_tree, ds)
        v_spine_tree = ds_dict(v_spine_tree, ds)
        v_dend_activated = ds_dict(v_dend_activated, ds)
        v_spine_activated = ds_dict(v_spine_activated, ds)
        i_mechs_dend = ds_dict(i_mechs_dend, ds)
        i_mechs_dend_tree = ds_dict(i_mechs_dend_tree, ds)
        i_mechs_spine_tree = ds_dict(i_mechs_spine_tree, ds)
        v_branch = ds_dict(v_branch, ds)
    

    return i_recording_site, v_recording_site, v_all_3D, Ca_all_3D, imp_all_3D, i_mechs_3D, vspine, v_spine_activated, vdend, v_dend_activated, vsoma, v_dend_tree, v_spine_tree, Ca_spine, Ca_dend, Ca_soma, i_mechs_dend, i_mechs_dend_tree, i_mechs_spine_tree, v_branch, zdend, ztransfer, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, record_dist, record_spine, spine_dist, capacitance, h.dt, burn_time, start_time


# SimClamp 
def SimClamp(
            sim_time, 
            stim_time,
            baseline,
            glut, 
            glut_frequency, 
            glutamate_locations, 
            glut_reversals,
            glut_time, 
            num_gluts, 
            dend_glut, 
            g_AMPA,
            ratio,
            gaba, 
            gaba_frequency, 
            gaba_locations,
            gaba_reversals,
            gaba_time, 
            g_GABA, 
            num_gabas, 
            dend_gaba, 
            specs, 
            scale_factors=None,
            gaba_tau1=0.9,
            gaba_tau2=18,
            rel_gaba_onsets=None,
            rel_glut_onsets=None,
            frequency=2000,
            d_lambda=0.05,
            dend2remove=None,
            v_init=-84, 
            AMPA=True,
            NMDA=True,
            physiological=True,
            timing_range=None, 
            add_noise=False,
            beta=0,                    
            B=1,                       
            add_sine=False, 
            axoshaft=False,
            cell_type='dspn',
            current_step=False,
            step_current=-200,
            step_duration=500,
            step_start = 300,
            holding_current=0,
            add_ramp=False,
            ramp_amplitude = 200,
            Cm=1,
            Ra=200,
            spine_per_length=1.61,
            spine_neck_diam=0.1,
            spine_neck_len=1,
            spine_head_diam=0.5,
            soma_diameter=None,
            neck_dynamics=False,
            space_clamp=False,
            record_dendrite=None, 
            record_location=None, 
            record_currents=False,     
            record_branch=False,       
            dend_glut2=None,
            record_mechs=False,
            mechs2record=None,
            record_path_dend=False,    
            record_path_spines=False,  
            record_all_path=True,      
            record_3D=False,           
            record_3D_impedance=False, 
            freq=10,                   
            record_3D_mechs=False,     
            record_Ca=False,
            record_3D_Ca=False,
            tonic=False,
            gbar_gaba=None,
            rectification=False,       
            distributed=False,         
            gaba_params=None,
            tonic_gaba_reversal=-60,
            dt=0.025,
            ds_imp=40,
            voltage_clamp=False,
            holding_potential=-84,
            voltage_clamp_site = 'soma[0]',
            voltage_clamp_spine=False,
            voltage_clamp_loc=0.5,    
            Rs=0
            ):

    # 1. initialize variables
    i_recording_site = []; v_recording_site = []; 
    vspine = []; vdend = []; vsoma = []; zdend = []; ztransfer = []
    v_dend_tree = {}; v_spine_tree = {}; i_mechs_dend = {}; i_mechs_dend_tree = {};
    v_spine_activated = {}; v_dend_activated = {};
    i_mechs_spine_tree = {}; i_mechs_spine_tree = {}; v_branch = {} 
    v_all_3D = {}; imp_all_3D = {}; Ca_all_3D = {}; i_mechs_3D = {}
    Ca_spine = []; Ca_dend = []; Ca_soma = []

    start_time = min(stim_time, *timing_range)
    burn_time = start_time - baseline  # time allowed to reach clamped potential

    # pA to nA for NEURON
    # if current_step: cs = np.array(step_current, ndmin=1) / 1e3  # cs = step_current/1e3       
    hc = holding_current/1e3

    if add_ramp: ramp_amplitude = ramp_amplitude/1e3  
        
    # 2. build cell 
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, dend2remove=dend2remove, neck_dynamics=neck_dynamics)

    # ensure 3D discretisation for all runs
    _ = set_nseg_from_n3d(cell, make_odd=False, include_soma=False, include_axon=False)

    # 3. set up tonic gaba
    if tonic:
        if distributed:
            tonic_gaba(cell=cell, gaba_reversal=tonic_gaba_reversal, gbar_gaba=gbar_gaba, d3=gaba_params['d3'], a4=gaba_params['a4'], a5=gaba_params['a5'], a6=gaba_params['a6'], a7=gaba_params['a7'], rectification=rectification)                
        else:
            tonic_gaba(cell=cell, gaba_reversal=tonic_gaba_reversal, gbar_gaba=gbar_gaba, rectification=rectification)

    # 4. change properties

    if Ra != 200:
        space_clamped(cell=cell, spines=spines, Ra=Ra)
        print(f"Ra is {Ra} Ωcm")
    
    if Cm != 1:
        cap(cell=cell, spines=spines, cm = Cm)
        print(f"Cm is {Cm} μFcm\u207B\u00B2")
  
    if scale_factors is not None:
        printed = set()  # track which mechanisms have already been reported
        for sec in h.allsec():
            for seg in sec:
                for mech, sf in scale_factors.items():
                    if hasattr(seg, mech):
                        getattr(seg, mech).sf = sf
                        if mech not in printed:
                            if mech in ['cal12', 'cal13', 'can', 'car', 'cav32', 'cav33']:
                                print(f"permeability (cm/s) scaled: '{mech}' scaling factor = {sf}")
                            else:
                                print(f"conductance (S/cm²) scaled: '{mech}' scaling factor = {sf}")
                            printed.add(mech)

    # change all spine neck diameters
    if spine_neck_diam != 0.1:
        spine_neck_diameter(cell=cell, spines=spines, diam=spine_neck_diam)
        print(f"spine neck diameter {spine_neck_diam} μm")
        
    # change all spine neck lengths
    if spine_neck_len != 1:
        spine_neck_length(cell=cell, spines=spines, length=spine_neck_len)
        print(f"spine neck length {spine_neck_len} μm")
        
    # change all spine head diameters
    if spine_head_diam != 0.5:
        spine_head_diameter(cell=cell, spines=spines, diam=spine_head_diam, length=spine_head_diam)
        print(f"spine_head_diameter {spine_head_diam} μm")

    capacitance = whole_cell_capacitance(cell, spines, Cm=Cm)
    
    # 5. prepare synaptic inputs
    if glut_frequency is not None:
        dstep1 = int(1 / glut_frequency * 1e3)
    else:
        dstep1 = 0

    if gaba_frequency is not None:
        dstep2 = int(1 / gaba_frequency * 1e3)

    glut_secs = [sec for target_dend in dend_glut for sec in cell.dendlist if sec.name() == target_dend]
    if num_gluts > 1 and len(glut_secs) == 1:
        glut_secs = glut_secs * num_gluts

    if gaba:
        gaba_secs = [sec for target_dend in dend_gaba for sec in cell.allseclist if sec.name() == target_dend]
    else:
        gaba_secs = []

    if num_gabas > 1 and len(gaba_secs) == 1:
        gaba_secs = gaba_secs * num_gabas

    # phasic glut
    if rel_glut_onsets:
        glut_onsets = [glut_time + x for x in rel_glut_onsets]
    else:
        if dstep1 > 0:
            glut_onsets = list(range(glut_time, glut_time + num_gluts * dstep1, dstep1))
        else:
            glut_onsets = [glut_time] * num_gluts  

    # phasic gaba
    # if rel_gaba_onsets is not None and rel_gaba_onsets.size > 0: # rel_gaba_onsets is numpy array

    if rel_gaba_onsets is not None and (
        (isinstance(rel_gaba_onsets, list) and len(rel_gaba_onsets) > 0) or
        (isinstance(rel_gaba_onsets, np.ndarray) and rel_gaba_onsets.size > 0)
    ):        
        gaba_onsets = [x + gaba_time for x in rel_gaba_onsets]
    else:
        if gaba_frequency is None:
            gaba_onsets = [gaba_time] * len(gaba_secs)
        else:
            gaba_onsets = list(range(gaba_time, gaba_time + num_gabas * dstep2, dstep2)) * len(gaba_secs)

    # 6a. place glutamatergic synapses
    if num_gluts > 1 and len(glutamate_locations) == 1:
        glutamate_locations = glutamate_locations*num_gluts

    glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs, final_spine_secs = glut_place4(
        cell = cell,
        spines = spines,
        physiological = physiological, 
        AMPA = AMPA, 
        g_AMPA = g_AMPA,
        NMDA = NMDA,
        ratio = ratio,
        glut_reversals = glut_reversals,
        glut = glut,
        glut_time = glut_time,
        glut_secs = glut_secs,
        glut_onsets = glut_onsets,
        glut_locs = glutamate_locations,
        num_gluts = num_gluts,
        return_currents = record_currents,
        axoshaft = axoshaft
    )
    glut_locs = []
    for spine in final_spines:
        glut_locs.append(spine.x)

    # 6b. place gabaergic synapses
    if num_gabas > 1 and len(gaba_locations) == 1:
        gaba_locations = gaba_locations*num_gabas

    gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances, gaba_locs = gaba_place3(
        physiological = physiological,
        gaba_reversals = gaba_reversals,
        gaba_weight = g_GABA,
        gaba_tau1=gaba_tau1,
        gaba_tau2=gaba_tau2,
        gaba_time = gaba_time,
        gaba_secs = gaba_secs,
        gaba_onsets = gaba_onsets,
        gaba_locations = gaba_locations,
        num_gabas = num_gabas,
        return_currents = record_currents
    )   
                            
    if not voltage_clamp:
        # 7. setup recording locations
        if record_location is None:
            if glut:
                # loc = sum(glut_locs) / len(glut_locs) # midpoint
                loc = glutamate_locations[0] # first location
            else:
                # loc = sum(gaba_locations) / len(gaba_locations) # midpoint
                loc = gaba_locations[0] # first location
        else:
            loc = record_location[0]
        spine_dist = []
        if record_dendrite is None: # will assume want 1st listed dendrite
            dendrite = glut_secs[0] if glut else glut_secs_orig[0] if gaba else None
            spine = final_spines[0] if glut else glut_secs_orig[0] if gaba else None
            spine_dist = h.distance(dendrite(glut_locs[0])) 
        else:
            for sec in cell.allseclist:
                if sec.name() == record_dendrite:
                    dendrite = sec
            # only record from spine head IF only 1 glutamatergic input
            if num_gluts == 1:
                spine = final_spines[0]
                spine_dist = h.distance(dendrite(glut_locs[0]))
            else: 
                spine = None

        print('recording at {} with location {}'.format(dendrite, round(loc,4)))

        # 8. configure basic current clamp
        t = h.Vector().record(h._ref_t)
        iclamp1 = h.IClamp(cell.soma(0.5))
        iclamp1.dur = sim_time
        if add_noise or add_sine or add_ramp:
            iclamp1.amp = 0 # nA
        else:
            iclamp1.amp = hc # nA

        # 9. add further types of stimulus
        # add coloured noise
        if add_noise:
            samples = int(sim_time/dt)  # number of samples to generate (time series extension)
            noise = B * cn.powerlaw_psd_gaussian(beta, samples) + hc
            noise_vector = h.Vector()
            noise_vector.from_python(noise)
            tvec = h.Vector(np.linspace(dt, sim_time, int(sim_time/dt)))
            noise_vector.play(iclamp1._ref_amp, tvec, True)

        # add sine wave
        if add_sine:
            time = np.linspace(dt, sim_time, int(sim_time/dt))
            tvec = h.Vector(time)
            sine_wave = amplitude/1e3 * np.sin(2*np.pi*frequency*time/1e3)  + hc
            sine_vector = h.Vector()
            sine_vector.from_python(sine_wave)
            sine_vector.play(iclamp1._ref_amp, tvec, True)

        # # add current step
        # if current_step:
        #     step_end = step_start + step_duration
        #     iclamp2 = h.IClamp(cell.soma(0.5))
        #     iclamp2.delay = step_start
        #     iclamp2.dur = step_duration
        #     iclamp2.amp = cs # nA
        #     print(f"step clamp parameters: {iclamp2.delay} ms, {iclamp2.dur} ms, {iclamp2.amp*1e3:.2f} pA")

        # add at least one current step
        if current_step:
            cs = np.array(step_current, ndmin=1) / 1e3  # convert to nA
            ds = np.array(step_duration, ndmin=1)
            ts = np.array(step_start, ndmin=1)
        
            n_steps = len(cs)
            iclamps = []
        
            for i in range(n_steps):
                iclamp = h.IClamp(cell.soma(0.5))
                iclamp.delay = float(ts[i])
                iclamp.dur   = float(ds[i])
                iclamp.amp   = float(cs[i])  # << use scaled current here
                iclamps.append(iclamp)
                print(f"Step {i+1}: {iclamp.delay}–{iclamp.delay+iclamp.dur} ms, {iclamp.amp*1e3:.1f} pA")
        

        # create ramp
        if add_ramp:
            time = np.linspace(dt, sim_time, int(sim_time / dt))
            tvec = h.Vector(time)
            ramp_wave = np.full_like(time, hc)  # holding current always on
        
            stim_index = int(stim_time / dt)
            ramp_wave[stim_index:] += ramp_amplitude * ((time[stim_index:] - stim_time) / (sim_time - stim_time))
        
            ramp_vector = h.Vector()
            ramp_vector.from_python(ramp_wave)
            ramp_vector.play(iclamp1._ref_amp, tvec, True)
    
    else:
        # 7. setup recording locations
        loc = voltage_clamp_loc
        if voltage_clamp_spine:
            spine = final_spines[0]
            section = spine.head
            loc = 0.5
        else:
            for sec in cell.allseclist:
                if sec.name() == voltage_clamp_site:
                    section = sec

        print('voltage clamp at {} with location {}'.format(section, round(loc,4)))

        # 8. setup voltage recording site for synaptic location (for voltage breakthrough)
        if glut:
            loc1 = glutamate_locations[0] # first location
        else:
            loc1 = gaba_locations[0] # first location

        spine_dist = []
        if record_dendrite is None: # will assume want 1st listed dendrite
            dendrite = glut_secs[0] if glut else glut_secs_orig[0] if gaba else None
            spine = final_spines[0] if glut else glut_secs_orig[0] if gaba else None
            spine_dist = h.distance(dendrite(glut_locs[0])) 
        else:
            for sec in cell.allseclist:
                if sec.name() == record_dendrite:
                    dendrite = sec
            # only record from spine head IF only 1 glutamatergic input
            if num_gluts == 1:
                spine = final_spines[0]
                spine_dist = h.distance(dendrite(glut_locs[0]))
            else: 
                spine = None

        # 9. configure basic voltage clamp

        t = h.Vector().record(h._ref_t)
        vclamp = h.SEClamp(section(loc))
        vclamp.dur1 = sim_time
        vclamp.amp1 = holding_potential # mV
        vclamp.rs = Rs # MOhm
        i_recording_site = h.Vector().record(vclamp._ref_i) 
    
    # 10. set up vectors
    vsoma = h.Vector()
    vdend = h.Vector()
    
    if voltage_clamp:
        v_recording_site = h.Vector()
        v_recording_site.record(section(loc)._ref_v)
        vdend.record(dendrite(loc1)._ref_v)
    else:
        vdend.record(dendrite(loc)._ref_v)
    vsoma.record(cell.soma(0.5)._ref_v)

    if num_gluts == 1:
        record_spine = True
        vspine = h.Vector()
        vspine.record(spine.head(0.5)._ref_v)
    else:
        record_spine = False 

    v_spine_activated, v_dend_activated, dists_spine_activated = record_all_activated_spine_v2(cell=cell, dendrite=dendrite, activated_spines=final_spines) 
    
    v_spine_activated = {
            'v': v_spine_activated,
            'dists': dists_spine_activated,
            'dendrites': final_spine_secs,
            'locs': final_spine_locs
        }
        
    v_dend_activated = {
            'v': v_dend_activated,
            'dists': dists_spine_activated
        }
    if record_Ca:
        Ca_soma = h.Vector(); Ca_dend = h.Vector()
        Ca_soma.record(cell.soma(0.5)._ref_cai)        
        Ca_dend.record(dendrite(loc)._ref_cai)
        
        if record_spine:
            Ca_spine = h.Vector()
            Ca_spine.record(spine.head(0.5)._ref_cai)

    if record_path_dend:
        # all voltages in path in dendrite
        v_dend_tree, dists_tree, dends_v = record_all_path_secs_v2(cell=cell, dend_tree=dend_tree, dendrite=dendrite)
        v_dend_tree = {
            'v': v_dend_tree,
            'dists': dists_tree,
            'dendrites': dends_v
        }    

    if record_path_spines:
        # all voltages in path in a spine
        v_spine_tree, dists_tree, dends_spine, locs_spine = record_all_path_secs_spine_v2(cell=cell, spines=spines, dend_tree=dend_tree, dendrite=dendrite, activated_spines=final_spines)
        v_spine_tree = {
            'v': v_spine_tree,
            'dists': dists_tree,
            'dendrites': dends_spine,
            'locs': locs_spine
        } 

    if record_mechs:
        # i_mechs at recording site
        mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kcnq', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']
        i_mechs_dend = record_i_mechs(cell=cell, dend=dendrite, loc=loc, return_currents=record_currents, mechs=mechs)
        i_mechs_dend ={
            'i': i_mechs_dend,
            'mechs': mechs
        }
        if record_path_dend:
            # all i_mechs in path in dendrite
            i_mechs_dend_tree, dists_tree, dends_i = record_all_path_secs_i_mechs(cell=cell, dend_tree=dend_tree, dendrite=dendrite, mechs=mechs)
            i_mechs_dend_tree = {
                'i': i_mechs_dend_tree,
                'dists': dists_tree,
                'dendrites': dends_i,
                'mechs': mechs
            } 

        if record_path_spines:
            # all i_mechs in path in spines
            spine_mechs = ['pas', 'kir', 'cal12', 'cal13', 'car', 'cav32', 'cav33', 'sk']
            i_mechs_spine_tree, dists_tree, dends_i = record_all_path_secs_spine_i_mechs(cell=cell, spines=spines, dend_tree=dend_tree, dendrite=dendrite, activated_spines=final_spines, spine_mechs=spine_mechs)
            i_mechs_spine_tree = {
                'i': i_mechs_spine_tree,
                'dists': dists_tree,
                'dendrites': dends_i,
                'mechs': spine_mechs
            }
    
    if record_branch:
        for dend_name in dend_glut2:
            dendrite_branch = None
            # Find the dendrite in the cell's section list
            for dend in cell.allseclist:
                if dend.name() == dend_name:
                    dendrite_branch = dend
                    break
            # Collect the data
            v, _, _, dists = sec_all_v(section=dendrite_branch, all_v={}, i=0)
            v_branch[dend_name] = {
                'v': v,
                'dists': dists
            }

    # print('\n---- nseg before record_all_3D ----')
    # for sec in h.allsec():
    #     print(f"{sec.name():25s}  nseg = {sec.nseg}")

    
    # if want 3D heatmaps
    if record_3D:
        # all voltages at 3D coordinates
        all_v, cell_coordinates, dends3D, dists3D = record_all_3D(cell)
        v_all_3D = {
            'v': all_v,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists3D,
            'dendrites': dends3D
        }   

    # print('\n---- nseg after record_all_3D ----')
    # for sec in h.allsec():
    #     print(f"{sec.name():25s}  nseg = {sec.nseg}")

    
    # if want 3D heatmaps
    if record_3D_Ca:
        # all voltages at 3D coordinates
        all_Ca, cell_coordinates, dends3D, dists3D = record_Ca_3D(cell)
        Ca_all_3D = {
            'Ca': all_Ca,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists3D,
            'dendrites': dends3D
        }   

    if record_3D_mechs:
        # i_mechs at recording site
        if mechs2record == None:
            mechs3D = ['cal12', 'cal13', 'can', 'car', 'cav32', 'cav33']
        else:
            mechs3D = mechs2record
        all_i_mechs, cell_coordinates, dends3D, dists3D = record_mechs_3D(cell, mechs=mechs3D)
        i_mechs_3D ={
            'i': all_i_mechs,
            'cell_coordinates': cell_coordinates,
            'dists': dists3D,
            'dendrites': dends3D,
            'mechs': mechs3D
        }        

    # 11. run simulation
    # try:
    #     h.cvode.use_fast_imem(1)
    # except:`1
    #     pass
    
    pc = h.ParallelContext()
    _nt = int(pc.nthread())
    
    # set threads BEFORE initialize
    if record_3D_impedance:
        pc.nthread(1)
    
    h.dt = dt
    h.finitialize(v_init)
    
    if record_3D_impedance:
        impedance_locations, cell_coordinates, dends, dists = setup_impedance_measurements(cell)
    #         impedance_transfer_locations, _, _, _ = setup_impedance_measurements(cell)
    
        # Initialize a dictionary to store impedance vectors for each location
        impedance_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_phase_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_transfer_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_transfer_phase_vectors = {loc: h.Vector() for loc in impedance_locations}
    
        if record_spine:
            imp_spine = h.Vector()
            imp_spine_phase = h.Vector()
        
        # Initialize Impedance object
        imp = h.Impedance()
        imp.loc(loc, sec=dendrite) # location of interest, impedance transfer is relative to this point 
        
        # Initialize a variable to track the next time to compute impedance
        next_impedance_time = 0 + burn_time # only start calculating impedance when burn_time is over
        # ds_imp = 40 # downsample to ds_imp * h.dt
        
        # Simulation loop
        while h.t < sim_time:
            if h.t >= round(next_impedance_time, 3):
                # Compute impedance at the specified frequency
                # second argument 1 calculates extended impedance; takes into account differential gating variables
                # of all membrane mechanisms 
                imp.compute(freq, 1)
    
                if num_gluts == 1:
                    # Record input impedance magnitude and phase at the spine head
                    imp_spine.append(imp.input(0.5, sec=spine.head))
                    imp_spine_phase.append(imp.input_phase(0.5, sec=spine.head))  # Phase in radians
    
                # Record impedance magnitude and phase at each location
                for (sec, loc) in impedance_locations:
                    # Append the impedance value to the corresponding vector
                    impedance_vectors[(sec, loc)].append(imp.input(loc, sec=sec))
                    impedance_phase_vectors[(sec, loc)].append(imp.input_phase(loc, sec=sec))  # Phase in radians
    
                    # Record transfer impedance magnitude and phase
                    impedance_transfer_vectors[(sec, loc)].append(imp.transfer(loc, sec=sec))
                    impedance_transfer_phase_vectors[(sec, loc)].append(imp.transfer_phase(loc, sec=sec))  # Transfer phase
    
                # Schedule the next impedance computation
                next_impedance_time += dt * ds_imp
    
            # Advance the simulation by one time step
            h.fadvance()
           
        imp_all_3D = {
            'imp': list(impedance_vectors.values()),
            'imp phase': list(impedance_phase_vectors.values()),
            'imp transfer': list(impedance_transfer_vectors.values()),
            'imp transfer phase': list(impedance_transfer_phase_vectors.values()),
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists,
            'dendrites': dends
        }
        pc.nthread(_nt)
    
    else:
        while h.t < sim_time:
            h.fadvance()

    record_dist = h.distance(dendrite(loc))
    
    # make outputs as numpy arrays
    if record_spine:
        vspine = np.array(vspine)
        if record_3D_impedance:
            imp_spine=np.array(imp_spine)
    
    if voltage_clamp:
        i_recording_site = np.array(i_recording_site); v_recording_site = np.array(v_recording_site)
    
    vdend = np.array(vdend); vsoma = np.array(vsoma)
    
    if record_Ca:  
        Ca_dend = np.array(Ca_dend); Ca_soma = np.array(Ca_soma)
        if record_spine:
            Ca_spine = np.array(Ca_spine)
            
    v_spine_activated['v'] = vec2np(v_spine_activated['v'])
    v_dend_activated['v'] = vec2np(v_dend_activated['v'])
    
    if record_path_dend:
        v_dend_tree['v'] = vec2np(v_dend_tree['v'])
    if record_path_spines:
        v_spine_tree['v'] = vec2np(v_spine_tree['v'])
    if record_mechs:
        i_mechs_dend['i'] = vec2np(i_mechs_dend['i'])

        if record_path_dend:        
            out = {}
            for name, data in i_mechs_dend_tree['i'].items():
                # initialize a list to hold NumPy arrays for each dendrite
                out[name] = vec2np(data)
            i_mechs_dend_tree['i'] = out

        if record_path_spines:    
            out = {}
            for name, data in i_mechs_spine_tree['i'].items():
                out[name] = vec2np(data)
            i_mechs_spine_tree['i'] = out

    
    if record_branch:
        # Convert NEURON Vectors in v_branch to NumPy arrays
        for dend_name, data in v_branch.items():
            # Initialize a list to hold NumPy arrays for the current dendrite
            v_branch_np_arrays = []

            # Iterate through each Vector in 'v' and convert to NumPy array
            for i, vector in data['v'].items():
                np_array = np.array(vector)
                v_branch_np_arrays.append(np_array)

            # Replace the original 'v' data with the list of NumPy arrays
            v_branch[dend_name]['v'] = vec2np(v_branch[dend_name]['v'])        
    
    if record_3D:
        v_all_3D['v'] = vec2np(v_all_3D['v'])
    
    if record_3D_Ca:
        Ca_all_3D['Ca'] = vec2np(Ca_all_3D['Ca'])

    if record_3D_impedance:
        imp_all_3D['imp'] = vec2np(imp_all_3D['imp'])
        imp_all_3D['imp phase'] = vec2np(imp_all_3D['imp phase'])
        imp_all_3D['imp transfer'] = vec2np(imp_all_3D['imp transfer'])
        imp_all_3D['imp transfer phase'] = vec2np(imp_all_3D['imp transfer phase'])
        if record_spine:
            imp_all_3D['imp spine'] = imp_spine

    if record_3D_mechs:        
        out = {}
        for name, data in i_mechs_3D['i'].items():
            # initialize a list to hold NumPy arrays for each dendrite
            out[name] = vec2np(data)
        i_mechs_3D['i'] = out
    
    if record_currents:
        ampa_currents = vec2np(ampa_currents)
        nmda_currents = vec2np(nmda_currents)
        gaba_currents = vec2np(gaba_currents)
        gaba_conductances = vec2np(gaba_conductances)

    return i_recording_site, v_recording_site, v_all_3D, Ca_all_3D, imp_all_3D, i_mechs_3D, vspine, v_spine_activated, vdend, v_dend_activated, vsoma, v_dend_tree, v_spine_tree, Ca_spine, Ca_dend, Ca_soma, i_mechs_dend, i_mechs_dend_tree, i_mechs_spine_tree, v_branch, zdend, ztransfer, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, record_dist, record_spine, spine_dist, capacitance, h.dt, burn_time, start_time
    
def CurrentClamp(sim_time, 
                    stim_time,
                    baseline,
                    glut, 
                    glut_frequency, 
                    glutamate_locations, 
                    glut_reversals,
                    glut_time, 
                    num_gluts, 
                    dend_glut, 
                    g_AMPA,
                    ratio,
                    gaba, 
                    gaba_frequency, 
                    gaba_locations,
                    gaba_reversals,
                    gaba_time, 
                    g_GABA, 
                    num_gabas, 
                    dend_gaba, 
                    specs, 
                    scaling_factor=None,
                    gaba_tau1=0.9,
                    gaba_tau2=18,
                    rel_gaba_onsets=None,
                    rel_glut_onsets=None,
                    frequency=2000,
                    d_lambda=0.05,
                    dend2remove=None,
                    v_init=-84, 
                    AMPA=True,
                    NMDA=True,
                    physiological=True,
                    timing_range=None, 
                    add_noise=False,
                    beta=0,                    
                    B=1,                       
                    add_sine=False, 
                    axoshaft=False,
                    cell_type='dspn',
                    current_step=False,
                    step_current=-200,
                    step_duration=500,
                    step_start = 300,
                    holding_current=0,
                    Cm=1,
                    Ra=200,
                    g_name_list=None,
                    g8_list=None,
                    spine_per_length=1.61,
                    spine_neck_diam=0.1,
                    spine_neck_len=1,
                    spine_head_diam=0.5,
                    soma_diameter=None,
                    neck_dynamics=False,
                    space_clamp=False,
                    record_dendrite=None, 
                    record_location=None, 
                    record_currents=False,     
                    record_branch=False,       
                    dend_glut2=None,
                    record_mechs=False,
                    mechs2record=None,
                    record_path_dend=False,    
                    record_path_spines=False,  
                    record_all_path=True,      
                    record_3D=False,           
                    record_3D_impedance=False, 
                    freq=10,                   
                    record_3D_mechs=False,     
                    record_Ca=False,
                    record_3D_Ca=False,
                    tonic=False,
                    gbar_gaba=None,
                    rectification=False,       
                    distributed=False,         
                    gaba_params=None,
                    tonic_gaba_reversal=-60,
                    dt =0.025
                    ):

    # 1. initialize variables
    vspine = []; vdend = []; vsoma = []; zdend = []; ztransfer = []
    v_dend_tree = {}; v_spine_tree = {}; i_mechs_dend = {}; i_mechs_dend_tree = {};
    v_spine_activated = {}; v_dend_activated = {};
    i_mechs_spine_tree = {}; i_mechs_spine_tree = {}; v_branch = {} 
    v_all_3D = {}; imp_all_3D = {}; Ca_all_3D = {}; i_mechs_3D = {}
    Ca_spine = []; Ca_dend = []; Ca_soma = []

    start_time = min(stim_time, *timing_range)
    burn_time = start_time - baseline  # time allowed to reach clamped potential

    if current_step: cs = step_current/1e3        
    hc = holding_current/1e3

    # 2. build cell 
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, dend2remove=dend2remove, neck_dynamics=neck_dynamics)

    # 3. set up tonic gaba
    if tonic:
        if distributed:
            tonic_gaba(cell=cell, gaba_reversal=tonic_gaba_reversal, gbar_gaba=gbar_gaba, d3=gaba_params['d3'], a4=gaba_params['a4'], a5=gaba_params['a5'], a6=gaba_params['a6'], a7=gaba_params['a7'], rectification=rectification)                
        else:
            tonic_gaba(cell=cell, gaba_reversal=tonic_gaba_reversal, gbar_gaba=gbar_gaba, rectification=rectification)

    # 4. change properties

    if Ra != 200:
        space_clamped(cell=cell, spines=spines, Ra=Ra)
        print(f"Ra is {Ra} ΜΩ")
    
    if Cm != 1:
        cap(cell=cell, spines=spines, cm = Cm)
        print(f"Cm is {Cm} μFcm\u207B\u00B2")

    if scale_factors is not None:
        printed = set()  # track which mechanisms have already been reported
        for sec in h.allsec():
            for seg in sec:
                for mech, sf in scale_factors.items():
                    if hasattr(seg, mech):
                        getattr(seg, mech).sf = sf
                        if mech not in printed:
                            if mech in ['cal12', 'cal13', 'can', 'car', 'cav32', 'cav33']:
                                print(f"permeability (cm/s) scaled: '{mech}' scaling factor = {sf}")
                            else:
                                print(f"conductance (S/cm²) scaled: '{mech}' scaling factor = {sf}")
                            printed.add(mech)
        
    # change all spine neck diameters
    if spine_neck_diam != 0.1:
        spine_neck_diameter(cell=cell, spines=spines, diam=spine_neck_diam)
        print(f"spine neck diameter {spine_neck_diam} μm")
        
    # change all spine neck lengths
    if spine_neck_len != 1:
        spine_neck_length(cell=cell, spines=spines, length=spine_neck_len)
        print(f"spine neck length {spine_neck_len} μm")
        
    # change all spine head diameters
    if spine_head_diam != 0.5:
        spine_head_diameter(cell=cell, spines=spines, diam=spine_head_diam, length=spine_head_diam)
        print(f"spine_head_diameter {spine_head_diam} μm")

    capacitance = whole_cell_capacitance(cell, spines, Cm=Cm)
    
    # 5. prepare synaptic inputs
    if glut_frequency is not None:
        dstep1 = int(1 / glut_frequency * 1e3)
    else:
        dstep1 = 0
        
    if gaba_frequency is not None:
        dstep2 = int(1 / gaba_frequency * 1e3)

    glut_secs = [sec for target_dend in dend_glut for sec in cell.dendlist if sec.name() == target_dend]
    if num_gluts > 1 and len(glut_secs) == 1:
        glut_secs = glut_secs * num_gluts

    if gaba:
        gaba_secs = [sec for target_dend in dend_gaba for sec in cell.allseclist if sec.name() == target_dend]
    else:
        gaba_secs = []

    if num_gabas > 1 and len(gaba_secs) == 1:
        gaba_secs = gaba_secs * num_gabas

    # phasic glut
    if rel_glut_onsets:
        glut_onsets = [glut_time + x for x in rel_glut_onsets]
    else:
        if dstep1 > 0:
            glut_onsets = list(range(glut_time, glut_time + num_gluts * dstep1, dstep1))
        else:
            glut_onsets = [glut_time] * num_gluts  


    # phasic gaba
    # if rel_gaba_onsets is not None and rel_gaba_onsets.size > 0: # rel_gaba_onsets is numpy array

    if rel_gaba_onsets is not None and (
        (isinstance(rel_gaba_onsets, list) and len(rel_gaba_onsets) > 0) or
        (isinstance(rel_gaba_onsets, np.ndarray) and rel_gaba_onsets.size > 0)
    ):        gaba_onsets = [x + gaba_time for x in rel_gaba_onsets]
    else:
        if gaba_frequency is None:
            gaba_onsets = [gaba_time] * len(gaba_secs)
        else:
            gaba_onsets = list(range(gaba_time, gaba_time + num_gabas * dstep2, dstep2)) * len(gaba_secs)

    # 6a. place glutamatergic synapses
    if num_gluts > 1 and len(glutamate_locations) == 1:
        glutamate_locations = glutamate_locations*num_gluts

    glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs, final_spine_secs = glut_place4(
        cell = cell,
        spines = spines,
        physiological = physiological, 
        AMPA = AMPA, 
        g_AMPA = g_AMPA,
        NMDA = NMDA,
        ratio = ratio,
        glut_reversals = glut_reversals,
        glut = glut,
        glut_time = glut_time,
        glut_secs = glut_secs,
        glut_onsets = glut_onsets,
        glut_locs = glutamate_locations,
        num_gluts = num_gluts,
        return_currents = record_currents,
        axoshaft = axoshaft
    )
    glut_locs = []
    for spine in final_spines:
        glut_locs.append(spine.x)

    # 6b. place gabaergic synapses
    if num_gabas > 1 and len(gaba_locations) == 1:
        gaba_locations = gaba_locations*num_gabas

    gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances, gaba_locs = gaba_place3(
        physiological = physiological,
        gaba_reversals = gaba_reversals,
        gaba_weight = g_GABA,
        gaba_tau1=gaba_tau1,
        gaba_tau2=gaba_tau2,
        gaba_time = gaba_time,
        gaba_secs = gaba_secs,
        gaba_onsets = gaba_onsets,
        gaba_locations = gaba_locations,
        num_gabas = num_gabas,
        return_currents = record_currents
    )   

    # 7. setup recording locations
    if record_location is None:
        if glut:
            # loc = sum(glut_locs) / len(glut_locs) # midpoint
            loc = glutamate_locations[0] # first location
        else:
            # loc = sum(gaba_locations) / len(gaba_locations) # midpoint
            loc = gaba_locations[0] # first location
    else:
        loc = record_location[0]
    spine_dist = []
    if record_dendrite is None: # will assume want 1st listed dendrite
        dendrite = glut_secs[0] if glut else glut_secs_orig[0] if gaba else None
        spine = final_spines[0] if glut else glut_secs_orig[0] if gaba else None
        spine_dist = h.distance(dendrite(glut_locs[0])) 
    else:
        for sec in cell.allseclist:
            if sec.name() == record_dendrite:
                dendrite = sec
        # only record from spine head IF only 1 glutamatergic input
        if num_gluts == 1:
            spine = final_spines[0]
            spine_dist = h.distance(dendrite(glut_locs[0]))
        else: 
            spine = None

    print('recording at {} with location {}'.format(dendrite, round(loc,4)))

    # 8. configure basic current clamp
    t = h.Vector().record(h._ref_t)
    iclamp1 = h.IClamp(cell.soma(0.5))
    iclamp1.dur = sim_time
    if add_noise:
        iclamp1.amp = 0 # nA
    else:
        iclamp1.amp = hc # nA

    # 9. add further types of stimulus
    # add coloured noise
    if add_noise:
        samples = int(sim_time/dt)  # number of samples to generate (time series extension)
        noise = B * cn.powerlaw_psd_gaussian(beta, samples) + hc
        noise_vector = h.Vector()
        noise_vector.from_python(noise)
        tvec = h.Vector(np.linspace(dt, sim_time, int(sim_time/dt)))
        noise_vector.play(iclamp1._ref_amp, tvec, True)

    # add sine wave
    if add_sine:
        time = np.linspace(dt, sim_time, int(sim_time/dt))
        tvec = h.Vector(time)
        sine_wave = amplitude/1e3 * np.sin(2*np.pi*frequency*time/1e3)
        sine_vector = h.Vector()
        sine_vector.from_python(sine_wave)
        sine_vector.play(iclamp1._ref_amp, tvec, True)

    # add current step
    if current_step:
        step_end = step_start + step_duration
        iclamp2 = h.IClamp(cell.soma(0.5))
        iclamp2.delay = step_start
        iclamp2.dur = step_duration
        iclamp2.amp = cs # nA

    # 10. set up vectors
    vsoma = h.Vector()
    vdend = h.Vector()
    vsoma.record(cell.soma(0.5)._ref_v)
    vdend.record(dendrite(loc)._ref_v)

    if num_gluts == 1:
        record_spine = True
        vspine = h.Vector()
        vspine.record(spine.head(0.5)._ref_v)
    else:
        record_spine = False 

    v_spine_activated, v_dend_activated, dists_spine_activated = record_all_activated_spine_v2(cell=cell, dendrite=dendrite, activated_spines=final_spines) 
    v_spine_activated = {
            'v': v_spine_activated,
            'dists': dists_spine_activated
        }  
    v_dend_activated = {
            'v': v_dend_activated,
            'dists': dists_spine_activated
        }
    if record_Ca:
        Ca_soma = h.Vector(); Ca_dend = h.Vector()
        Ca_soma.record(cell.soma(0.5)._ref_cai)        
        Ca_dend.record(dendrite(loc)._ref_cai)
        
        if record_spine:
            Ca_spine = h.Vector()
            Ca_spine.record(spine.head(0.5)._ref_cai)

    if record_path_dend:
        # all voltages in path in dendrite
        v_dend_tree, dists_tree, dends_v = record_all_path_secs_v2(cell=cell, dend_tree=dend_tree, dendrite=dendrite)
        v_dend_tree = {
            'v': v_dend_tree,
            'dists': dists_tree,
            'dendrites': dends_v
        }    

    if record_path_spines:
        # all voltages in path in a spine
        v_spine_tree, dists_tree, dends_spine, locs_spine = record_all_path_secs_spine_v2(cell=cell, spines=spines, dend_tree=dend_tree, dendrite=dendrite, activated_spines=final_spines)
        v_spine_tree = {
            'v': v_spine_tree,
            'dists': dists_tree,
            'dendrites': dends_spine,
            'locs': locs_spine
        } 

    if record_mechs:
        # i_mechs at recording site
        mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kcnq', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']
        i_mechs_dend = record_i_mechs(cell=cell, dend=dendrite, loc=loc, return_currents=record_currents, mechs=mechs)
        i_mechs_dend ={
            'i': i_mechs_dend,
            'mechs': mechs
        }
        if record_path_dend:
            # all i_mechs in path in dendrite
            i_mechs_dend_tree, dists_tree, dends_i = record_all_path_secs_i_mechs(cell=cell, dend_tree=dend_tree, dendrite=dendrite, mechs=mechs)
            i_mechs_dend_tree = {
                'i': i_mechs_dend_tree,
                'dists': dists_tree,
                'dendrites': dends_i,
                'mechs': mechs
            } 

        if record_path_spines:
            # all i_mechs in path in spines
            spine_mechs = ['pas', 'kir', 'cal12', 'cal13', 'car', 'cav32', 'cav33', 'sk']
            i_mechs_spine_tree, dists_tree, dends_i = record_all_path_secs_spine_i_mechs(cell=cell, spines=spines, dend_tree=dend_tree, dendrite=dendrite, activated_spines=final_spines, spine_mechs=spine_mechs)
            i_mechs_spine_tree = {
                'i': i_mechs_spine_tree,
                'dists': dists_tree,
                'dendrites': dends_i,
                'mechs': spine_mechs
            }
    
    if record_branch:
        for dend_name in dend_glut2:
            dendrite_branch = None
            # Find the dendrite in the cell's section list
            for dend in cell.allseclist:
                if dend.name() == dend_name:
                    dendrite_branch = dend
                    break
            # Collect the data
            v, _, _, dists = sec_all_v(section=dendrite_branch, all_v={}, i=0)
            v_branch[dend_name] = {
                'v': v,
                'dists': dists
            }
    
    # if want 3D heatmaps
    if record_3D:
        # all voltages at 3D coordinates
        all_v, cell_coordinates, dends3D, dists3D = record_all_3D(cell)
        v_all_3D = {
            'v': all_v,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists3D,
            'dendrites': dends3D
        }   


    # if want 3D heatmaps
    if record_3D_Ca:
        # all voltages at 3D coordinates
        all_Ca, cell_coordinates, dends3D, dists3D = record_Ca_3D(cell)
        Ca_all_3D = {
            'Ca': all_Ca,
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists3D,
            'dendrites': dends3D
        }   

    if record_3D_mechs:
        # i_mechs at recording site
        if mechs2record == None:
            mechs3D = ['cal12', 'cal13', 'can', 'car', 'cav32', 'cav33']
        else:
            mechs3D = mechs2record
        all_i_mechs, cell_coordinates, dends3D, dists3D = record_mechs_3D(cell, mechs=mechs3D)
        i_mechs_3D ={
            'i': all_i_mechs,
            'cell_coordinates': cell_coordinates,
            'dists': dists3D,
            'dendrites': dends3D,
            'mechs': mechs3D
        }        

    # 11. run simulation
    h.dt = dt
    h.finitialize(v_init)
        
    if record_3D_impedance:
        impedance_locations, cell_coordinates, dends, dists = setup_impedance_measurements(cell)
#         impedance_transfer_locations, _, _, _ = setup_impedance_measurements(cell)

        # Initialize a dictionary to store impedance vectors for each location
        impedance_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_phase_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_transfer_vectors = {loc: h.Vector() for loc in impedance_locations}
        impedance_transfer_phase_vectors = {loc: h.Vector() for loc in impedance_locations}

        if record_spine:
            imp_spine = h.Vector()
            imp_spine_phase = h.Vector()
        
        # Initialize Impedance object
        imp = h.Impedance()
        
        # set location of transfer; sinusoidal current from this point
        # location of interest, impedance transfer is relative to this point
        # impedance transfer is from this point to the specified locations below 
        # notre transfer phase is symmetric i.e Zij(f) = Zji)f so if calculate from dendrite then to dendrite from locations is the same
        imp.loc(loc, sec=dendrite) 
        print('Transfer impedance measured relative to section {} at location {:.4f}'.format(dendrite, loc))
        print('Note: By reciprocity, Zij(f) = Zji(f); symmetric for both magnitude and phase when using imp.compute(freq, 1)')
                
        # Initialize a variable to track the next time to compute impedance
        next_impedance_time = 0 + burn_time # only start calculating impedance when burn_time is over
        ds_imp = 40 # downsample to ds_imp * h.dt
        
        # Simulation loop
        while h.t < sim_time:
            if h.t >= round(next_impedance_time, 3):
                # Compute impedance at the specified frequency
                # second argument 1 calculates extended impedance; takes into account differential gating variables
                # of all membrane mechanisms 
                imp.compute(freq, 1)

                if num_gluts == 1:
                    # Record input impedance magnitude and phase at the spine head
                    imp_spine.append(imp.input(0.5, sec=spine.head))
                    imp_spine_phase.append(imp.input_phase(0.5, sec=spine.head))  # Phase in radians

                # Record impedance magnitude and phase at each location
                for (sec, loc) in impedance_locations:
                    # Append the impedance value to the corresponding vector
                    impedance_vectors[(sec, loc)].append(imp.input(loc, sec=sec))
                    impedance_phase_vectors[(sec, loc)].append(imp.input_phase(loc, sec=sec))  # Phase in radians

                    # Record transfer impedance magnitude and phase
                    impedance_transfer_vectors[(sec, loc)].append(imp.transfer(loc, sec=sec))
                    impedance_transfer_phase_vectors[(sec, loc)].append(imp.transfer_phase(loc, sec=sec))  # Transfer phase

                # Schedule the next impedance computation
                next_impedance_time += dt * ds_imp

            # Advance the simulation by one time step
            h.fadvance()
           
        imp_all_3D = {
            'imp': list(impedance_vectors.values()),
            'imp phase': list(impedance_phase_vectors.values()),
            'imp transfer': list(impedance_transfer_vectors.values()),
            'imp transfer phase': list(impedance_transfer_phase_vectors.values()),
            'cell_coordinates': cell_coordinates,
            'cell_coordinates_col': ['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'], 
            'dists': dists,
            'dendrites': dends
        }

    else:
        while h.t < sim_time:
            h.fadvance()

    record_dist = h.distance(dendrite(loc))
    
    # make outputs as numpy arrays
    if record_spine:
        vspine = np.array(vspine)
        if record_3D_impedance:
            imp_spine=np.array(imp_spine)
    
    vdend = np.array(vdend); vsoma = np.array(vsoma)
    
    if record_Ca:  
        Ca_dend = np.array(Ca_dend); Ca_soma = np.array(Ca_soma)
        if record_spine:
            Ca_spine = np.array(Ca_spine)
            
    v_spine_activated['v'] = vec2np(v_spine_activated['v'])
    v_dend_activated['v'] = vec2np(v_dend_activated['v'])
    
    if record_path_dend:
        v_dend_tree['v'] = vec2np(v_dend_tree['v'])
    if record_path_spines:
        v_spine_tree['v'] = vec2np(v_spine_tree['v'])
    if record_mechs:
        i_mechs_dend['i'] = vec2np(i_mechs_dend['i'])

        if record_path_dend:        
            out = {}
            for name, data in i_mechs_dend_tree['i'].items():
                # initialize a list to hold NumPy arrays for each dendrite
                out[name] = vec2np(data)
            i_mechs_dend_tree['i'] = out

        if record_path_spines:    
            out = {}
            for name, data in i_mechs_spine_tree['i'].items():
                out[name] = vec2np(data)
            i_mechs_spine_tree['i'] = out

    
    if record_branch:
        # Convert NEURON Vectors in v_branch to NumPy arrays
        for dend_name, data in v_branch.items():
            # Initialize a list to hold NumPy arrays for the current dendrite
            v_branch_np_arrays = []

            # Iterate through each Vector in 'v' and convert to NumPy array
            for i, vector in data['v'].items():
                np_array = np.array(vector)
                v_branch_np_arrays.append(np_array)

            # Replace the original 'v' data with the list of NumPy arrays
            v_branch[dend_name]['v'] = vec2np(v_branch[dend_name]['v'])        
    
    if record_3D:
        v_all_3D['v'] = vec2np(v_all_3D['v'])
    
    if record_3D_Ca:
        Ca_all_3D['Ca'] = vec2np(Ca_all_3D['Ca'])

    if record_3D_impedance:
        imp_all_3D['imp'] = vec2np(imp_all_3D['imp'])
        imp_all_3D['imp phase'] = vec2np(imp_all_3D['imp phase'])
        imp_all_3D['imp transfer'] = vec2np(imp_all_3D['imp transfer'])
        imp_all_3D['imp transfer phase'] = vec2np(imp_all_3D['imp transfer phase'])
        if record_spine:
            imp_all_3D['imp spine'] = imp_spine

    if record_3D_mechs:        
        out = {}
        for name, data in i_mechs_3D['i'].items():
            # initialize a list to hold NumPy arrays for each dendrite
            out[name] = vec2np(data)
        i_mechs_3D['i'] = out
    
    if record_currents:
        ampa_currents = vec2np(ampa_currents)
        nmda_currents = vec2np(nmda_currents)
        gaba_currents = vec2np(gaba_currents)
        gaba_conductances = vec2np(gaba_conductances)

    return v_all_3D, Ca_all_3D, imp_all_3D, i_mechs_3D, vspine, v_spine_activated, vdend, v_dend_activated, vsoma, v_dend_tree, v_spine_tree, Ca_spine, Ca_dend, Ca_soma, i_mechs_dend, i_mechs_dend_tree, i_mechs_spine_tree, v_branch, zdend, ztransfer, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, record_dist, record_spine, spine_dist, capacitance, h.dt, burn_time, start_time

def syn_reversals(cell_type, specs, spine_per_length, soma_diameter, frequency, d_lambda, dend_glut, glut_reversal, glutamate_locations, num_gluts, dend_gaba, gaba_reversal, gaba_locations, num_gabas, sim_time, dt=0.025, v_init=-84, dend2remove=None, neck_dynamics=False):
    
    print("calculating reversal potentials for all unique locations...")

    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, verbose=False, dend2remove=dend2remove, neck_dynamics=neck_dynamics)
#     if tonic:
#         if distributed:
#             tonic_gaba(cell=cell, gaba_reversal=gaba_reversal, gbar_gaba=gbar_gaba, d3=gaba_params['d3'], a4=gaba_params['a4'], a5=gaba_params['a5'], a6=gaba_params['a6'], a7=gaba_params['a7'], rectification=rectification)                
#         else:
#             tonic_gaba(cell=cell, gaba_reversal=gaba_reversal, gbar_gaba=gbar_gaba, rectification=rectification)

    if gaba_reversal == 'Edend':

        if len(gaba_locations) != num_gabas and len(gaba_locations) == 1:
            gaba_locations = gaba_locations * num_gabas

        gaba_reversals = membrane_potentials(cell=cell, 
                                       dends=dend_gaba, 
                                       locs=gaba_locations,
                                       sim_time=sim_time,
                                       dt=dt,
                                       v_init=v_init
                                       )
    else:
        gaba_reversals = [gaba_reversal] * num_gabas

    if glut_reversal == 'Edend':

        if len(glutamate_locations) != num_gluts and len(glutamate_locations) == 1:
            glutamate_locations = glutamate_locations * num_gluts

        glut_reversals = membrane_potentials(cell=cell, 
                                       dends=dend_glut, 
                                       locs=glutamate_locations,
                                       sim_time=sim_time,
                                       dt=dt,
                                       v_init=v_init
                                       )
    else:
        glut_reversals = [glut_reversal] * num_gluts
        
    unique_gaba = pairs_in_order(dend_gaba, gaba_reversals)
    formatted_strs = ["{}: {:.2f} mV".format(d, round(rev, 2)) for d, rev in unique_gaba]
    formatted_output = "; ".join(formatted_strs)
    print("{} reversals: {}".format('GABA', formatted_output))
    
    unique_glut = pairs_in_order(dend_glut, glut_reversals)
    formatted_strs = ["{}: {:.2f} mV".format(d, round(rev, 2)) for d, rev in unique_glut]
    formatted_output = "; ".join(formatted_strs)
    print("{} reversals: {}".format('GLUT', formatted_output))
    
    return glut_reversals, gaba_reversals

def params_selector(cell_type, specs):
    model = specs[cell_type]['model']
    morphology = specs[cell_type]['morph']
    if cell_type == 'dspn':        
        if model == 0:
            params='params_dMSN0.json'
        elif model == 1:
            params='params_dMSN1.json'
        elif model == 2:
            params='params_dMSN2.json'
        elif model == 3:
            params='params_dMSN3.json'
            
    if cell_type == 'ispn':        
        if model == 0:
            params='params_iMSN0.json'
        elif model == 1:
            params='params_iMSN1.json'
        elif model == 2:
            params='params_iMSN2.json'
        elif model == 3:
            params='params_iMSN3.json'
    return(params)

def spines_per_dend(cell, spines):
    for sec in cell.dendlist:
        print(sec.name(), len(spines[sec.name()])) 
        
# function gives the distance of EVERY segment within a given dendrite from soma 
def seg_dist(cell, dend):
    dist = []
    for sec in cell.dendlist:
        if sec.name() == dend:
            for i,seg in enumerate(sec):
                dist.append(h.distance(seg))
    return(dist)

# get idxs for spines with UNIQUE locations on a particular dendrite 
# then use to find a unique spines in a given dendrite
def spine_idx(cell, spines, dend):
    for sec in cell.dendlist:
        if sec.name() == dend:
            Nseg = sec.nseg
    spine_locs = (2*np.linspace(1, Nseg, Nseg)-1)/Nseg/2
#     spine_loc = spine_locs[0]
#     spine_loc
    # Get possible spines from section
    candidate_spines = []
    sec_spines = list(spines[dend].items())

    for spine_i, spine_obj in sec_spines: 
        candidate_spines.append(spine_obj)

    # len(sec_spines)
    candidate_spines_locs = []
    for spine in candidate_spines:
        candidate_spines_locs.append(spine.x)
    # candidate_spines_locs
    # spine_idxs = []
    output = []
    for ii in range(Nseg):
        spine_loc = spine_locs[ii]
        a = abs(candidate_spines_locs - spine_loc)
        idx = np.argmin(a)
        # spine_idxs.append(idx)  
        output.append(candidate_spines[idx])
        # only return unique spines
        output = list(dict.fromkeys(output))
    return(output)

def calculate_dist(d3, dist, a4, a5,  a6,  a7, g8):
    '''
    Used for setting the maximal conductance of a segment.
    Scales the maximal conductance based on somatic distance and distribution type.

    Parameters:
    d3   = distribution type:
         0 linear, 
         1 sigmoidal, 
         2 exponential
         3 step function
    dist = somatic distance of segment
    a4-7 = distribution parameters 
    g8   = base conductance (similar to maximal conductance)
    '''

    if   d3 == 0: 
        value = a4 + a5*dist
    elif d3 == 1: 
        value = a4 + a5/(1 + np.exp((dist-a6)/a7) )
    elif d3 == 2: 
        value = a4 + a5*np.exp((dist-a6)/a7)
    elif d3 == 3:
        if (dist > a6) and (dist < a7):
            value = a4
        else:
            value = a5

    if value < 0:
        value = 0

    value = value*g8
    return value


# finds dendrites with at least 3 spines
def dend_spine_selector(cell, spines, branch_groups, n=2):
    dends_with_spines = []
    # Make list of dendrite sections with at least 2 spines 
    for dend in cell.dendlist:
        sec_spines = list(spines[dend.name()].items())
        for group in branch_groups: # for each nrn dendrite sec, one plot per branch
            if dend in group:
                if len(group) > n:
                    if len(sec_spines) > n:
                        dends_with_spines.append(dend)
    return dends_with_spines



# rectification is False then ohmic else rectification Pavlov 
def tonic_gaba(cell, gaba_reversal, gbar_gaba, d3=0, a4=1, a5=0, a6=0, a7=0, rectification=False):
    if rectification:
        for sec in cell.dendlist:
            sec.e_gaba2 = gaba_reversal
        for sec in cell.somalist:
            sec.e_gaba2 = gaba_reversal
        cell.distribute_channels('dend', 'gbar_gaba2', d3, a4, a5, a6, a7, gbar_gaba)
        
        g_name = 'gaba2'
        g = []
        gbar = 'gbar_{}'.format(g_name)  
        for sec in cell.dendlist:
            g.append((eval('sec.{}'.format(gbar))))        
        
        if g[0] < g[-1]:
            cell.distribute_channels('soma', 'gbar_gaba2', 0, 1, 0, 0, 0, g[0])
        else:
            cell.distribute_channels('soma', 'gbar_gaba2', 0, 1, 0, 0, 0, gbar_gaba)

    else:
        for sec in cell.dendlist:
            sec.e_gaba1 = gaba_reversal
        for sec in cell.somalist:
            sec.e_gaba1 = gaba_reversal
        cell.distribute_channels('dend', 'gbar_gaba1', d3, a4, a5, a6, a7, gbar_gaba)
        cell.distribute_channels('soma', 'gbar_gaba1', 0, 1, 0, 0, 0, gbar_gaba)

        g_name = 'gaba1'
        g = []
        gbar = 'gbar_{}'.format(g_name)  
        for sec in cell.dendlist:
            g.append((eval('sec.{}'.format(gbar))))        
        
        if g[0] < g[-1]:
            cell.distribute_channels('soma', 'gbar_gaba1', 0, 1, 0, 0, 0, g[0])
        else:
            cell.distribute_channels('soma', 'gbar_gaba1', 0, 1, 0, 0, 0, gbar_gaba)

    
# Get dendrite branches, list for each unique branch structure (TODO: there's probably a neuron func for this)
def get_children(dend, branch_list):
    branch_list.append(dend)
    branches = []

    for child in dend.children():
        branch_list_cpy = branch_list.copy()
        branches.append(get_children(child, branch_list_cpy))

    if len(branches) == 0:
        return branch_list
    else:
        return branches
    

# Parser helper func
def branch_parser_helper(tree):
    for branch in tree:
        if all(type(b) == list for b in branch):
            # need to keep parsing
            for b in branch:
                tree.append(b)
            tree.remove(branch)
        # done parsing branch
    branch_parser(tree)
        
# Parses children into list format
def branch_parser(tree):
    for branch in tree:
        if all(type(b) == list for b in branch):
            branch_parser_helper(tree)
    return

# Takes nrn cell and int for origin dendrite segment index that branches occur from
# Returns parsed list with each entry a list of each unique branch path from origin dendrite segment to termination
def get_dend_branches_from(cell, origin):
    i = 0 
    for dend in cell.dendlist:
        if i == origin: # origin dendrite number to get branches from
            dend_tree = []
            dend_tree = get_children(dend, dend_tree)
            branch_parser(dend_tree)
            return dend_tree
        i += 1
        
def get_root_branches(cell):
    sref_soma = h.SectionRef(sec=cell.soma)

    # Get sec roots (excluding axon)
    roots = []

    for child in sref_soma.child:
        roots.append(child)

    roots = roots[1:]

    # Get dend tree from all roots
    root_tree = []

    for root in roots:
        dend_branch = []
        branch = get_children(root, dend_branch)
        branch_parser(branch)
        root_tree.append(branch)
    
    return root_tree

# gets path from dend to soma
def path_finder2(cell, dend_tree, dend):
    dend_tree2 = [num for sublist in dend_tree for num in sublist]
    for XX in dend_tree2:
        if not isinstance(XX, list):
            XX = [XX]
        for XXX in XX:
            if XXX == dend:
                return XX

def include_upto(iterable, value):
    for it in iterable:
        yield it
        if it == value:
            return

def path_finder(cell, dend_tree, dend):               
    pathlist = []
    pathlist = path_finder2(cell=cell, dend_tree=dend_tree, dend=dend)
    pathlist =  [cell.soma] + pathlist
    return list(include_upto(pathlist, dend))      
            
# Takes cell
# Return the dendrites that are in a branch (with root dendrite first followed by children in that branch ordered by first instance in tree)
# Useful if you want to do something to all dendrites in a branch, without the ordered duplication of root branches 
def get_root_groups(cell):
    root_tree = get_root_branches(cell)
    branch_groups = []
    for branch in root_tree:
        dend_list = []

        for dend in branch:
            
            if isinstance(dend, list):
                for d in dend:
                    if d not in dend_list:
                        dend_list.append(d)
            else:
                if dend not in dend_list:
                    dend_list.append(dend)
                    
        branch_groups.append(dend_list)
    return branch_groups

def nsegs(cell):
    nsegs =[]
    for sec in cell.dendlist:
        nsegs.append(sec.nseg)
    N = sum(nsegs)
    return(N)

def extract(d):
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    return y

def extract2(d):
    out = []
    for x in d:
        out.append(x)
    return out

def list2df(lst):
    df = pd.DataFrame() 
    df['time'] = extract2(lst[0])
    df['pas'] = extract2(lst[1])
    df['kdr'] = extract2(lst[2])
    df['naf'] = extract2(lst[3])
    df['kaf'] = extract2(lst[4])
    df['kas'] = extract2(lst[5])
    df['kir'] = extract2(lst[6])
    df['cal12'] = extract2(lst[7])
    df['cal13'] = extract2(lst[8])
    df['can'] = extract2(lst[9])
    df['car'] = extract2(lst[10])
    df['cav32'] = extract2(lst[11])
    df['cav33'] = extract2(lst[12])
    #     df['kcnq'] = extract2(lst[xx])
    df['sk'] = extract2(lst[13])
    df['bk'] = extract2(lst[14])  
    return df

def plot_mech(d, mech_name):
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    plt.title(mech_name)
    plt.plot(x, y)
    plt.show()
    
# return all dendritic inserted mechanisms
def mechanisms(cell):
    d_ = {}
    df = pd.DataFrame()
    # mechs = ['kdr', 'naf', 'kaf', 'kas', 'kdr', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kdr
    lists = sorted(d_.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples

    df['dist'] = x
    
    df['kdr'] = y

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_naf
    df['naf'] = extract(d_)

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kaf
    df['kaf'] = extract(d_)

    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kas
    df['kas'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kdr
    df['kdr'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_kir
    df['kir'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_cal12
    df['cal12'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_cal13
    df['cal13'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_can
    df['can'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_car
    df['car'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_cav32
    df['cav32'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.pbar_cav33
    df['cav33'] = extract(d_)


    # for sec in cell.dendlist:
    #      d_[h.distance(sec(0.5))] = sec.gbar_kcnq
    # mechanisms.append(extract(d_))
    for sec in cell.dendlist:      
         d_[h.distance(sec(0.5))] = sec.gbar_sk
    df['sk'] = extract(d_)


    for sec in cell.dendlist:
         d_[h.distance(sec(0.5))] = sec.gbar_bk
    df['bk'] = extract(d_)

    return(df)


# Set up branch assignment
def branch_selection(cell, cell_type='dpsn'):
    branch1_dends = [None] * 2 
    branch2_dends = [None] * 2 
    branch3_dends = [None] * 2 
    branch4_dends = [None] * 2 
    branch5_dends = [None] * 2 

    # Define dendrite sections for each predefined branch (chosen as stereotypical primary dendrites)
    if cell_type == 'dspn':    
        for dend in cell.dendlist:
            if dend.name() == 'dend[28]':
                branch1_dends[-1] = dend
            if dend.name() == 'dend[25]':
                branch1_dends[0] = dend

            if dend.name() == 'dend[15]':
                branch2_dends[-1] = dend
            if dend.name() == 'dend[13]':
                branch2_dends[0] = dend

            if dend.name() == 'dend[46]':
                branch3_dends[-1] = dend
            if dend.name() == 'dend[43]':
                branch3_dends[0] = dend

            if dend.name() == 'dend[36]':
                branch4_dends[-1] = dend
            if dend.name() == 'dend[32]':
                branch4_dends[0] = dend

            if dend.name() == 'dend[57]':
                branch5_dends[-1] = dend
            if dend.name() == 'dend[55]':
                branch5_dends[0] = dend

    elif cell_type == 'ispn':
        for dend in cell.dendlist:
            if dend.name() == 'dend[29]':
                branch1_dends[-1] = dend
            if dend.name() == 'dend[27]':
                branch1_dends[0] = dend

            if dend.name() == 'dend[15]':
                branch2_dends[-1] = dend
            if dend.name() == 'dend[13]':
                branch2_dends[0] = dend

            if dend.name() == 'dend[17]':
                branch3_dends[-1] = dend
            if dend.name() == 'dend[12]':
                branch3_dends[0] = dend

            if dend.name() == 'dend[45]':
                branch4_dends[-1] = dend
            if dend.name() == 'dend[41]':
                branch4_dends[0] = dend

            if dend.name() == 'dend[36]':
                branch5_dends[-1] = dend
            if dend.name() == 'dend[32]':
                branch5_dends[0] = dend
    
    # For sparse plotting
    return [branch1_dends] + [branch2_dends] + [branch3_dends] + [branch4_dends] + [branch5_dends]

# change all spine neck diameters
def spine_neck_diameter(cell, spines, diam):
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.neck.diam = diam

def spine_neck_length(cell, spines, length):
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.neck.L = length
            
def spine_head_diameter(cell, spines, diam, length):
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.head.diam = diam
            spine_obj.head.L = length
            
# Set up branch assignment and add glutamate
def glut_add(cell=None,
               branch1_glut = False, 
               branch2_glut = True, 
               branch3_glut = False, 
               branch4_glut = False, 
               branch5_glut = False, 
               num_gluts = 15,
               glut_placement = 'distal',
               glut = True,
               cell_type='dspn'):
    [branch1_dends, branch2_dends, branch3_dends, branch4_dends, branch5_dends] = branch_selection(cell, cell_type=cell_type) 
    glut_secs = []
    glut_secs_orig = []
    # Define placement on dendritic branch (prox/dist)
    if 'proximal' in glut_placement:
        glut_site = 0
    else:
        glut_site = -1

    # Define branch for glutamate (multiple possible)
    if branch1_glut:
        glut_secs.append(branch1_dends[glut_site])
        glut_secs_orig.append(branch1_dends[glut_site])

    if branch2_glut:
        glut_secs.append(branch2_dends[glut_site])
        glut_secs_orig.append(branch2_dends[glut_site])

    if branch3_glut:
        glut_secs.append(branch3_dends[glut_site])
        glut_secs_orig.append(branch3_dends[glut_site])

    if branch4_glut:
        glut_secs.append(branch4_dends[glut_site])
        glut_secs_orig.append(branch4_dends[glut_site])

    if branch5_glut:
        glut_secs.append(branch5_dends[glut_site])
        glut_secs_orig.append(branch5_dends[glut_site])

    # Number of glutamatergic inputs per section is num_gluts
    glut_secs *= num_gluts     

    if glut:
        print("glut:{}".format(glut_secs))
    else:
        # No glutamate
        glut_secs = []
    return glut_secs, glut_secs_orig

def glut_place(spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               num_gluts=15,
               return_currents = True,
               model = 1):
    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    if len(glut_secs) > 0: 
        glut_id = 0 # index used for glut_synapse list and printing
        final_spine_locs = []
        random.seed(42)
        for dend_glut in glut_secs:
            # Get possible spines from section
            candidate_spines = []
            sec_spines = list(spines[dend_glut.name()].items())

            if model in [1,2]:
            
                for spine_i, spine_obj in sec_spines: 
                    candidate_spines.append(spine_obj)

                if len(glut_secs) < len(sec_spines):
                    if method==1:
                        # reversed order so activate along dendrite towards soma
                        spine_idx = 2*len(candidate_spines)//3-1 # arbitrary start point at 2/3 of spines
                        spine = candidate_spines[spine_idx - glut_id] 
                    else:
                        spine_idx = 2*len(candidate_spines)//3 - num_gluts # arbitrary start point at 1/3 of spines
                        if spine_idx < 0:
                            if len(candidate_spines) >= num_gluts:
                                spine_idx = len(candidate_spines) - num_gluts
                            else:
                                spine_idx = 0        
                        spine = candidate_spines[spine_idx + glut_id] 
                else:
                    spine = random.choice(candidate_spines)
                    
            else:
            
                for spine_i, spine_obj in sec_spines: 
                    candidate_spines.append(spine_obj)
                if len(glut_secs) < len(sec_spines):
                    spine_idx = len(candidate_spines)//3-1 # arbitrary start point at 1/3 of spines
                    spine = candidate_spines[spine_idx + glut_id] 

                else:
                    spine = random.choice(candidate_spines)


            spine_loc = spine.x
            spine_head = spine.head
            final_spine_locs.append(spine_loc) 

            # Define glutamate syn 
            glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
            if physiological:
                if AMPA:
                    glut_synapses[glut_id].gmax_AMPA = g_AMPA
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0
                if NMDA:
                    glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                else:
                    glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                glut_synapses[glut_id].tau2_ampa = 4.8                
                # physiological kinetics for NMDA from Chapman et al. 2003, 
                # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                glut_synapses[glut_id].tau1_nmda = 5.52
                glut_synapses[glut_id].tau2_nmda = 231   
                # alpha and beta determine neg slope of Mg block for NMDA
                glut_synapses[glut_id].alpha = 0.096
                glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
            else:
                glut_synapses[glut_id].gmax_AMPA = 0.001 
                glut_synapses[glut_id].gmax_NMDA = 0.007
                # physiological kinetics for NMDA from Chapman et al. 2003, 
                # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                glut_synapses[glut_id].tau1_nmda = 5.52
                glut_synapses[glut_id].tau2_nmda = 231            

            # Stim to play back spike times as defined by onsets
            glut_stimulator[glut_id] = h.VecStim()
            glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

            # Connect stim and syn
            glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
            glut_connection[glut_id].weight[0] = 0.35

            if return_currents:
                # Record NMDA current for synapse
                nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

            glut_id += 1 # Increment glutamate counter

        print("# glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, final_spine_locs, glut_onsets))
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents

def glut_place_alt(spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               num_gluts=15,
               return_currents = True,
               model = 1):

    # Pairing each element in glut_secs with its corresponding onset in glut_onsets
    paired = list(zip(glut_secs, glut_onsets))

    # Sorting the pairs based on the dendrite element (the first item in each pair)
    sorted_pairs = sorted(paired, key=lambda x: str(x[0]))

    # Unpacking the sorted pairs back into separate lists
    glut_secs, glut_onsets = zip(*sorted_pairs)


    unique_glut_secs = list(set(glut_secs))

    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    if len(glut_secs) > 0: 
        glut_id = 0 # index used for glut_synapse list and printing
        final_spine_locs = []
        random.seed(42)

        for dend in unique_glut_secs:
            idx = [jj for jj, x in enumerate(glut_secs) if x.name() == dend.name()]
            selected_glut_secs = [glut_secs[i] for i in idx]
            id = 0
            for dend_glut in selected_glut_secs:
                # Get possible spines from section
                candidate_spines = []
                sec_spines = list(spines[dend_glut.name()].items())

                if model in [1,2]:

                    for spine_i, spine_obj in sec_spines: 
                        candidate_spines.append(spine_obj)

                    if len(glut_secs) < len(sec_spines):
                        if method==1:
                            # reversed order so activate along dendrite towards soma
                            spine_idx = 2*len(candidate_spines)//3-1 # arbitrary start point at 2/3 of spines
                            spine = candidate_spines[spine_idx - id] 
                        else:
                            spine_idx = 2*len(candidate_spines)//3 - num_gluts # arbitrary start point at 1/3 of spines
                            if spine_idx < 0:
                                if len(candidate_spines) >= num_gluts:
                                    spine_idx = len(candidate_spines) - num_gluts
                                else:
                                    spine_idx = 0        
                            spine = candidate_spines[spine_idx + id] 
                    else:
                        spine = random.choice(candidate_spines)

                else:

                    for spine_i, spine_obj in sec_spines: 
                        candidate_spines.append(spine_obj)
                    if len(glut_secs) < len(sec_spines):
                        spine_idx = len(candidate_spines)//3-1 # arbitrary start point at 1/3 of spines
                        spine = candidate_spines[spine_idx + id] 

                    else:
                        spine = random.choice(candidate_spines)

                spine_loc = spine.x
                spine_head = spine.head
                final_spine_locs.append(spine_loc) 

                # Define glutamate syn 
                glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
                if physiological:
                    if AMPA:
                        glut_synapses[glut_id].gmax_AMPA = g_AMPA
                    else:
                        glut_synapses[glut_id].gmax_AMPA = 0
                    if NMDA:
                        glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                    else:
                        glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                    # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                    glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                    glut_synapses[glut_id].tau2_ampa = 4.8                
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231   
                    # alpha and beta determine neg slope of Mg block for NMDA
                    glut_synapses[glut_id].alpha = 0.096
                    glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0.001 
                    glut_synapses[glut_id].gmax_NMDA = 0.007
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231            

                # Stim to play back spike times as defined by onsets
                glut_stimulator[glut_id] = h.VecStim()
                glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

                # Connect stim and syn
                glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
                glut_connection[glut_id].weight[0] = 0.35

                if return_currents:
                    # Record NMDA current for synapse
                    nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                    ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

                glut_id += 1 # Increment glutamate counter
                id += 1

        rounded_locs = [round(value, 4) for value in final_spine_locs]
        print("# glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents

# finds distances of syanpses from soma
def synapse_dist(spines,
               method=0,
               glut_secs = None,
               num_gluts=15):
    if len(glut_secs) > 0: 
        glut_id = 0 # index used for glut_synapse list and printing
        final_spine_dists = []
        random.seed(42)
        for dend_glut in glut_secs:
            # Get possible spines from section
            candidate_spines = []
            sec_spines = list(spines[dend_glut.name()].items())

            for spine_i, spine_obj in sec_spines: 
                candidate_spines.append(spine_obj)

            if len(glut_secs) < len(sec_spines):
                if method==1:
                    # reversed order so activate along dendrite towards soma
                    spine_idx = 2*len(candidate_spines)//3-1 # arbitrary start point at 2/3 of spines
                    spine = candidate_spines[spine_idx - glut_id] 
                else:
                    spine_idx = 2*len(candidate_spines)//3 - num_gluts # arbitrary start point at 1/3 of spines
                    if spine_idx < 0:
                        if len(candidate_spines) >= num_gluts:
                            spine_idx = len(candidate_spines) - num_gluts
                        else:
                            spine_idx = 0        
                    spine = candidate_spines[spine_idx + glut_id] 
            else:
                spine = random.choice(candidate_spines)

            spine_loc = spine.x
            final_spine_dists.append(h.distance(dend_glut(spine_loc))) 
            glut_id += 1 # Increment glutamate counter
    return final_spine_dists

def gaba_onset(gaba_time, num_gabas, num_branch2, model=1):
    if model == 0:
        gaba_onsets = list(range(gaba_time, gaba_time + int(num_gabas/3)+1)) * 3 * num_branch2
        gaba_onsets = gaba_onsets[:num_gabas]
    else:
        if num_branch2 in [0,1]:
            if (num_gabas < 4):
                gaba_onsets = list(range(gaba_time, gaba_time + num_gabas)) 
            else:
                if num_gabas % 3 == 0:
                    gaba_onsets = list(range(gaba_time, gaba_time + int(num_gabas/3))) * 3 * num_branch2
                else:
                    gaba_onsets = list(range(gaba_time, gaba_time + int(num_gabas/3)+1)) * 3 * num_branch2
            gaba_onsets = gaba_onsets[:num_gabas]
        else:
            onsets = list(range(gaba_time, gaba_time + num_gabas)) 
            gaba_onsets = [x for x in onsets for _ in range(num_branch2)]
    return gaba_onsets


def gaba_add(cell=None,
               gaba=True, 
               branch1_gaba = False, 
               branch2_gaba = True, 
               branch3_gaba = False, 
               branch4_gaba = False, 
               branch5_gaba = False, 
               gaba_placement = 'distal',
               num_gabas=15,
               show=True,
               cell_type='dspn'):

    if gaba > 0: 
        [branch1_dends, branch2_dends, branch3_dends, branch4_dends, branch5_dends] = branch_selection(cell, cell_type) 

        gaba_secs = []

        # Define gaba spatial placement 
        if 'soma' in gaba_placement:
            gaba_secs.append(cell.soma)

            gaba_secs *= num_gabas # need to duplicate sections to place synapses 

        elif 'everywhere' in gaba_placement: # append to every dendrite section
            for dend in cell.dendlist:
                gaba_secs.append(dend)

            gaba_secs *= num_gabas # need to duplicate sections to place synapses 


        elif 'distributed_branch' in gaba_placement: # append to specific branches

            # Define placement on dendritic branch (prox/dist)
            if 'proximal' in gaba_placement:
                gaba_site = 0
            else:
                gaba_site = -1

            # Define branch for gaba (multiple possible)
            if branch1_gaba:
                gaba_secs.append(branch1_dends[gaba_site])

            if branch2_gaba:
                gaba_secs.append(branch2_dends[gaba_site])

            if branch3_gaba:
                gaba_secs.append(branch3_dends[gaba_site])

            if branch4_gaba:
                gaba_secs.append(branch4_dends[gaba_site])

            if branch5_gaba:
                gaba_secs.append(branch5_dends[gaba_site])

            gaba_secs *= num_gabas # need to duplicate sections to place synapses 

    else:
        # No gaba
        gaba_secs = []

    if show:
        print("gaba:{}".format(gaba_secs))
    return gaba_secs

def gaba_place(physiological=True,
               gaba_reversal = -60,
               gaba_weight = 0.001,
               gaba_time = 200,
               gaba_secs = None,
               gaba_onsets=None,
               gaba_locations = None,
               num_gabas=15,
               return_currents = True,
               show=True):
    
    gaba_conductances = [0] * len(gaba_secs)
    gaba_currents = [0] * len(gaba_secs)
    gaba_synapses = [0]*len(gaba_secs) # list of gaba synapses
    gaba_stimulator = {}
    gaba_connection = {}
    if gaba_locations is None:
        gaba_locations = [0.5] * len(gaba_secs)
        
    # Place gabaergic synapses
    if len(gaba_secs) > 0:

        gaba_id = 0 # index used for gaba_synapse list and printing
        gaba_locs = []

        for dend_gaba in gaba_secs:

            # For now, just assign to middle of section instead of uniform random
            gaba_loc = gaba_locations[gaba_id]

            # Choose random location along section
    #                 gaba_loc = round(random.uniform(0, 1), 2)

            gaba_locs.append(gaba_loc)

            # Define gaba synapse
            gaba_synapses[gaba_id] = h.gabasynapse(dend_gaba(gaba_loc)) 
            if physiological:
                gaba_synapses[gaba_id].tau1 = 0.9 
                gaba_synapses[gaba_id].tau2 = 18
            else:
                gaba_synapses[gaba_id].tau2 = 0.9 # TODO: Tune tau2 further for accurate response 
            gaba_synapses[gaba_id].erev = gaba_reversal

            # Stim to play back spike times
            gaba_stimulator[gaba_id] = h.VecStim()

            # Use with deterministic onset times
            gaba_stimulator[gaba_id].play(h.Vector(1, gaba_onsets[gaba_id]))

            # Connect stim and syn
            gaba_connection[gaba_id] = h.NetCon(gaba_stimulator[gaba_id], gaba_synapses[gaba_id])
            gaba_connection[gaba_id].weight[0] = gaba_weight # Depending on desired EPSP response at soma, tune this

            if return_currents:
                # Measure conductance and current
                gaba_currents[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_i)
                gaba_conductances[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_g)

            gaba_id += 1 # increment gaba counter

        if show:
            print("# gaba synapses added:{} on:{} with locs:{} with timing onsets:{}".format(gaba_id, gaba_secs, gaba_locs, gaba_onsets))
    return gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances

def glut_place2(cell,
               spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut=True,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               glut_locs = None,
               num_gluts=15,
               return_currents = True):
    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    final_spine_locs = []
    final_spines = []
    if num_gluts > 0: 
        glut_id = 0 # index used for glut_synapse list and printing

        for ii in list(range(0,num_gluts)):
            synapse_loc = glut_locs[ii]
            # Get candidate spines from section
            candidates = []
            sec_spines = list(spines[glut_secs[ii].name()].items())

            for spine_i, spine_obj in sec_spines: 
                candidates.append(spine_obj)

            locs = []
            for spine in candidates:
                locs.append(spine.x)
            loc, idx = find_closest_value(locs, synapse_loc)
            spine = candidates[idx] # choose last spine

            spine_loc = spine.x
            spine_head = spine.head
            final_spine_locs.append(spine_loc) 
            final_spines.append(spine)
            if glut:
                # Define glutamate syn 
                glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
                if physiological:
                    if AMPA:
                        glut_synapses[glut_id].gmax_AMPA = g_AMPA
                    else:
                        glut_synapses[glut_id].gmax_AMPA = 0
                    if NMDA:
                        glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                    else:
                        glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                    # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                    glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                    glut_synapses[glut_id].tau2_ampa = 4.8                
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231   
                    # alpha and beta determine neg slope of Mg block for NMDA
                    glut_synapses[glut_id].alpha = 0.096
                    glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0.001 
                    glut_synapses[glut_id].gmax_NMDA = 0.007
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231            

                # Stim to play back spike times as defined by onsets
                glut_stimulator[glut_id] = h.VecStim()
                glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

                # Connect stim and syn
                glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
                glut_connection[glut_id].weight[0] = 0.35

                if return_currents:
                    # Record NMDA current for synapse
                    nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                    ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

            glut_id += 1 # Increment glutamate counter

#         rounded_locs = [round(value, 4) for value in glut_locs]
        rounded_locs = [round(value, 4) for value in final_spine_locs]
        if glut:
            print("# glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs

def glut_place3(cell,
               spines,
               method=0, 
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut=True,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               glut_locs = None,
               num_gluts=15,
               return_currents = True,
               axoshaft=False):
    
    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    final_spine_locs = []
    final_spines = []
    if num_gluts > 0: 
        glut_id = 0 # index used for glut_synapse list and printing

        for ii in list(range(0,num_gluts)):
            synapse_loc = glut_locs[ii]
            # Get candidate spines from section
            candidates = []
            sec_spines = list(spines[glut_secs[ii].name()].items())

            for spine_i, spine_obj in sec_spines: 
                candidates.append(spine_obj)

            locs = []
            for spine in candidates:
                locs.append(spine.x)
            loc, idx = find_closest_value(locs, synapse_loc)
            spine = candidates[idx] # choose last spine

            spine_loc = spine.x
            spine_head = spine.head
            final_spine_locs.append(spine_loc) 
            final_spines.append(spine)
            if glut:
                # Define glutamate syn 
                if not axoshaft:
                    glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
                else:
                    glut_synapses[glut_id] = h.glutsynapse(glut_secs[ii](glut_locs[ii]))
                if physiological:
                    if AMPA:
                        glut_synapses[glut_id].gmax_AMPA = g_AMPA
                    else:
                        glut_synapses[glut_id].gmax_AMPA = 0
                    if NMDA:
                        glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                    else:
                        glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                    # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                    glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                    glut_synapses[glut_id].tau2_ampa = 4.8                
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231   
                    # alpha and beta determine neg slope of Mg block for NMDA
                    glut_synapses[glut_id].alpha = 0.096
                    glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0.001 
                    glut_synapses[glut_id].gmax_NMDA = 0.007
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231            

                # Stim to play back spike times as defined by onsets
                glut_stimulator[glut_id] = h.VecStim()
                glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

                # Connect stim and syn
                glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
                glut_connection[glut_id].weight[0] = 0.35

                if return_currents:
                    # Record NMDA current for synapse
                    nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                    ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

            glut_id += 1 # Increment glutamate counter

#         rounded_locs = [round(value, 4) for value in glut_locs]
        rounded_locs = [round(value, 4) for value in final_spine_locs]
        if glut:
            if not axoshaft:
                print("# axospinous glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
            else:
                print("# axoshaft glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
    
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs

def glut_place4(cell,
               spines,
               physiological=True, 
               AMPA=True, 
               g_AMPA = 0.001,
               NMDA=True,
               ratio = 2,
               glut_reversals=[0]*15, 
               glut=True,
               glut_time = 200,
               glut_secs = None,
               glut_onsets=None,
               glut_locs = None,
               num_gluts=15,
               return_currents = True,
               axoshaft=False):
    
    nmda_currents = [None]*len(glut_secs)
    ampa_currents = [None]*len(glut_secs)
    glut_synapses = [0]*len(glut_secs)
    glut_stimulator = {}
    glut_connection = {}
    final_spine_locs = []
    final_spines = []
    final_spine_secs = [] 

    if num_gluts > 0: 
        glut_id = 0 # index used for glut_synapse list and printing

        for ii in list(range(0,num_gluts)):
            synapse_loc = glut_locs[ii]
            # Get candidate spines from section
            candidates = []
            sec_spines = list(spines[glut_secs[ii].name()].items())

            for spine_i, spine_obj in sec_spines: 
                candidates.append(spine_obj)

            locs = []
            for spine in candidates:
                locs.append(spine.x)
            loc, idx = find_closest_value(locs, synapse_loc)
            spine = candidates[idx] # choose last spine

            spine_loc = spine.x
            spine_head = spine.head
            final_spine_locs.append(spine_loc) 
            final_spines.append(spine)
            final_spine_secs.append(glut_secs[ii].name())
            if glut:
                # Define glutamate syn 
                if not axoshaft:
                    glut_synapses[glut_id] = h.glutsynapse(spine_head(0.5))
                else:
                    glut_synapses[glut_id] = h.glutsynapse(glut_secs[ii](glut_locs[ii]))
                if physiological:
                    if AMPA:
                        glut_synapses[glut_id].gmax_AMPA = g_AMPA
                    else:
                        glut_synapses[glut_id].gmax_AMPA = 0
                    if NMDA:
                        glut_synapses[glut_id].gmax_NMDA = g_AMPA*ratio # 
                    else:
                        glut_synapses[glut_id].gmax_NMDA = 0 # NMDA:AMPA ratio is 0.5
                    # values from Ding et al., 2008; AMPA decay value similar in Kreitzer & Malenka, 2007
                    glut_synapses[glut_id].tau1_ampa = 0.86 # 10-90% rise 1.9; tau = 1.9/2.197
                    glut_synapses[glut_id].tau2_ampa = 4.8                
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    # values from Kreitzer & Malenka, 2007 are 2.5 and 50 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231   
                    # alpha and beta determine neg slope of Mg block for NMDA
                    glut_synapses[glut_id].alpha = 0.096
                    glut_synapses[glut_id].beta = 17.85  # ie 5*3.57  
                else:
                    glut_synapses[glut_id].gmax_AMPA = 0.001 
                    glut_synapses[glut_id].gmax_NMDA = 0.007
                    # physiological kinetics for NMDA from Chapman et al. 2003, 
                    # NMDA decay is weighted average of fast and slow 231 +- 5 ms
                    # rise time 10-90% is 12.13 ie tau = 12.13 / 2.197 
                    glut_synapses[glut_id].tau1_nmda = 5.52
                    glut_synapses[glut_id].tau2_nmda = 231            
               
                glut_synapses[glut_id].erev = glut_reversals[ii]

                # Stim to play back spike times as defined by onsets
                glut_stimulator[glut_id] = h.VecStim()
                glut_stimulator[glut_id].play(h.Vector(1, glut_onsets[glut_id]))

                # Connect stim and syn
                glut_connection[glut_id] = h.NetCon(glut_stimulator[glut_id], glut_synapses[glut_id])
                glut_connection[glut_id].weight[0] = 0.35

                if return_currents:
                    # Record NMDA current for synapse
                    nmda_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_nmda)
                    ampa_currents[glut_id] = h.Vector().record(glut_synapses[glut_id]._ref_i_ampa)

            glut_id += 1 # Increment glutamate counter

#         rounded_locs = [round(value, 4) for value in glut_locs]
        rounded_locs = [round(value, 4) for value in final_spine_locs]
        if glut:
            if not axoshaft:
                print("# axospinous glutamate added:{}, on sections:{}, with final spine locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
            else:
                print("# axoshaft glutamate added:{}, on sections:{}, with locs:{} with timing onsets:{}".format(glut_id, glut_secs, rounded_locs, glut_onsets))
    
    return glut_synapses, glut_stimulator, glut_connection, ampa_currents, nmda_currents, final_spines, final_spine_locs, final_spine_secs

def gaba_place2(physiological=True,
               gaba_reversal = -60,
               gaba_weight = 0.001,
               gaba_time = 200,
               gaba_secs = None,
               gaba_onsets=None,
               gaba_locations = None,
               num_gabas=15,
               return_currents = True,
               show=True):
    
    gaba_conductances = [0] * len(gaba_secs)
    gaba_currents = [0] * len(gaba_secs)
    gaba_synapses = [0]*len(gaba_secs) # list of gaba synapses
    gaba_stimulator = {}
    gaba_connection = {}
    if gaba_locations is None:
        gaba_locations = [0.5] * len(gaba_secs)
    gaba_locs = []    
    # Place gabaergic synapses
    if len(gaba_secs) > 0:

        gaba_id = 0 # index used for gaba_synapse list and printing
        for dend_gaba in gaba_secs:

            # For now, just assign to middle of section instead of uniform random
            gaba_loc = gaba_locations[gaba_id]

            # Choose random location along section
    #                 gaba_loc = round(random.uniform(0, 1), 2)

            gaba_locs.append(gaba_loc)

            # Define gaba synapse
            gaba_synapses[gaba_id] = h.gabasynapse(dend_gaba(gaba_loc)) 
            if physiological:
                gaba_synapses[gaba_id].tau1 = 0.9 
                gaba_synapses[gaba_id].tau2 = 18
            else:
                gaba_synapses[gaba_id].tau2 = 0.9 # TODO: Tune tau2 further for accurate response 
            gaba_synapses[gaba_id].erev = gaba_reversal

            # Stim to play back spike times
            gaba_stimulator[gaba_id] = h.VecStim()

            # Use with deterministic onset times
            gaba_stimulator[gaba_id].play(h.Vector(1, gaba_onsets[gaba_id]))

            # Connect stim and syn
            gaba_connection[gaba_id] = h.NetCon(gaba_stimulator[gaba_id], gaba_synapses[gaba_id])
            gaba_connection[gaba_id].weight[0] = gaba_weight # Depending on desired EPSP response at soma, tune this

            if return_currents:
                # Measure conductance and current
                gaba_currents[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_i)
                gaba_conductances[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_g)

            gaba_id += 1 # increment gaba counter

        rounded_locs = [round(value, 4) for value in gaba_locs]
        print("# gaba synapses added:{} on:{} with locs:{} with timing onsets:{}".format(gaba_id, gaba_secs, gaba_locs, gaba_onsets))
    return gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances, gaba_locs

def gaba_place3(physiological=True,
               gaba_reversals = [-60]*15,
               gaba_weight = 0.001,
               gaba_tau1=0.9,
               gaba_tau2=18,
               gaba_time = 200,
               gaba_secs = None,
               gaba_onsets=None,
               gaba_locations = None,
               num_gabas=15,
               return_currents = True,
               show=True):
    
    gaba_conductances = [0] * len(gaba_secs)
    gaba_currents = [0] * len(gaba_secs)
    gaba_synapses = [0]*len(gaba_secs)
    gaba_stimulator = {}
    gaba_connection = {}
    if gaba_locations is None:
        gaba_locations = [0.5] * len(gaba_secs)
    gaba_locs = []    

    if len(gaba_secs) > 0:

        gaba_id = 0
        for dend_gaba, gaba_reversal in zip(gaba_secs, gaba_reversals):

            gaba_loc = gaba_locations[gaba_id]
            gaba_locs.append(gaba_loc)

            gaba_synapses[gaba_id] = h.gabasynapse(dend_gaba(gaba_loc)) 
            if physiological:
                tau1 = gaba_tau1[gaba_id] if isinstance(gaba_tau1, (list, np.ndarray)) else gaba_tau1
                tau2 = gaba_tau2[gaba_id] if isinstance(gaba_tau2, (list, np.ndarray)) else gaba_tau2
                gaba_synapses[gaba_id].tau1 = tau1
                gaba_synapses[gaba_id].tau2 = tau2
            else:
                gaba_synapses[gaba_id].tau2 = 0.9
            gaba_synapses[gaba_id].erev = gaba_reversal

            gaba_stimulator[gaba_id] = h.VecStim()
            gaba_stimulator[gaba_id].play(h.Vector(1, gaba_onsets[gaba_id]))

            gaba_connection[gaba_id] = h.NetCon(gaba_stimulator[gaba_id], gaba_synapses[gaba_id])
            weight = gaba_weight[gaba_id] if isinstance(gaba_weight, (list, np.ndarray)) else gaba_weight
            gaba_connection[gaba_id].weight[0] = weight

            if return_currents:
                gaba_currents[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_i)
                gaba_conductances[gaba_id] = h.Vector().record(gaba_synapses[gaba_id]._ref_g)

            gaba_id += 1

        rounded_locs = [round(value, 4) for value in gaba_locs]
        tau1_vals = list(gaba_tau1) if isinstance(gaba_tau1, (list, np.ndarray)) else [gaba_tau1]*gaba_id
        tau2_vals = list(gaba_tau2) if isinstance(gaba_tau2, (list, np.ndarray)) else [gaba_tau2]*gaba_id
        print("# gaba synapses added:{} on:{} with locs:{} tau1:{} tau2:{} with timing onsets:{}".format(
            gaba_id, gaba_secs, rounded_locs, tau1_vals, tau2_vals, gaba_onsets))

    return gaba_synapses, gaba_stimulator, gaba_connection, gaba_currents, gaba_conductances, gaba_locs

def count_unique_dends(input_list):
    unique_names = set(input_list)
    count = len(unique_names)
    return count

# Records voltage across all sections
def record_all_v(cell, loc=0.4):
    all_v = {}
    for sec in cell.allseclist:
        all_v[sec.name()] = h.Vector()
        all_v[sec.name()].record(sec(loc)._ref_v) # given a sec with multiple seg, 
    return all_v

# Records voltage across selected sections
def record_v(cell, seclist, loc=0.4):
    all_v = {}
    for sec in seclist:
        all_v[sec.name()] = h.Vector()
        all_v[sec.name()].record(sec(loc)._ref_v) # given a sec with multiple seg, 
    return all_v


# function gets all unique locations on path to soma
def record_all_path_secs_v(cell, dend_tree, dendrite):
    all_v = {}
    dists = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
    # get path to soma
    if dendrite.name() != 'soma[0]':
        pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    else:
        pathlist = [dendrite]
    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        for seg in sec:
            dist = h.distance(seg)
            dists.append(dist)
            loc = seg.x
            all_v[i] = h.Vector()
            all_v[i].record(sec(loc)._ref_v) # given a sec with multiple seg
            i = i + 1
    return all_v, dists

def record_all_path_secs_v2(cell, dend_tree, dendrite):
    all_v = {}
    dists = []
    dends = []
    i = 0

    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend

    # Get path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)

    # Record data for each dendrite in the path
    for sec in pathlist:
        all_v, i, sec_dends, sec_dists = sec_all_v(sec, all_v, i)
        dends.extend(sec_dends)
        dists.extend(sec_dists)

    return all_v, dists, dends


def sec_all_v(section, all_v, i):
    """
    Records data from each segment in a given section.

    Args:
        section (object): The neuron section to record from.
        all_v (dict): Dictionary to store recorded vectors.
        i (int): Index for storing in the dictionary.

    Returns:
        tuple: Updated dictionary `all_v`, updated index `i`, list of section names, and list of distances.
    """
    dends = []
    dists = []

    for seg in section:
        dends.append(section.name())
        dist = h.distance(seg)
        dists.append(dist)
        loc = seg.x
        all_v[i] = h.Vector()
        all_v[i].record(section(loc)._ref_v)  # Record voltage at this segment
        i += 1

    return all_v, i, dends, dists

def unique_path_secs_v_old(cell_type, specs, dendrite, spine_per_length=1.61, soma_diameter=None, frequency=2000, d_lambda=0.05, axospine=True, n=1, dend2remove=None, neck_dynamics=False):
    """
    Gets all unique locations on the path to soma for a given cell type and dendrite.
    
    Parameters:
        cell_type (str): The type of the cell.
        specs (Dict): The specifications for the cell.
        dendrite_name (str): The name of the dendrite.
        axospiny (bool): Whether to include dendrites with at least n spines.
        n (int): Minimum number of spines required for a dendrite to be included.
               
    Returns:
        three lists - dends, locs, and dists.
    """
        # Build cell
    cell, spines, dend_tree, branch_groups= cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, branch=True, dend2remove=dend2remove, neck_dynamics=neck_dynamics)
    
    # Identify the target dendrite
    dendrite = next((dend for dend in cell.allseclist if dend.name() == dendrite), None)
    if dendrite is None:
        raise ValueError(f"No dendrite found with name: {dendrite}")
    
    # Find path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    
    # determine which dendrites have at least n=1 spines
    if axospine:
        dends_required = dend_spine_selector(cell=cell, spines=spines, branch_groups=branch_groups, n=n)
    else:
        dends_required = pathlist
    
    dends, locs, dists = [], [], []
    
    # Iterate through each dendrite in the path and find unique locations corresponding to each segment
    for sec in pathlist:
        if sec in dends_required:
            for seg in sec:
                dist = h.distance(seg)
                dists.append(dist)
                locs.append(seg.x)
                dends.append(sec.name())
            
    return dends[::-1], locs[::-1], dists[::-1]

# returns locations of all spines        
def spine_locations(pathlist, dends_required, spines):
    spine_dends = []
    spine_locs = []
    spine_dists = []

    for sec in pathlist:
        if sec in dends_required:
            names = list(spines[sec.name()].keys())
            for name in names:
                spine_dends.append(sec.name())
                loc = spines[sec.name()][name].x
                spine_locs.append(loc)
                spine_dists.append(h.distance(sec(loc)))

    return spine_dends[::-1], spine_locs[::-1], spine_dists[::-1]

                
def shaft_locs(pathlist):
    dends = []
    locs = []
    dists = []
    # Iterate through each dendrite in the path and find unique locations corresponding to each segment
    for sec in pathlist:
        for seg in sec:
            dist = h.distance(seg)
            dists.append(dist)
            locs.append(seg.x)
            dends.append(sec.name())
    return dends[::-1], locs[::-1], dists[::-1]


def unique_path_secs_v(cell_type, specs, dendrite, spine_per_length=1.61, soma_diameter=None, frequency=2000, d_lambda=0.05, axospine=True, n=1, dend2remove=None, neck_dynamics=False):
    """
    gets all unique locations on the path to soma for a given cell type and dendrite.
    
    parameters:
        cell_type (str): The type of the cell.
        specs (Dict): The specifications for the cell.
        dendrite_name (str): The name of the dendrite.
        axospiny (bool): Whether to include dendrites with at least n spines.
        n (int): Minimum number of spines required for a dendrite to be included.
               
    Returns:
        three lists - dends, locs, and dists.
    """
    # Build cell
    cell, spines, dend_tree, branch_groups= cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, branch=True, dend2remove=dend2remove, neck_dynamics=neck_dynamics)

    # Identify the target dendrite
    dendrite = next((dend for dend in cell.allseclist if dend.name() == dendrite), None)
    if dendrite is None:
        raise ValueError(f"No dendrite found with name: {dendrite}")

    # Find path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)

    # Find path to soma
    pathlist = [dendrite] if dendrite.name() == 'soma[0]' else path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)

    # determine which dendrites have at least n=1 spines
    dends, locs, dists = shaft_locs(pathlist) # all unique locations

    if axospine:
        dends_required = dend_spine_selector(cell=cell, spines=spines, branch_groups=branch_groups, n=n)
        spine_dends, spine_locs, spine_dists = spine_locations(pathlist=pathlist, dends_required=dends_required, spines=spines)

        idxs = []
        # Using zip to loop through targets and dend_names in pairs
        for target, dend_name in zip(locs, dends):
            idx = find_closest_loc_index(spine_locs, spine_dends, target, dend_name)
            idxs.append(idx)


        idxs = [item for item in idxs if str(item).startswith('No locations') is False]

        # remove duplicates
        seen = set()
        idxs2 = []

        for idx in idxs:
            if idx not in seen:
                idxs2.append(idx)
                seen.add(idx)

        dends, locs, dist = [], [], []
        for idx in idxs2:
            locs.append(spine_locs[idx])
            dends.append(spine_dends[idx])
            dist.append(spine_dists[idx])

    return dends, locs, dists

def find_closest_loc_index(locs, dends, target, dend_name):
    # Filter the locs for the specified dend_name and keep track of the original indices
    filtered_locs = [(loc, index) for index, (loc, dend) in enumerate(zip(locs, dends)) if dend == dend_name]
    
    # If there are no matches, return a message
    if not filtered_locs:
        return f"No locations found for {dend_name}"
    
    # Find the index of the loc closest to the target
    closest_loc, closest_index = min(filtered_locs, key=lambda x: abs(x[0] - target))
    
    return closest_index


# find location in dend that is closest to dend_gaba with gaba_locations = [0.5] 
def find_closest_loc(locs, dends, target, dend_name):
    # Filter the locs for the specified dend_name
    filtered_locs = [loc for loc, dend in zip(locs, dends) if dend == dend_name]
    
    # If there are no matches, return a message
    if not filtered_locs:
        return f"No locations found for {dend_name}"
    
    # Find the loc closest to the target
    closest_loc = min(filtered_locs, key=lambda x: abs(x - target))
    
    return closest_loc

# Records Cai across selected sections
def record_cai(cell, seclist, loc=0.4, return_Ca=True):
    all_cai = {}
    if return_Ca:
        for sec in seclist:
            all_cai[sec.name()] = h.Vector()
            all_cai[sec.name()].record(sec(loc)._ref_cai) # given a sec with multiple seg, 
    return all_cai

# Returns vectors for impedance recording
def record_impedance(dend, loc=0.4):
    imp = h.Impedance()
    # imp.loc(0.5, sec=cell.soma) 
    # define location either for current stim or voltage measuring electrode
    # this is needed for the transfer impedance calculation
    imp.loc(loc, sec=dend) # location of interest; nb voltages are measured at 0.4 ; not necessary if computing imp.input()  
    zvec1 = h.Vector()  
    zvec1.append(0)
    zvec2 = h.Vector()  
    zvec2.append(0)
    return imp, zvec1, zvec2

def record_i_mechs(cell, dend, loc=0.4, return_currents=True, silent=False, mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kcnq', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']):
    i_mechs_out = []
    if return_currents:
        # Record time vector
        t = h.Vector().record(h._ref_t)
        i_mechs_out.append(t)

        if not silent: 
            print("i_mechanisms recorded in {}".format(dend))

        # Predefined dictionary for mechanism references
        mech_refs = {
            'pas': '_ref_i_pas',
            'kdr': '_ref_ik_kdr',
            'naf': '_ref_ina_naf',
            'kaf': '_ref_ik_kaf',
            'kas': '_ref_ik_kas',
            'kcnq': '_ref_ik_kcnq',
            'kir': '_ref_ik_kir',
            'cal12': '_ref_ical_cal12',
            'cal13': '_ref_ical_cal13',
            'can': '_ref_ica_can',
            'car': '_ref_ica_car',
            'cav32': '_ref_ical_cav32',
            'cav33': '_ref_ical_cav33',
            'sk': '_ref_ik_sk',
            'bk': '_ref_ik_bk'
        }

        # Loop over requested mechanisms
        for mech in mechs:
            ref_attr = mech_refs.get(mech)
            if ref_attr and hasattr(dend(loc), ref_attr):
                # Record the mechanism if it exists
                record_vector = h.Vector().record(getattr(dend(loc), ref_attr))
                i_mechs_out.append(record_vector)
            else:
                print(f"warning: mechanism '{mech}' not recognized or not present at the specified location")

        return i_mechs_out
    
    
# for plotting, returns all branch dendrites
def dend2plot(cell, cell_type='dspn'):
    [branch1_dends, branch2_dends, branch3_dends, branch4_dends, branch5_dends] = branch_selection(cell, cell_type) 
    branch_dends = [branch1_dends] + [branch2_dends] + [branch3_dends] + [branch4_dends] + [branch5_dends]
    branch_dends = [num for sublist in branch_dends for num in sublist]
    return [cell.soma] + branch_dends

def plot1(cell=None, dend=None, t=None, v=None, seclist=None, sparse=False, protocol=''):
    import plotly.graph_objects as go
    v_data = []
    if sparse:
        for group in seclist: # for each nrn dendrite sec, one plot per branch
            if dend in group: # Use if you want sparse plotting
                for sec in group:
                    v_data.append(go.Scatter(x=t, y=v[sec.name()], name='{}:{}'.format(sec.name(), round(h.distance(sec(0.5)), 2))))
                v_data.append(go.Scatter(x=t, y=v['soma[0]'], name='soma'))
    else:
        for sec in seclist: # for each nrn dendrite sec, one plot per branch
            v_data.append(go.Scatter(x=t, y=v[sec.name()], name='{}:{}'.format(sec.name(), round(h.distance(sec(0.5)), 2))))
    
    # Plot vdata
    fig = go.Figure(data=v_data)
    fig.update_layout(
        title="{}".format(protocol),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title="V (mV)",
        legend_title="section")
    return fig   

def plot1_Ca(cell=None, dend=None, t=None, Ca=None, seclist=None, sparse=False, protocol=''):
    import plotly.graph_objects as go
    Ca_data = []
    if sparse:
        for group in seclist: # for each nrn dendrite sec, one plot per branch
            if dend in group: # Use if you want sparse plotting
                for sec in group:
                    Ca_data.append(go.Scatter(x=t, y=Ca[sec.name()]*1e3, name='{}:{}'.format(sec.name(), round(h.distance(sec(0.5)), 2))))
                Ca_data.append(go.Scatter(x=t, y=Ca['soma[0]']*1e3, name='soma'))
    else:
        for sec in seclist: # for each nrn dendrite sec, one plot per branch
            Ca_data.append(go.Scatter(x=t, y=Ca[sec.name()]*1e3, name='{}:{}'.format(sec.name(), round(h.distance(sec(0.5)), 2))))
    
    # Plot vdata
    fig = go.Figure(data=Ca_data)
    fig.update_layout(
        title="{}".format(protocol),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title="[Ca] (uM)",
        legend_title="section")
    return fig   


def plot2(soma_v_data, dend_v_data, glut_placement=None, yaxis='V (mV)'):
    # import plotly.graph_objects as go
    if yaxis=='V (mV)':
        title1='soma PSP'
        if glut_placement == 'distal':
            title2 = 'distal dendrite PSP'
        elif glut_placement == 'proximal':
            title2 = 'proximal dendrite PSP'
        else:
            title2 = 'dendrite PSP'
    else:
        title1='soma impedance'
        if glut_placement == 'distal':
            title2 = 'distal dendrite impedance'
        elif glut_placement == 'proximal':
            title2 = 'proximal dendrite impedance'
        else:
            title2 = 'dendrite impedance'
            
    fig_soma = go.Figure(data=soma_v_data)
    fig_soma.update_layout(
        title=title1,
        title_x=0.5,
        xaxis_title='time (ms)',
        yaxis_title=yaxis,
        legend_title='sim')
#     fig_soma.show() 

    fig_dend = go.Figure(data=dend_v_data)
    fig_dend.update_layout(
        title=title2,
        title_x=0.5,
        xaxis_title='time (ms)',
        yaxis_title=yaxis,
        legend_title='sim')
#         fig_dend.show() 
    return fig_soma, fig_dend  

def plot3(somaV, dendV, glut_placement=None, yaxis='V (mV)',
          yrange_soma=[-86, -60], yrange_dend=[-86, -30],
          stim_time=150, sim_time=400, black_trace=None, gray_trace=None,
          x_err_bar=100, y_err_bar=10, y_err_bar_shift=5,
          palette='oleron', alpha=0.8, reverse=False,
          baseline=20, dt=0.025, width=1000, height=400,
          black_shift=200, ds=10, offset=False, offset_ms=None, offset_y=None, lwd=1):

    if offset and offset_ms is None:
        offset_ms = 20
           
    n = len(somaV)
    
    if black_trace is None and gray_trace is None:
        # normal version (e.g. somaV[2:])
        cols = palette_cols(palette, n, alpha=alpha, reverse=reverse)
    
    elif black_trace is not None and gray_trace is None:
        # add only black
        cols = ['#000000'] + palette_cols(palette, n - 1, alpha=alpha, reverse=reverse)

    elif black_trace is None and gray_trace is not None:
        # add only black
        cols = ['#C0C0C0'] + palette_cols(palette, n - 1, alpha=alpha, reverse=reverse)
    
    elif black_trace is not None and gray_trace is not None:
        # add black + gray, same palette start as n-2 version
        base_cols = palette_cols(palette, n - 2, alpha=alpha, reverse=reverse)
        cols = ['#000000', '#C0C0C0'] + base_cols
            
    def update_layout(fig, main, yaxis, yrange, width, height):
        font = 'Myriad Pro'
        fig.update_layout(
            autosize=False,
            width=width,
            height=height,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=main,
            title_x=0.45,
            title_font=dict(family=font, size=14),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(
                side='right',
                tick0=yrange[0],
                dtick=(yrange[1] - yrange[0]),
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            legend=dict(title='sim', x=1.1, y=0.95),
            font=dict(family=font, size=14, color='black')
        )

    def add_corner_errbars(fig, x_right, y_base, x_err_bar, y_err_bar, shift):
        y0 = y_base + shift
        fig.add_shape(
            type='line',
            x0=x_right - x_err_bar,
            y0=y0,
            x1=x_right,
            y1=y0,
            line=dict(color='black', width=lwd),
            xref='x', yref='y'
        )
        fig.add_annotation(
            x=x_right - x_err_bar / 2,
            y=y0 - 0.5,
            text=f'{x_err_bar:g} ms',
            showarrow=False,
            font=dict(color='black', size=12),
            yanchor='top',
            xref='x', yref='y'
        )
        fig.add_shape(
            type='line',
            x0=x_right - x_err_bar,
            y0=y0,
            x1=x_right - x_err_bar,
            y1=y0 + y_err_bar,
            line=dict(color='black', width=1),
            xref='x', yref='y'
        )
        fig.add_annotation(
            x=x_right - x_err_bar - 5,
            y=y0 + y_err_bar / 2,
            text=f'{y_err_bar:g} mV',
            showarrow=False,
            font=dict(color='black', size=12),
            textangle=-90,
            xanchor='right',
            yanchor='middle',
            xref='x', yref='y'
        )

    def plot3_(somaV, dendV, glut_placement, yaxis, cols,
               yrange_soma, yrange_dend, x_err_bar, y_err_bar, bl, shift):

        if yaxis == 'V (mV)':
            title1 = 'soma PSP'
            title2 = f"{glut_placement or 'dendritic'} PSP"
        else:
            title1 = 'soma impedance'
            title2 = f"{glut_placement or 'dendritic'} impedance"

        figSoma = go.Figure()
        figDend = go.Figure()
        ind1 = 0
        ind2 = int((sim_time - stim_time + bl) / dt)
        ind3 = int((stim_time - bl) / dt)
        ind4 = int(sim_time / dt)

        x_min_all, x_max_all = np.inf, -np.inf
        offset_val = 0
        offset_val_y = 0

        for ii in range(len(somaV)):
            if gray_trace is not None and ii == gray_trace:
                continue

            dat_s, dat_d = somaV[ii], dendV[ii]
            xvals = dat_s['x'][ind1:ind2][::ds]
            y_soma = dat_s['y'][ind3:ind4][::ds]
            y_dend = dat_d['y'][ind3:ind4][::ds]

            if black_trace is not None and ii == black_trace:
                xvals = xvals - black_shift
            
            if offset:
                if offset_ms is not None:
                    xvals = xvals + offset_val
                    offset_val += offset_ms
                if black_trace is None and offset_y is not None:
                    y_soma = np.array(y_soma, dtype=float) + offset_val_y
                    y_dend = np.array(y_dend, dtype=float) + offset_val_y
                    offset_val_y += offset_y

            x_min_all = min(x_min_all, xvals.min())
            x_max_all = max(x_max_all, xvals.max())

            figSoma.add_trace(go.Scatter(x=xvals, y=y_soma, mode='lines',
                                         line=dict(color=cols[ii], width=lwd)))
            figDend.add_trace(go.Scatter(x=xvals, y=y_dend, mode='lines',
                                         line=dict(color=cols[ii], width=lwd)))

        if gray_trace is not None:
            dat_s, dat_d = somaV[gray_trace], dendV[gray_trace]
            xvals = dat_s['x'][ind1:ind2][::ds]
            y_soma = dat_s['y'][ind3:ind4][::ds]
            y_dend = dat_d['y'][ind3:ind4][::ds]
            if black_trace is not None and gray_trace == black_trace:
                xvals = xvals - black_shift
            x_min_all = min(x_min_all, xvals.min())
            x_max_all = max(x_max_all, xvals.max())
            figSoma.add_trace(go.Scatter(x=xvals, y=y_soma, mode='lines',
                                         line=dict(color=cols[gray_trace], width=lwd)))
            figDend.add_trace(go.Scatter(x=xvals, y=y_dend, mode='lines',
                                         line=dict(color=cols[gray_trace], width=lwd)))

        for fig, yrange in zip([figSoma, figDend], [yrange_soma, yrange_dend]):
            fig.add_shape(
                type='line',
                x0=x_min_all,
                x1=x_max_all,
                y0=yrange[0],
                y1=yrange[0],
                line=dict(color='gray', width=lwd, dash='dot'),
                xref='x', yref='y'
            )
            fig.add_shape(
                type='line',
                x0=x_min_all,
                x1=x_max_all,
                y0=yrange[1],
                y1=yrange[1],
                line=dict(color='gray', width=lwd, dash='dot'),
                xref='x', yref='y'
            )
            x_shift = (x_max_all - x_min_all) * 0.008
            fig.add_annotation(
                x=x_max_all + x_shift,
                y=yrange[0],
                text=f'{yrange[0]}',
                showarrow=False,
                font=dict(color='gray', size=14),
                xanchor='left',
                yanchor='middle'
            )
            fig.add_annotation(
                x=x_max_all + x_shift,
                y=yrange[1],
                text=f'{yrange[1]}',
                showarrow=False,
                font=dict(color='gray', size=14),
                xanchor='left',
                yanchor='middle'
            )
    
        y_base_soma = yrange_soma[0]
        y_base_dend = yrange_dend[0]
        x_right = x_max_all

        add_corner_errbars(figSoma, x_right, y_base_soma, x_err_bar, y_err_bar, y_err_bar_shift)
        add_corner_errbars(figDend, x_right, y_base_dend, x_err_bar, y_err_bar, y_err_bar_shift)

        update_layout(figSoma, title1, yaxis, yrange_soma, width, height)
        update_layout(figDend, title2, yaxis, yrange_dend, width, height)
        return figSoma, figDend

    fig_soma_master, fig_dend_master = plot3_(
        somaV=somaV, dendV=dendV, glut_placement=glut_placement,
        yaxis=yaxis, cols=cols,
        yrange_soma=yrange_soma, yrange_dend=yrange_dend,
        x_err_bar=x_err_bar, y_err_bar=y_err_bar, bl=baseline,
        shift=y_err_bar_shift
    )

    for fig in [fig_soma_master, fig_dend_master]:

        # makes svg better in Adobe Illustrator
        fig.update_layout(font=dict(family='Myriad Pro', size=10, color='black'))
        fig.update_traces(line_simplify=False)
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
        
        fig.update_traces(cliponaxis=False)  # keep this only
        fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)
        fig.update_layout(annotationdefaults=dict(visible=True))
        fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)

        
    return fig_soma_master, fig_dend_master

def save_fig2(soma_fig=None, dend_fig=None, cell_type='dspn', model=None, physiological=True, sim=None, g_name=None):
    import datetime 
    time = datetime.datetime.now()
    path_cell = "{}".format(cell_type)
    if not os.path.exists(path_cell):
        os.mkdir(path_cell)    
    path1 = "{}/model{}".format(path_cell, model)
    if not os.path.exists(path1):
        os.mkdir(path1)
    if physiological: 
        path2 = "{}/physiological".format(path1)
    else:
        path2 = "{}/nonphysiological".format(path1)
    if not os.path.exists(path2):
        os.mkdir(path2)  
    path3 = "{}/images".format(path2)
    if not os.path.exists(path3):
        os.mkdir(path3)

    image_dir = "{}/sim{}".format(path3, sim)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)    

    if (g_name is None):
        soma_fig.write_image("{}/soma_fig{}.svg".format(image_dir, time))
        dend_fig.write_image("{}/dend_fig{}.svg".format(image_dir, time))
        soma_fig.write_html("{}/soma_fig{}.html".format(image_dir, time))
        dend_fig.write_html("{}/dend_fig{}.html".format(image_dir, time))
    else:
        soma_fig.write_image("{}/{}_soma_fig{}.svg".format(image_dir, g_name, time))
        dend_fig.write_image("{}/{}_dend_fig{}.svg".format(image_dir, g_name, time))  
        soma_fig.write_html("{}/{}_soma_fig{}.html".format(image_dir, g_name, time))
        dend_fig.write_html("{}/{}_dend_fig{}.html".format(image_dir, g_name, time))

    
def convert2df(d, g_name):
    df = pd.DataFrame() 
    lists = sorted(d.items()) # sorted by key, return a list of tuples
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    df['dist'] = x
    df[g_name] = y
    return df

def dist_(cell, g_name):
    if g_name in ['naf', 'kaf', 'kas', 'kdr', 'kir', 'sk', 'bk', 'gaba1', 'gaba2']:
        gbar = 'gbar_{}'.format(g_name)
    else:
        gbar = 'pbar_{}'.format(g_name)        
    d__ = {}
    for sec in cell.dendlist:
         d__[h.distance(sec(0.5))] = eval('sec.{}'.format(gbar))
    return convert2df(d__, g_name)

def plot4(data, g_name): 
    import plotly.graph_objects as go
    y = data[0].y
    num = y.max()
    sig_fig = len(str(num)) - str(num).find('.') - 1
    y1 = 2*round(num, sig_fig)
    if (y1==0):
        y1 = 1e-4
    fig = go.Figure(data=data)
    fig.update_layout(
        title="{}".format(g_name),
        title_x=0.5,
        yaxis=dict(range=[0, y1]),
        xaxis_title="distance (um)",
        yaxis_title="conductance (S/cm2)",
        legend_title="cond")
    return fig

def plot5(X, dt, dists, xaxis_range=[0,150], yaxis_range=[0,8], normalised=True, title='', voltage=True, palette='oleron', alpha=1, reverse=False):
    t2 = np.arange(0, len(X[0]), 1) * dt
    v_data = []
    if normalised:
        yaxis_title="normalised amplitude"
    else:
        if voltage:
            yaxis_title="V (mV)" 
        else:
            yaxis_title="I (pA)" 

    cols = palette_cols(palette, len(X), alpha=alpha, reverse=reverse)

    for ii in list(range(len(X))):
        v_data.append(go.Scatter(x=t2, y=X[ii], name='{}'.format(round(dists[ii], 2)),
                                 line=dict(color=cols[ii])))
    fig = go.Figure(data=v_data)
    fig.update_layout(
        title="{}".format(title),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title=yaxis_title,
        xaxis_range=xaxis_range,
        yaxis_range=yaxis_range,
        legend_title="distance (um)")

    return fig

def plot5a(X, dt, locs, xaxis_range=[0,150], yaxis_range=[0,-30], normalised=False, col=[], title=''):
    t2 = np.arange(0, len(X[0]), 1) * dt
    import plotly.graph_objects as go
    v_data = []
    if normalised:
        yaxis_title="normalised amplitude"
    else:
        yaxis_title="I (pA)"        
    for ii in list(range(len(X))):
        if len(col) == 0:
            v_data.append(go.Scatter(x=t2, y=X[ii], name='{}'.format(locs[ii])))
        else:
            v_data.append(go.Scatter(x=t2, y=X[ii], line=dict(color=col[ii]), name='{}'.format(locs[ii])))
    # Plot vdata
    fig = go.Figure(data=v_data)
    fig.update_layout(
        title="{}".format(title),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="location")
    return fig

def plot5b(X, dt, locs, xaxis_range=[0,150], yaxis_range=[0,-30], normalised=False, dotted=False, col=[], title=''):
    t2 = np.arange(0, len(X[0]), 1) * dt
    import plotly.graph_objects as go
    v_data = []
    if normalised:
        yaxis_title="normalised amplitude"
    else:
        yaxis_title="V (mV)"        
    for ii in list(range(len(X))):
        if dotted:
            v_data.append(go.Scatter(x=t2, y=X[ii], line=dict(dash='dot', color='gray'), showlegend=False))
        else:
            if len(col) == 0: 
                v_data.append(go.Scatter(x=t2, y=X[ii], name='{}'.format(locs[ii])))
            else:
                v_data.append(go.Scatter(x=t2, y=X[ii], line=dict(color=col[ii]), name='{}'.format(locs[ii])))
    # Plot vdata
    fig = go.Figure(data=v_data)
    fig.update_layout(
        title="{}".format(title),
        title_x=0.5,
        xaxis_title="time (ms)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="location")
    return fig

# remove offsets
def normalise(X, stim_time, burn_time, dt):    
    def mean(x):
        n = len(x)
        sum = 0
        for i in x:
            sum = sum + i
        return(sum/n)
    ind1 = int(burn_time/dt)
    ind2 = int(stim_time/dt)
    return(X[ind1:len(X)] - mean(X[ind1:ind2]) )

def plot6(y, x, xaxis_range=[200,0], yaxis_range=[0,1.01], normalised=True, palette='oleron', alpha=1, reverse=False):
    import plotly.graph_objects as go

    if normalised:
        yaxis_title="normalised amplitude"
    else:
        yaxis_title="V (mV)"

    cols = palette_cols(palette, len(y), alpha=alpha, reverse=reverse)

    fig2 = go.Figure()
    for ii in range(len(y)):
        fig2.add_trace(go.Scatter(
            x=[x[ii]],
            y=[y[ii]],
            mode='markers',
            marker=dict(color=cols[ii], size=7),
            name='{}'.format(round(x[ii], 2)),
            showlegend=False
        ))

    fig2.update_layout(
        title="{}".format(''),
        title_x=0.5,
        xaxis_title="distance (um)",
        yaxis_title=yaxis_title,
        xaxis_range=xaxis_range,
        yaxis_range=yaxis_range,
        legend_title="attenuation")
    
    return fig2

def plot6a(mat, x, xaxis_range=[200,0], yaxis_range=[0,1.01], normalised=True, col=[], current=True):
    import plotly.graph_objects as go
    i_data = []
    if normalised:
        if current:
            yaxis_title="normalised PSC"
        else:
            yaxis_title="normalised PSP"
    else:
        if current:
            yaxis_title="I (pA)" 
        else:
            yaxis_title="V (mV)" 

    rows, columns = mat.shape
    if rows == 3:
        names =['spine', 'dendrite', 'soma']
    else:
        names =['dendrite', 'soma']
    for ii in list(range(rows)):
        if len(col) == 0:
            i_data.append(go.Scatter(x=x, y=mat[ii,:], line=dict(color='gray'), name='{}'.format(names[ii]), showlegend=True))
        else:
            i_data.append(go.Scatter(x=x, y=mat[ii,:], line=dict(color=col[ii]), name='{}'.format(names[ii]), showlegend=True))

    fig2 = go.Figure(data=i_data)
    fig2.update_layout(
        title="{}".format(''),
        title_x=0.5,
        xaxis_title="distance (um)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="attenuation")
    return fig2

def plot6aa(mat, x, xaxis_range=[200,0], yaxis_range=[0,1.01], normalised=True, col=[], current=True):
    import plotly.graph_objects as go
    i_data = []
    if normalised:
        if current:
            yaxis_title="normalised PSC"
        else:
            yaxis_title="normalised PSP"
    else:
        if current:
            yaxis_title="I (pA)" 
        else:
            yaxis_title="V (mV)" 

    rows, columns = mat.shape
    if rows == 3:
        names =['spine', 'dendrite', 'soma']
    else:
        names =['dendrite', 'soma']
    for ii in list(range(rows)):
        if len(col) == 0:
            i_data.append(go.Scatter(x=x, y=mat[ii,:], line=dict(color='gray'), name='{}'.format(names[ii]), showlegend=True))
        else:
            i_data.append(go.Scatter(x=x, y=mat[ii,:], line=dict(color=col[ii]), name='{}'.format(names[ii]), showlegend=True))

    fig2 = go.Figure(data=i_data)
    fig2.update_layout(
        title="{}".format(''),
        title_x=0.5,
        xaxis_type='log',
        xaxis_title="series resistance (MOhm)",
        yaxis_title=yaxis_title,
        xaxis_range = xaxis_range,
        yaxis_range = yaxis_range,
        legend_title="attenuation")
    return fig2

def plot9(x, ydict, yaxis='', xaxis='', _range=None, _range_subset=None,
          yaxis_range=[-110, 30], xaxis_range=[200, 1500],
          y_err_bar=10, x_err_bar=100, y_err_bar_shift=5,
          ybar_units='mV', xbar_units='ms',
          palette='oleron', reverse=False, col=None, alpha=1,
          lw=1, width=600, height=520, fig_title='',
          ds=10, offset=False, offset_ms=20, yabline=None, text_color='grey'):

    fig = go.Figure()
    N = len(ydict)
    if col is None:
        cols = palette_cols(palette, N, alpha=alpha, reverse=reverse)
    else:
        cols = [col.replace('1.00', f'{alpha:.2f}')] * N
    x_mask = (x >= xaxis_range[0]) & (x <= xaxis_range[1])
    x_trim = x[x_mask][::ds]
    if _range_subset is not None:
        indices_to_plot = [i - 1 for i in _range_subset if 1 <= i <= N]
    else:
        indices_to_plot = range(N)
    offset_val = 0
    x_all, y_all = [], []
    for ii in indices_to_plot:
        y_trim = np.array(ydict[ii])[x_mask][::ds]
        xvals = x_trim + offset_val if offset else x_trim
        if offset:
            offset_val += offset_ms
        x_all.append(xvals)
        y_all.append(y_trim)
        name = str(_range[ii]) if _range is not None else f'Trace {ii+1}'
        fig.add_trace(go.Scatter(x=xvals, y=y_trim, mode='lines',
                                 line=dict(color=cols[ii], width=lw),
                                 name=name, showlegend=True))
    all_x = np.concatenate(x_all)
    all_y = np.concatenate(y_all)
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    xaxis_range_extended = [x_min - x_pad, x_max + x_pad]
    yaxis_range_auto = [y_min - y_pad, y_max + y_pad]
    fig.update_layout(autosize=False, width=width, height=height,
                      margin=dict(l=40, r=200, t=40, b=60),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      yaxis=dict(range=yaxis_range if yaxis_range is not None else yaxis_range_auto, showticklabels=False, title=yaxis),
                      xaxis=dict(range=xaxis_range_extended, showticklabels=False, title=xaxis),
                      title=dict(text=fig_title, x=0.5, xanchor='center',
                                 font=dict(color=text_color, size=12)),
                      legend=dict(x=1.2, y=1, xanchor='left', yanchor='top',
                                  bgcolor='rgba(255,255,255,0)', borderwidth=0,
                                  font=dict(color=text_color)))
    if yabline is not None:
        for yv in yabline:
            fig.add_shape(type='line', x0=x_min, x1=x_max, y0=yv, y1=yv,
                          line=dict(color=text_color, width=lw, dash='dot'),
                          xref='x', yref='y')
    if yabline is not None:
        x_shift = (x_max - x_min) * 0.01
        for yv in yabline:
            fig.add_annotation(x=x_max + x_shift, y=yv, text=f'{round(yv, 3):g}',
                               showarrow=False, xref='x', yref='y',
                               font=dict(color=text_color, size=12),
                               xanchor='left', yanchor='middle')
        
    fig.add_shape(type='line', x0=1.08, y0=0.1, x1=1.18, y1=0.1,
                  xref='paper', yref='paper', line=dict(color=text_color, width=lw))
    fig.add_annotation(x=1.13, y=0.095, text=f'{x_err_bar:g} {xbar_units}',
                       xref='paper', yref='paper', showarrow=False,
                       font=dict(color=text_color, size=12),
                       yanchor='top', xanchor='center')
    fig.add_shape(type='line', x0=1.08, y0=0.1, x1=1.08, y1=0.25,
                  xref='paper', yref='paper', line=dict(color=text_color, width=lw))
    fig.add_annotation(x=1.075, y=0.175, text=f'{y_err_bar:g} {ybar_units}',
                       xref='paper', yref='paper', showarrow=False,
                       font=dict(color=text_color, size=12),
                       textangle=-90, xanchor='right', yanchor='middle')
    
    # makes svg better in Adobe Illustrator
    fig.update_layout(font=dict(family='Myriad Pro', size=10, color=text_color))
    fig.update_traces(line_simplify=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    
    fig.update_traces(cliponaxis=False)  # keep this only
    fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)
    fig.update_layout(annotationdefaults=dict(visible=True))
    fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)

    return fig
   
################## plot functions ##################
# units mA/cm²
def extract_mechs_dict2array(data_dict, mech, scale=1):
    mech_arrays = []
    for sim_key, sim_data in data_dict.items():
        if mech in sim_data['mechs']:
            idx = sim_data['mechs'].index(mech)
            mech_current = np.array(sim_data['i'][idx + 1]) * scale
            mech_arrays.append(mech_current)
    return mech_arrays

def round_up_to_sig(x):
    if x == 0:
        return 0
    exp = math.floor(math.log10(abs(x)))
    frac = abs(x) / 10**exp
    if frac <= 1:
        sig = 1
    elif frac <= 2:
        sig = 2
    elif frac <= 5:
        sig = 5
    else:
        sig = 10
    return math.copysign(sig * 10**exp, x)
    
def plot_mech_current(mech, data_dict, _range=None, _range_subset=None, sim_image_path=None, 
                      sim_time=None, step_start=0, dt=0.1, ds=10, text_color='grey', save=False):
    
    out = extract_mechs_dict2array(data_dict['i_mechs_dend'], mech=mech)
    x = np.arange(0, len(out[1]) * dt, dt)

    if _range_subset is not None:
        indices = [i-1 for i in _range_subset if 1 <= i <= len(out)]
    else:
        indices = list(range(len(out)))

    if not indices:
        indices = list(range(len(out)))

    out_sel = [out[i] for i in indices]

    ymin = min(np.min(y) for y in out_sel)
    ymax = max(np.max(y) for y in out_sel)
    
    if ymin >= 0:
        yaxis_min = -0.05 * ymax
    else:
        yaxis_min = 1.05 * ymin
    
    yaxis_range = [yaxis_min, ymax]

    y_err_bar = round_up_to_sig(ymax / 10)
    y_err_bar_shift = y_err_bar / 2
    
    fig = plot9(x=x, ydict=out, _range=_range, _range_subset=_range_subset, xaxis_range=[step_start-100, sim_time], ds=ds, text_color=text_color,
                yaxis_range=yaxis_range, y_err_bar=y_err_bar, y_err_bar_shift=y_err_bar_shift, ybar_units='mA/cm²', fig_title=mech)

    fig.show(config={"toImageButtonOptions": {"format": "svg"}})

    fig.update_layout(font=dict(family='Myriad Pro', size=10, color='black'))
    fig.update_traces(line_simplify=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    
    fig.update_traces(cliponaxis=False)  # keep this only
    fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)
    fig.update_layout(annotationdefaults=dict(visible=True))

    if save:
        fig.write_image(f"{sim_image_path}/fig_{mech}_summary.svg", format='svg', scale=3)
    
    return fig

    
def gaba_idx(dend):
    locs = []
    for seg in dend:
        locs.append(seg.x)
    return locs

def all_synapses_tree(cell, dend_tree, dendrite, glut):
    candidate_list = []
    locs_list = []
    for dend in cell.dendlist:
        if dend.name() == dendrite:
            dendrite = dend
        # get path to soma
    pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    if glut:
        # get all unique spine locations to place glut synapses
        for sec in pathlist:
            dend_dist = h.distance(cell.soma(), sec(1)) # distance to end of that dendrite from middle of soma
            locs = []
            if dend_dist < 30:
                candidates = []
            else:
                candidates = spine_idx(cell=cell, spines=spines, dend=sec.name())
                for candidate in candidates:
                    locs.append(candidate.x)

            candidate_list.append(candidates)
            locs_list.append(locs)
    else:
        # get all unique gaba synapse locations
        locs_list = []
        for sec in pathlist:
            locs = gaba_idx(sec)
            locs_list.append(locs)

    return pathlist, locs_list, candidate_list

def space_clamped(cell, spines, Ra = 1.59e-10):
    for sec in cell.allseclist:
        sec.Ra = Ra
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.head.Ra = Ra
            spine_obj.neck.Ra = Ra

def cap(cell, spines, cm = 1):
    for sec in cell.allseclist:
        sec.cm = cm
    for sec in cell.dendlist:
        sec_spines = list(spines[sec.name()].items())
        for spine_i, spine_obj in sec_spines: 
            spine_obj.head.cm = cm
            spine_obj.neck.cm = cm
            
def find_closest_value(test, target_value):
    import numpy as np
    test = np.array(test)
    distances = np.abs(test - target_value)
    closest_index = np.argmin(distances)
    closest_value = test[closest_index]
    return closest_value, closest_index

def rounded(number, n=10):
    if number >= 0:
        rounded = n*math.ceil(number/n)
    else:
        rounded = n*math.floor(number/n)
    return rounded


def IR(X, step_start, step_end, step):   
    ind1 = int((step_start-5)/dt)
    ind2 = int(step_start/dt)
    ind3 = int((step_end-5)/dt)
    ind4 = int(step_end/dt)

    def mean(x):
        n = len(x)
        sum = 0
        for i in x:
            sum = sum + i
        return(sum/n)
 
    return(1e3 *( mean(X[ind1:ind2]) - mean(X[ind3:ind4]) )  / -step) # MOhm


def whole_cell_capacitance(cell, spines=None, Cm=1):
    # for seg in sec.allseg():
    #     print(seg.area())

    # for sec in cell.dendlist:
    #     print(seg.area())
    areas = []
    for sec in cell.dendlist:
        for i,seg in enumerate(sec):
            areas.append(seg.area())

    if spines is None:
        AREA = sum(areas)
    else:
        spine_areas = []
        for sec in cell.dendlist:
            sec_spines = list(spines[sec.name()].items())
            for spine_i, spine_obj in sec_spines: # area of sphere + cylinder less areas that are connections to head and dendrite
                spine_areas.append(4 * math.pi * ((spine_obj.head.diam/2) ** 2) + 2 * math.pi * spine_obj.neck.L * spine_obj.neck.diam/2  - 2 * math.pi * ((spine_obj.neck.diam/2)**2) ) # diam

        AREA = sum(spine_areas) + sum(areas) +  4 * math.pi * cell.soma.diam/2 ** 2 # in um sq
    # AREA in um sq
    # Cm in uF/cm2
    # convert uF/cm2 to F/um2
    # To convert square centimeters (cm²) to square micrometers (μm²), you can use the following conversion factor:
    # 1 cm² = 1e8 μm²
    # convert uF to pF conversion factor
    # 1uF = 1e6 pF
    
    # AREA * Cm * 1e-8 * 1e6 # (cm2)
    cap = AREA * Cm * 1e-2 # pF
    return cap
 
def sampler(names, n, replacement=True):
    if replacement:
        return [random.choice(names) for _ in range(n)]
    else:
        return random.sample(names, n)
    
def uniform_values(n):
    return [random.uniform(0, 1) for _ in range(n)]

# function to determine what is varying
def variable_detector(xrange):
    vary = False
    differences = [abs(xrange[i] - xrange[i + 1]) for i in range(len(xrange) - 1)]
    sum_of_differences = sum(differences)
    if sum_of_differences > 0:
        vary = True
    return vary

def spine_locator(cell_type, specs, spine_per_length, frequency, d_lambda, dend_glut, num_gluts=1, soma_diameter=None, method=0, rel_x=None, dend2remove=None):
    if rel_x is None:
        rel_x = 2/3
    
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, verbose=False, dend2remove=dend2remove)
   
    glut_secs = [sec for target_dend in dend_glut for sec in cell.dendlist if sec.name() == target_dend] * num_gluts
    
    final_spine_locs = []
    glut_id = 0
    for dend_glut in glut_secs:
        sec_spines = list(spines[dend_glut.name()].items())
        if sec_spines:
            candidate_spines = [spine_obj for _, spine_obj in sec_spines]
            if len(glut_secs) < len(sec_spines):
                if method == 1:
                    spine_idx = int(rel_x * len(candidate_spines)) - 1
                    spine = candidate_spines[spine_idx - glut_id]
                else:
                    spine_idx = int(rel_x * len(candidate_spines)) - num_gluts
                    if spine_idx < 0:
                        if len(candidate_spines) >= num_gluts:
                            spine_idx = len(candidate_spines) - num_gluts
                        else:
                            spine_idx = 0
                    spine = candidate_spines[spine_idx + glut_id]
                final_spine_locs.append(spine.x)
            glut_id += 1
    if len(glut_secs) < len(sec_spines):
        if method == 1:
            if not all(final_spine_locs[i] > final_spine_locs[i+1] for i in range(len(final_spine_locs)-1)):
                all_spine_locs = [spine.x for spine in candidate_spines]
                final_spine_locs = all_spine_locs[:len(final_spine_locs)]
                final_spine_locs.reverse()
        else:
            if not all(final_spine_locs[i] < final_spine_locs[i+1] for i in range(len(final_spine_locs)-1)):
                all_spine_locs = [spine.x for spine in candidate_spines]
                final_spine_locs = all_spine_locs[:len(final_spine_locs)]
    return final_spine_locs
    
# relative version of gaba_onsets
def rel_gaba_onset(n, N):
    if N in [0,1]:
        if (n < 4):
            gaba_onsets = list(range(0, 0 + n)) 
        else:
            if n % 3 == 0:
                gaba_onsets = list(range(0, 0 + int(n/3))) * 3 * N
            else:
                gaba_onsets = list(range(0, 0 + int(n/3)+1)) * 3 * N
        gaba_onsets = gaba_onsets[:n]
    else:
        n1 = round(n / N)
        if n1 % 3 == 0:
            n1 = round(n / N)
            onsets = list(range(0, n1)) 
        else:
            onsets = list(range(0, n1+1)) 
        gaba_onsets = [x for x in onsets for _ in range(N)]
        gaba_onsets = gaba_onsets[:n]
    return gaba_onsets


def plt1(data_dict):
    # Unpack metadata
    metadata = data_dict['metadata']
    keys = ['sim_time', 'stim_time', 'model', 'cell_type', 'showplot', 'save', 'sim', 'physiological', 'dt']
    sim_time, stim_time, model, cell_type, showplot, save, sim, physiological, dt = (metadata[key] for key in keys)
    
    # Setup time axis
    N = len(data_dict['vsoma'][0])
    time_axis = np.linspace(0, sim_time, N)

    # Define voltage trace containers
    soma_v_traces = []
    dend_v_traces = []

    # Generate traces
    for vsoma, vdend in zip(data_dict['vsoma'], data_dict['vdend']):
        soma_v_traces.append(go.Scatter(x=time_axis, y=vsoma))
        dend_v_traces.append(go.Scatter(x=time_axis, y=vdend))

    # Set y-axis ranges based on model or cell type
    yrange_soma, yrange_dend = ([-85, -60], [-85, -30])  # Default ranges
    if model == 2 or cell_type == 'ispn':
        yrange_soma, yrange_dend = ([-85, -50], [-85, -20])

    # Generate plots
    fig_soma, fig_dend = plot3(somaV=soma_v_traces, dendV=dend_v_traces, glut_placement='', yaxis='V (mV)', yrange_soma=yrange_soma, yrange_dend=yrange_dend, stim_time=stim_time, sim_time=sim_time, black_trace=0, gray_trace=1, err_bar=50, baseline=20, dt=dt,width=600, height=400)

    # Show plots
    if showplot:
        fig_soma.show()
        fig_dend.show()

    # Save plots
    if save:
        base_dir = os.path.join(cell_type, f"model{model}", 'physiological' if physiological else 'nonphysiological', f"images/sim{sim}")
        os.makedirs(base_dir, exist_ok=True)
        for fig, name in zip([fig_soma, fig_dend], ['soma', 'dend']):
            fig_path = os.path.join(base_dir, f"fig1_{name}.svg")
            fig.write_image(fig_path)
            fig.write_html(fig_path.replace('.svg', '.html'))
            
        
        
def plt2(data_dict, sim_time, dt, model, cell_type, stim_time, showplot, save, physiological, sim, offset=40, spine=True
    ):
    
    time = np.arange(0, len(data_dict['vsoma'][0]) * dt, dt)
    
    idx1, idx2, idx3  = 0, int(sim_time/dt), int(stim_time/dt)
    soma_v_master, dend_v_master, spine_v_master = [], [], []
    peaks_v_soma, peaks_v_dend, peaks_v_spine = [], [], []
    mins_v_soma, mins_v_dend, mins_v_spine = [], [], []
    x=time[idx1:idx2]
    for ii, (soma_v, dend_v, spine_v) in enumerate(zip(data_dict['vsoma'], data_dict['vdend'], data_dict['vspine'])):
        
        y=extract2(soma_v)[idx1:idx2]
        soma_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_soma.append(max(soma_v))
        mins_v_soma.append(soma_v[idx3])
        
        y=extract2(dend_v)[idx1:idx2]
        dend_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_dend.append(max(dend_v))
        mins_v_dend.append(dend_v[idx3])
        
        y=extract2(spine_v)[idx1:idx2]
        spine_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_spine.append(max(spine_v))
        mins_v_spine.append(spine_v[idx3])
    ysoma_range = [0.1*math.floor(min(mins_v_soma)/0.1), 0.1*math.ceil(max(peaks_v_soma)/0.1)]
    ydend_range = [math.floor(min(mins_v_dend)), math.ceil(max(peaks_v_dend))]
    yspine_range = [math.floor(min(mins_v_spine)), math.ceil(max(peaks_v_spine))]

    figs = plot3a(somaV=soma_v_master, dendV=dend_v_master, spineV=spine_v_master, ysoma_range=ysoma_range, 
                  ydend_range=ydend_range, yspine_range=yspine_range, stim_time=stim_time, sim_time=sim_time, 
                  dt=dt, offset=offset,spine=spine)

    if showplot:
        for fig in figs:
            fig.show()
              
    if save:
        path_format = f'{cell_type}/model{model}/{{}}/images/sim{sim}'
        folder = path_format.format('physiological' if physiological else 'nonphysiological')
        os.makedirs(folder, exist_ok=True)

        plots_to_do = ['spine', 'dend', 'soma'] if spine else ['dend', 'soma']
        for idx, name in enumerate(plots_to_do):
            figs[idx].write_image(f'{folder}/fig1_{name}{time}.svg')
            figs[idx].write_html(f'{folder}/fig1_{name}{time}.html')


def hex_palette(n):
    colors = ['#6A5ACD', '#CD5C5C', '#458B74', '#9932CC', '#FF8247'] # Set your custom color palette
    if n < len(colors):
        colors = colors[0:n]
    else:
        colors = sns.blend_palette(colors,n)
    cols = list(map(mpl.colors.rgb2hex, colors))
    return cols

def update_layout(fig, title, yaxis, yrange, width, height):
    font = 'Droid Sans'
    font_size = 18
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        title_x=0.45,
        title_font_family=font,
        title_font_size=font_size,
        xaxis=dict(showticklabels=False, titlefont=dict(size=font_size, family=font), tickfont=dict(size=font_size, family=font), showgrid=False),
        yaxis=dict(side='right', tick0=yrange[0], dtick=yrange[1]-yrange[0], tickfont=dict(size=font_size, family=font), showgrid=False),
        legend=dict(title='sim', x=1.1, y=0.95)
    )

# offset corrects baseline
# stim_time - offset gives some baseline dependent on the smallest value of timing_range ie 
# baseline = offset - stim_time + min(timing_range) ie if stim_time = 150, first timing at 120 then offset=40 gives a 10 ms baseline

def plot_trace(fig, data_list, cols, dt, err_bar, yrange, stim_time, sim_time, offset):
    ind1, ind2 = 0, int((sim_time - stim_time + offset)/dt)
    ind3, ind4 = int((stim_time - offset)/dt), int(sim_time/dt)
    dy = (yrange[1] - yrange[0])*9/10
    for dat, color in zip(data_list, cols):
        fig.add_trace(go.Scatter(x=dat['x'][ind1:ind2], y=dat['y'][ind3:ind4], mode='lines', line=dict(color=color)))
    fig.add_hline(y=yrange[0], line_width=2, line_dash="dot", line_color="gray")
    fig.add_hline(y=yrange[1], line_width=2, line_dash="dot", line_color="gray")
    fig.add_shape(type='line', x0=ind2*dt-err_bar, y0=yrange[0]+dy, x1=ind2*dt, y1=yrange[0]+dy, line=dict(color='black'))

def plot3a(somaV, dendV, spineV, ysoma_range, ydend_range, yspine_range, stim_time, sim_time, dt, offset, width=800, height=400, controls=True, spine=True):
    n = len(somaV)
    if controls:
        cols = hex_palette(n-2)
        cols.insert(0,'#000000')
        cols.insert(1,'#D3D3D3')
    else:
        cols = hex_palette(n)
   
    titles = ['spine PSP', 'dendritic PSP', 'soma PSP'] if spine else ['dendritic PSP', 'soma PSP']
    yranges = [yspine_range, ydend_range, ysoma_range] if spine else [ydend_range, ysoma_range]
    data_list = [spineV, dendV, somaV] if spine else [dendV, somaV]
    
    figs = []
    for data, title, yrange in zip(data_list, titles, yranges):
        fig = go.Figure()
        plot_trace(fig, data, cols, dt, 25, yrange, stim_time, sim_time, offset)
        update_layout(fig, title, 'V (mV)', yrange, width, height)
        figs.append(fig)
    return figs




def check_sim(sim, sims):
    """
    Checks if sim starts with or is equivalent to any value in values_to_check.
    
    :param sim: Value to check (can be integer or string)
    :param values_to_check: List of values to check against (mix of integers and strings)
    :return: True if there's a match, False otherwise
    """
    strings_to_check = [str(val) for val in sims]  # Convert all values to strings

    sim_str = str(sim)  # Convert sim to a string regardless of its type

    return any(sim_str.startswith(s) or sim_str == s for s in strings_to_check)


# to store sim-generated variables

def update_data_dict(data_dict, protocol, v_tree, v_tree_spine, v_branch, soma_v, dend_v, spine_v, timing, t, dists, dists_spine, dends_v, dends_spine, i_dend_mechs, i_mechs_all, i_mechs_all_spine, dists_i_mechs, dists_spine_i_mechs, dends_i_mechs, dends_spine_i_mechs, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, time, record_dist, impedance=False, return_currents=False, record_spine=False):
    
    data_dict['v_tree'][protocol] = v_tree
    data_dict['v_tree_spine'][protocol] = v_tree_spine
    data_dict['v_branch'][protocol] = v_branch
    data_dict['soma_v'].append(soma_v[0])
    data_dict['dend_v'].append(dend_v[0])
    if record_spine:
        data_dict['spine_v'].append(spine_v[0])
    data_dict['timing'].append(timing)
    data_dict['time'].append(np.asarray(t))
    data_dict['dists'].append(dists)
    data_dict['dists_spine'].append(dists_spine)
    data_dict['dendrites_v'].append(dends_v)
    data_dict['dendrites_spine'].append(dends_spine)
    data_dict['i_mechs'][protocol] = i_dend_mechs
    data_dict['i_mechs_all'][protocol] = i_mechs_all
    data_dict['i_mechs_all_spine'][protocol] = i_mechs_all_spine
    data_dict['dists_i_mechs'].append(dists_i_mechs)
    data_dict['dists_spine_i_mechs'].append(dists_spine_i_mechs)
    data_dict['dendrites_i_mechs'].append(dends_i_mechs)
    data_dict['dendrites_spine_i_mechs'].append(dends_spine_i_mechs)
    data_dict['record_dists'].append(record_dist)
    data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
    data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
    data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
    data_dict['g_gaba'][protocol] = pd.DataFrame(gaba_conductances).transpose()
    
    if impedance:
        data_dict['z_input'].append(z_input[0])
        data_dict['z_transfer'].append(z_transfer[0])
        
    if return_currents:
        data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
        data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
        data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
        data_dict['g_gaba'][protocol] = pd.DataFrame(gaba_conductances).transpose()

    data_dict['timestamp'][protocol] = time
    
def plot_cumulative_frequency(cell_type, specs, spine_per_length=1.61, soma_diameter=None, frequency=2000, d_lambda=0.05, dend2remove=None):
    """
    Plots the cumulative frequency of spine distances from the soma.

    Parameters:
    - cell_type: the type of cell
    - specs: cell specifications
    - spines: spine data

    Returns:
    - A plotly figure displaying the cumulative frequency
    """

    import plotly.graph_objects as go

    # Build the cell
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, dend2remove=dend2remove)

    # Compute distances of each spine from the soma
    dist_spine = []
    for dend in cell.dendlist:
        sec_spines = list(spines[dend.name()].items())
        for spine in sec_spines:
            dist_spine.append(h.distance(dend(spine[1].x)))

    # Sort the data in ascending order
    sorted_data = sorted(dist_spine)

    # Calculate cumulative frequencies
    cumulative_freq = [i / len(sorted_data) for i in range(1, len(sorted_data) + 1)]

    # Create the cumulative frequency plot
    fig = go.Figure(data=go.Scatter(x=sorted_data, y=cumulative_freq, mode='lines'))

    # Set plot labels and title
    fig.update_layout(
        title={
            'text': 'cumulative frequency',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='distance (um)',
        yaxis_title='cumulative frequency',
        xaxis_range=[0, 300],  # Set x-axis range
    )

    # Display the plot
    fig.show()
    
    
def plot_spine_distance_histogram(cell_type, specs,  spine_per_length=1.61, soma_diameter=None, frequency=2000, d_lambda=0.05, bin_size=10, dend2remove=None):
    """
    Plots the histogram of spine distances from the soma.

    Parameters:
    - cell_type: the type of cell
    - specs: cell specifications
    - bin_size: size of the bins for the histogram

    Returns:
    - A plotly figure displaying the histogram
    """

    import plotly.graph_objects as go

    # Build the cell
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, dend2remove=dend2remove)

    # Compute distances of each spine from the soma
    dist_spine = []
    for dend in cell.dendlist:
        sec_spines = list(spines[dend.name()].items())
        for spine in sec_spines:
            dist_spine.append(h.distance(dend(spine[1].x)))

    # Create the histogram plot
    fig = go.Figure(data=go.Histogram(x=dist_spine, nbinsx=int(max(dist_spine) / bin_size)))

    # Set plot labels and title
    fig.update_layout(
        title={
            'text': 'Histogram of Spine Distances from Soma',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='distance (um)',
        yaxis_title='frequency',
        xaxis_range=[0, 300],  # Set x-axis range
    )

    # Display the plot
    fig.show()


def filter_master(v_master, filter_array):
    return [scatter for scatter, to_keep in zip(v_master, filter_array) if to_keep]

def plt3(data_dict, metadata, sim_time, dt, model, cell_type, stim_time, showplot, save, physiological, sim, offset=40, gaba=True, glut=True
    ):
    
    time = np.arange(0, len(data_dict['vsoma'][0]) * dt, dt)
    
    idx1, idx2, idx3  = 0, int(sim_time/dt), int(stim_time/dt)
    soma_v_master, dend_v_master, spine_v_master = [], [], []
    peaks_v_soma, peaks_v_dend, peaks_v_spine = [], [], []
    mins_v_soma, mins_v_dend, mins_v_spine = [], [], []
    x=time[idx1:idx2]
    
    for ii, (soma_v, dend_v, spine_v) in enumerate(zip(data_dict['vsoma'], data_dict['vdend'], data_dict['vspine'])):
        y=extract2(soma_v)[idx1:idx2]
        soma_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_soma.append(max(soma_v))
        mins_v_soma.append(soma_v[idx3])
        
        y=extract2(dend_v)[idx1:idx2]
        dend_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_dend.append(max(dend_v))
        mins_v_dend.append(dend_v[idx3])
        
        y=extract2(spine_v)[idx1:idx2]
        spine_v_master.append(go.Scatter(x=x, y=y))
        peaks_v_spine.append(max(spine_v))
        mins_v_spine.append(spine_v[idx3])
    ysoma_range = [0.1*math.floor(min(mins_v_soma)/0.1), 0.1*math.ceil(max(peaks_v_soma)/0.1)]
    ydend_range = [math.floor(min(mins_v_dend)), math.ceil(max(peaks_v_dend))]
    yspine_range = [math.floor(min(mins_v_spine)), math.ceil(max(peaks_v_spine))]

    gaba_range = metadata['gaba_range']
    glut_range = metadata['glut_range']

    if gaba and glut:
        f = [f1 and f2 for f1, f2 in zip(gaba_range, glut_range)] # glut and gaba
    elif gaba and not glut:
        f = [f1 and not f2 for f1, f2 in zip(gaba_range, glut_range)] # gaba only
    elif glut and not gaba:
        f = [not f1 and f2 for f1, f2 in zip(gaba_range, glut_range)] # glut only
     
    spine_v_master = filter_master(spine_v_master, f)
    dend_v_master = filter_master(dend_v_master, f)
    soma_v_master = filter_master(soma_v_master, f)

    figs = plot3a(somaV=soma_v_master, dendV=dend_v_master, spineV=spine_v_master, ysoma_range=ysoma_range, 
                  ydend_range=ydend_range, yspine_range=yspine_range, stim_time=stim_time, sim_time=sim_time, 
                  dt=dt, offset=offset, controls=False)
    
    return figs

def plt4(title, figs1, figs2, figs3, index, sim, time, show_subtitles=False, showplot=True, save=True, cell_type='dspn', physiological=True, model=1):
    fig = make_subplots(rows=1, cols=3)
    fig1 = figs1[index]
    fig2 = figs2[index]
    fig3 = figs3[index]

    shapes = copy.deepcopy(fig1.layout.shapes)
    yaxis_range = [round(val, 1) for val in [shapes[0]['y0'], shapes[1]['y0']]]

    annotations = []
    subplot_titles = ['glut + gaba', 'glut only', 'gaba only']

    for col, subplot in enumerate([fig1, fig2, fig3], start=1):
        for trace in subplot.data:
            trace_copy = copy.deepcopy(trace)
            color = trace.line.color
            trace_copy.line.color = color
            fig.add_trace(trace_copy, row=1, col=col)

        fig.update_yaxes(
            range=yaxis_range,
            row=1,
            col=col,
            tickvals=yaxis_range,
            ticktext=[str(val) for val in yaxis_range] if col == 1 else [],
            showticklabels=(col == 1)
        )
        fig.update_xaxes(showticklabels=False, ticks="", row=1, col=col)

        if show_subtitles:
            annotations.append(dict(
                    xref='paper',  # This will refer to the entire width of the plotting area
                    yref='paper',
                    x = col / 2.5 - 0.3 ,  # Manually adjust to line up with your subplots
                    y=1.07,
                    text=subplot_titles[col - 1],
                    showarrow=False,
                    font=dict(size=12),
                ))

    fig.update_layout(
        shapes=shapes,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        annotations=annotations if show_subtitles else []
    )
    
    if showplot:
        fig.show()
              
    if save:
        path_format = f'{cell_type}/model{model}/{{}}/images/sim{sim}'
        folder = path_format.format('physiological' if physiological else 'nonphysiological')
        os.makedirs(folder, exist_ok=True)
        fig.write_image(f'{folder}/fig1 {title} {time}.svg')
        fig.write_html(f'{folder}/fig1 {title} {time}.html')       
        

# returns dists, dends and locs for all unique locations on a dendritic tree with dendrite being the most proximal
def path_secs(cell, dend_tree, dendrite):
    dists = []
    dends = []
    locs = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
    # get path to soma
    if dendrite.name() != 'soma[0]':
        pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    else:
        pathlist = [dendrite]
    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        for seg in sec:
            dist = h.distance(seg)
            dists.append(dist)
            loc = seg.x
            dends.append(sec.name())
            locs.append(loc)
            i = i + 1
    return dists[::-1], dends[::-1], locs[::-1]

# input comprises a list of dend names and their equivalent locations given in locs returns 
# returns record vectors for v in NEURON
def record_all_path(cell, dends, locs):
    
    pathlist = []
    for dend in dends:
        for sec in cell.allseclist:
            if sec.name() == dend:
                pathlist.append(sec)
        
    all_v = {}
    i=0
    for sec,loc in zip(pathlist,locs) :
        all_v[i] = h.Vector()
        all_v[i].record(sec(loc)._ref_v) # given a sec with multiple seg
        i = i + 1
    
    return all_v

# gives local membrane potential for dendrites with their equivalent locations
def membrane_potentials(cell, 
                  dends, 
                  locs, 
                  sim_time=150, 
                  dt=0.025,
                  v_init=-85
                  ):
    t = h.Vector().record(h._ref_t)
    iclamp1 = h.IClamp(cell.soma(0.5))
    iclamp1.dur = sim_time

    v_path = record_all_path(cell=cell, dends=dends, locs=locs)

    # Initialize cell starting voltage
    h.finitialize(v_init)
    # Run simulation
    h.dt = dt

    while h.t < sim_time:
        h.fadvance()

    v_tree = []
    for ii in list(range(len(v_path))):
        v_tree.append(np.array(v_path[ii]))
        
    all_revs = []
    for v in v_tree:
        all_revs.append(v[int(sim_time/dt)])
        
    return all_revs

                    
def pairs_in_order(dend, reversals):
    seen = set()
    unique_pairs = []
    for d, rev in zip(dend, reversals):
        pair = (d, rev)  # No rounding here
        if pair not in seen:
            unique_pairs.append((d, rev))
            seen.add(pair)
    return unique_pairs
                                


# function gets all unique locations on path to soma
def record_all_path_secs_i_mechs(cell, dend_tree, dendrite, mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']):
    all_i_mechs = {}
    dists = []
    dends = []
    
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
    # get path to soma
    if dendrite.name() != 'soma[0]':
        pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
    else:
        pathlist = [dendrite]
    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        for seg in sec:
            dends.append(sec.name())
            dist = h.distance(seg)
            dists.append(dist)
            loc = seg.x
            all_i_mechs[i] = record_i_mechs(cell=cell, dend=sec, loc=loc, return_currents=True, silent=True, mechs=mechs)
            i = i + 1
    return all_i_mechs, dists, dends

# records from all activated spine locations
def record_all_activated_spine_v2(cell, dendrite, activated_spines):
    all_v_spines= {}
    all_v_dendrite = {}
    dists = []
    locs  = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend

    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for spine in activated_spines:
        locs.append(spine.x)
        all_v_spines[i] = h.Vector()
        all_v_spines[i].record(spine.head(0.5)._ref_v)
        locs.append(spine.x)
        all_v_dendrite[i] = h.Vector()
        all_v_dendrite[i].record(dendrite(spine.x)._ref_v) 
        dists.append(h.distance(dendrite(spine.x)))
        i = i + 1

    return all_v_spines, all_v_dendrite, dists


# finds all unique spine locations on path to soma
# ignores spine if active
# records voltage changes in that spine to postsynaptic potential at a remote site
def record_all_path_secs_spine_v2(cell, spines, dend_tree, dendrite, activated_spines):
    all_v = {}
    dists = []
    dends = []
    locs  = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
        # get path to soma
        if dendrite.name() != 'soma[0]':
            pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
            # Remove soma[0] from pathlist if it exists
            pathlist = [sec for sec in pathlist if sec.name() != 'soma[0]']
        else:
            pathlist = [dendrite]  

    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        sec_spines = list(spines[sec.name()].items())
        sec_spines_x = []
        for spine_i, spine_obj in sec_spines: 
            sec_spines_x.append(spine_obj.x)
        if sec_spines_x:
            for seg in sec:
                dends.append(sec.name())
                dist = h.distance(seg)
                dists.append(dist)
                loc = seg.x
                # Find the index of the value in sec_spines_x closest to loc
                idx = min(range(len(sec_spines_x)), key=lambda i: abs(sec_spines_x[i] - loc))
                spine_id, spine = sec_spines[idx]
                while spine in activated_spines:
                    idx += 1
                    spine_id, spine = sec_spines[idx]
                locs.append(spine.x)
                all_v[i] = h.Vector()
                all_v[i].record(spine.head(0.5)._ref_v) # given a sec with multiple seg
                i = i + 1

    return all_v, dists, dends, locs

def record_all_secs_spines(spines):
    """Record voltage at every spine head across all dendrites."""
    all_v = {}
    dists = []
    dends = []
    locs  = []
    i = 0
    for sec_name, sec_spines in spines.items():
        for spine_id, spine in sec_spines.items():
            all_v[i] = h.Vector()
            all_v[i].record(spine.head(0.5)._ref_v)
            dists.append(h.distance(spine.x, sec=spine.parent))
            dends.append(sec_name)
            locs.append(spine.x)
            i += 1
    return all_v, dists, dends, locs

# finds all unique spine locations on path to soma
# ignores spine if active
# records current mechanisms specified in that spine to synaptic event at a remote site
def record_all_path_secs_spine_i_mechs(cell, spines, dend_tree, dendrite, activated_spines, spine_mechs=['pas', 'kir', 'cal12', 'cal13', 'car', 'cav32', 'cav33', 'sk']):
    all_i_mechs = {}
    dists = []
    dends = []
    for dend in cell.allseclist:
        if dend.name() == dendrite:
            dendrite = dend
        # get path to soma
        if dendrite.name() != 'soma[0]':
            pathlist = path_finder(cell=cell, dend_tree=dend_tree, dend=dendrite)
            # Remove soma[0] from pathlist if it exists
            pathlist = [sec for sec in pathlist if sec.name() != 'soma[0]']
        else:
            pathlist = [dendrite]  

    # for each dendrite in path find unique locations corresponding to each seg of that dendrite
    i=0
    for sec in pathlist:
        sec_spines = list(spines[sec.name()].items())
        sec_spines_x = []
        for spine_i, spine_obj in sec_spines: 
            sec_spines_x.append(spine_obj.x)
        if sec_spines_x:
            for seg in sec:
                dends.append(sec.name())
                dist = h.distance(seg)
                dists.append(dist)
                loc = seg.x
                # Find the index of the value in sec_spines_x closest to loc
                idx = min(range(len(sec_spines_x)), key=lambda i: abs(sec_spines_x[i] - loc))
                spine_id, spine = sec_spines[idx]
                while spine in activated_spines:
                    idx += 1
                    spine_id, spine = sec_spines[idx]
                # for spines loc = 0.5 to record in the middle of the spine head
                all_i_mechs[i] = record_i_mechs(cell=cell, dend=spine.head, loc=0.5, return_currents=True, silent=True,
                    mechs=spine_mechs)
                i = i + 1

    return all_i_mechs, dists, dends
def extract_column(dict_df, column_name):
    """
    Extracts a specified column from each DataFrame in a dictionary.

    Parameters:
    dataframes (dict): A dictionary of DataFrames.
    column_name (str): The name of the column to extract.

    Returns:
    list: A list containing the extracted column from each DataFrame.
    """
    extracted_columns = []
    for key, df in dict_df.items():
        extracted_columns.append(df[column_name].values)
    return extracted_columns


# Iterate through each Vector in 'v' and convert to NumPy array
def vec2np(V):
    out = []
    # check if 'V' is a dictionary
    if isinstance(V, dict):
        # if 'V' is a dictionary, iterate over its values
        for vector in V.values():
            np_array = np.array(vector)
            out.append(np_array)
    # check if 'V' is a list
    elif isinstance(V, list):
        # If 'V' is a list, iterate directly over it
        for vector in V:
            np_array = np.array(vector)
            out.append(np_array)
    return out

def interpolate_3d(sec, seg_x):
    '''
    This function, interpolate_3d, interpolates the 3D coordinates and diameter for the center of a segment. 
    Use within record_all_3D to get the interpolated coordinates and diameter for each segment in cell.dendlist 
    and record the voltage at each segment's center.
    
    '''# Number of 3D points in the section
    n3d = int(h.n3d(sec=sec))
    
    # Segment's relative position along the section's total length
    seg_pos = seg_x * sec.L
    
    # Initialize variables to hold the interpolated values
    x, y, z, diam = None, None, None, None
    
    # Length along the section up to the current 3D point
    length = 0
    
    for i in range(n3d - 1):
        # Get the 3D coordinates and diameter of the current and next point
        x0, y0, z0, diam0 = h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec), h.diam3d(i, sec=sec)
        x1, y1, z1, diam1 = h.x3d(i + 1, sec=sec), h.y3d(i + 1, sec=sec), h.z3d(i + 1, sec=sec), h.diam3d(i + 1, sec=sec)
        
        # Calculate the distance between these two points
        point_dist = ((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)**0.5
        
        if length + point_dist >= seg_pos:
            # Segment's center falls between these two points, interpolate
            ratio = (seg_pos - length) / point_dist
            x = x0 + ratio * (x1 - x0)
            y = y0 + ratio * (y1 - y0)
            z = z0 + ratio * (z1 - z0)
            diam = diam0 + ratio * (diam1 - diam0)
            break
        
        length += point_dist
    
    return x, y, z, diam

def record_all_3D(cell):
    all_v = [] 
    cell_coordinates = []
    dists = []
    dends = []

    # Record for somatic section, only at the center
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Access the middle segment of the soma
        v_vec = h.Vector()
        v_vec.record(seg._ref_v)
        all_v.append(v_vec)

        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())  # You can distinguish soma from dendrites here if needed
        dists.append(h.distance(0.5, sec=sec))

    for sec in cell.dendlist:
        h('access ' + sec.name())
        # sec.nseg = int(h.n3d())

        for seg in sec:
            v_vec = h.Vector()
            v_vec.record(seg._ref_v)
            all_v.append(v_vec)

            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return all_v, cell_coordinates, dends, dists

def record_mechs_3D(cell, mechs):
    all_i_mechs = {}
    cell_coordinates = []
    dists = []
    dends = []

    # Record for somatic section, only at the center
    for sec in cell.somalist:
        h('access ' + sec.name())
        out = record_i_mechs(cell=cell, dend=sec, loc=0.5, return_currents=True, silent=True, mechs=mechs)
        all_i_mechs[0] = out[1:]  # drops first vector (time)
        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())
        dists.append(h.distance(0.5, sec=sec))

    i = 1
    for sec in cell.dendlist:
        h('access ' + sec.name())
        # sec.nseg = int(h.n3d())  # Ensure this is necessary and correct

        for seg in sec:
            loc = seg.x
            out = record_i_mechs(cell=cell, dend=sec, loc=loc, return_currents=True, silent=True, mechs=mechs)  # Update 'out' within the loop
            all_i_mechs[i] = out[1:]  # Append excluding the first element
            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
            i = i+1
    
    return all_i_mechs, cell_coordinates, dends, dists

def record_Ca_3D(cell):
    all_Ca = [] 
    cell_coordinates = []
    dists = []
    dends = []

    # Record for somatic section, only at the center
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Access the middle segment of the soma
        Ca_vec = h.Vector()
        Ca_vec.record(seg._ref_cai)
        all_Ca.append(Ca_vec)

        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())  # You can distinguish soma from dendrites here if needed
        dists.append(h.distance(0.5, sec=sec))

    for sec in cell.dendlist:
        h('access ' + sec.name())
        # sec.nseg = int(h.n3d())

        for seg in sec:
            Ca_vec = h.Vector()
            Ca_vec.record(seg._ref_cai)
            all_Ca.append(Ca_vec)

            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return all_Ca, cell_coordinates, dends, dists

def record_mechs_distr_3D(cell, mechs=['pas', 'kdr', 'naf', 'kaf', 'kas', 'kir', 'cal12', 'cal13', 'can', 'car', 'cav32', 'cav33', 'sk', 'bk']):
    
    dist_mechs_out = []
    cell_coordinates = []
    dists = []
    dends = []

    mech_refs = {
        'pas': 'g_pas',
        'kdr': 'gbar_kdr',
        'naf': 'gbar_naf',
        'kaf': 'gbar_kaf',
        'kas': 'gbar_kas',
        'kir': 'gbar_kir',
        'kcnq': 'gbar_kcnq',
        'cal12': 'pbar_cal12',
        'cal13': 'pbar_cal13',
        'can': 'pbar_can',
        'car': 'pbar_car',
        'cav32': 'pbar_cav32',
        'cav33': 'pbar_cav33',
        'sk': 'gbar_sk',
        'bk': 'gbar_bk'
    }

    # Results dictionary to hold conductance/permeability values
    results = {}

    for sec in cell.somalist:
        h('access ' + sec.name())
        out = []
        for mech in mechs:
            ref_attr = mech_refs.get(mech)
            if ref_attr and hasattr(sec(0.5), ref_attr):
                # Record the mechanism if it exists
                out.append(getattr(sec(0.5), ref_attr))
            else:
                print(f"warning: mechanism '{mech}' not recognized or not present at the specified location")
        dist_mechs_out.append(out)
        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())
        dists.append(h.distance(0.5, sec=sec))

    i = 1
    for sec in cell.dendlist:
        h('access ' + sec.name())
        # sec.nseg = int(h.n3d())  

        for seg in sec:
            loc = seg.x
            out = []
            for mech in mechs:
                ref_attr = mech_refs.get(mech)
                if ref_attr and hasattr(sec(loc), ref_attr):
                    # Record the mechanism if it exists
                    out.append(getattr(sec(loc), ref_attr))
                else:
                    print(f"warning: mechanism '{mech}' not recognized or not present at the specified location")
            dist_mechs_out.append(out)
            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))

        i = i+1
    
    df1 = pd.DataFrame(dist_mechs_out, columns=mechs)
    df2 = pd.DataFrame(cell_coordinates, columns=['secname', 'loc', 'x3d', 'y3d', 'z3d', 'dist', 'diam'])
        
    output = {
        'distributions': df1,
        'cell_coordinates': df2,
        'dists': dists,
        'dendrites': dends
    }          
     

    return output

def setup_impedance_measurements(cell):
    impedance_locations = []
    cell_coordinates = []
    dists = []
    dends = []

    # Setup for somatic sections
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Middle segment of the soma

        # Store information for impedance measurement
        impedance_locations.append((sec, seg.x))

        x, y, z, diam = interpolate_3d(sec, seg.x)
        cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
        dends.append(sec.name())  # Soma could be distinguished here if needed
        dists.append(h.distance(seg.x, sec=sec))

    # Setup for dendritic sections
    for sec in cell.dendlist:
        h('access ' + sec.name())
        # sec.nseg = int(h.n3d())  # Ensure there's a segment at each 3D point

        for seg in sec:
            # Store information for impedance measurement
            impedance_locations.append((sec, seg.x))

            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return impedance_locations, cell_coordinates, dends, dists

# routine to calculate spine_per_length_dict values below
def solve_spine_per_length(target_total_spines, cell_type, specs, soma_diameter=None, frequency=2000, d_lambda=0.05, initial_tol=5, max_iterations=100, iteration_threshold=20, dend2remove=None):
    lower_bound = 0
    upper_bound = 4  # This might need to be adjusted based on your specific use case
    tolerance = initial_tol
    spine_per_length = (upper_bound + lower_bound) / 2
    iterations = 0
    iteration_since_last_adjustment = 0

    while iterations < max_iterations:
        _, spines, _ = cell_build(cell_type, specs, addSpines=True, spine_per_length=spine_per_length, soma_diameter=soma_diameter, frequency=frequency, d_lambda=d_lambda, verbose=False, dend2remove=dend2remove)
        current_total_spines = sum(len(spine_list) for spine_list in spines.values())

        # Check if the current total is within the tolerance of the target
        if abs(current_total_spines - target_total_spines) <= tolerance:
            return spine_per_length  # Return the found spine_per_length value

        # Adjust spine_per_length based on whether the current total is less than or greater than the target
        elif current_total_spines < target_total_spines:
            lower_bound = spine_per_length
        else:
            upper_bound = spine_per_length

        spine_per_length = (upper_bound + lower_bound) / 2
        iterations += 1
        iteration_since_last_adjustment += 1

        # Increase tolerance if no solution is found within iteration_threshold
        if iteration_since_last_adjustment >= iteration_threshold:
            tolerance *= 2  # Double the tolerance
            iteration_since_last_adjustment = 0  # Reset the counter for adjustments

    # If the function reaches this point, it means no solution was found within max_iterations
    # You might want to return a special value or raise an exception here
    return None

def update_dictionary(data_dict, protocol, i_recording_site, v_recording_site, v_all_3D, Ca_all_3D, imp_all_3D, i_mechs_3D, vspine, v_spine_activated, vdend, v_dend_activated, vsoma, v_dend_tree, v_spine_tree, Ca_spine, Ca_dend, Ca_soma, timing, i_mechs_dend, i_mechs_dend_tree, i_mechs_spine_tree, v_branch, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, record_dist, time, record_currents=False, record_spine=False, spine_dist=None, cap=None):
    
    data_dict['i_recording_site'].append(i_recording_site)
    data_dict['v_recording_site'].append(v_recording_site)
    data_dict['v_all_3D'][protocol] = v_all_3D
    data_dict['Ca_all_3D'][protocol] = Ca_all_3D
    data_dict['imp_all_3D'][protocol] = imp_all_3D
    data_dict['i_mechs_3D'][protocol] = i_mechs_3D
    data_dict['v_dend_tree'][protocol] = v_dend_tree
    data_dict['v_spine_tree'][protocol] = v_spine_tree
    data_dict['v_branch'][protocol] = v_branch
    data_dict['vsoma'].append(vsoma)
    data_dict['vdend'].append(vdend)
    data_dict['v_dend_activated'][protocol] = v_dend_activated
    data_dict['vspine'].append(vspine)
    data_dict['v_spine_activated'][protocol] = v_spine_activated
    
    data_dict['Ca_soma'].append(Ca_soma)
    data_dict['Ca_dend'].append(Ca_dend)
    data_dict['Ca_spine'].append(Ca_spine)
    
    data_dict['timing'].append(timing)

    data_dict['i_mechs_dend'][protocol] = i_mechs_dend
    data_dict['i_mechs_dend_tree'][protocol] = i_mechs_dend_tree
    data_dict['i_mechs_spine_tree'][protocol] = i_mechs_spine_tree
    
    data_dict['record_dists'].append(record_dist)
    data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
    data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
    data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
    data_dict['g_gaba'][protocol] = pd.DataFrame(gaba_conductances).transpose()
    
    data_dict['spine_dist'].append(spine_dist)
    data_dict['capacitance'].append(cap)
        
    # if record_currents:
    #     data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
    #     data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
    #     data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
    #     data_dict['g_gaba'][protocol] = pd.DataFrame(gaba_conductances).transpose()

    data_dict['timestamp'][protocol] = time
    
    
def update_data_dictionary(data_dict, protocol, v_all_3D, Ca_all_3D, imp_all_3D, i_mechs_3D, vspine, v_spine_activated, vdend, v_dend_activated, vsoma, v_dend_tree, v_spine_tree, Ca_spine, Ca_dend, Ca_soma, timing, i_mechs_dend, i_mechs_dend_tree, i_mechs_spine_tree, v_branch, ampa_currents, nmda_currents, gaba_currents, gaba_conductances, record_dist, time, record_currents=False, record_spine=False, spine_dist=None, cap=None):
    
    data_dict['v_all_3D'][protocol] = v_all_3D
    data_dict['Ca_all_3D'][protocol] = Ca_all_3D
    data_dict['imp_all_3D'][protocol] = imp_all_3D
    data_dict['i_mechs_3D'][protocol] = i_mechs_3D
    data_dict['v_dend_tree'][protocol] = v_dend_tree
    data_dict['v_spine_tree'][protocol] = v_spine_tree
    data_dict['v_branch'][protocol] = v_branch
    data_dict['vsoma'].append(vsoma)
    data_dict['vdend'].append(vdend)
    data_dict['v_dend_activated'][protocol] = v_dend_activated
    data_dict['vspine'].append(vspine)
    data_dict['v_spine_activated'][protocol] = v_spine_activated
    
    data_dict['Ca_soma'].append(Ca_soma)
    data_dict['Ca_dend'].append(Ca_dend)
    data_dict['Ca_spine'].append(Ca_spine)
    
    data_dict['timing'].append(timing)

    data_dict['i_mechs_dend'][protocol] = i_mechs_dend
    data_dict['i_mechs_dend_tree'][protocol] = i_mechs_dend_tree
    data_dict['i_mechs_spine_tree'][protocol] = i_mechs_spine_tree
    
    data_dict['record_dists'].append(record_dist)
    data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
    data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
    data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
    data_dict['g_gaba'][protocol] = pd.DataFrame(gaba_conductances).transpose()
    
    data_dict['spine_dist'].append(spine_dist)
    data_dict['capacitance'].append(cap)
        
    # if record_currents:
    #     data_dict['i_ampa'][protocol] = pd.DataFrame(ampa_currents).transpose()
    #     data_dict['i_nmda'][protocol] = pd.DataFrame(nmda_currents).transpose()
    #     data_dict['i_gaba'][protocol] = pd.DataFrame(gaba_currents).transpose()
    #     data_dict['g_gaba'][protocol] = pd.DataFrame(gaba_conductances).transpose()

    data_dict['timestamp'][protocol] = time

def plots_return(v_tree, vspine, dists, spine_dist, num_gluts, start_time=150, burn_time=140, 
            dt=0.025, xaxis_range=[0,100], Nsim_plot=False, Nsim_save=False, sim_image_path=None, 
            time=None, width=1000, height=360, plot_color='grey', palette='oleron'):
    # only do if want to view each sim or save sim graphs
    if Nsim_plot or Nsim_save:
       # normalise tree data
        norm_v_all = []
        dists1 = dists
        for v in v_tree:
            norm_v_all.append( normalise(v, start_time, burn_time, dt) )
        if num_gluts == 1:
            norm_v_all.append(normalise(vspine, start_time, burn_time, dt)  )
            dists1.append(spine_dist)

        # find peak values
        peak_v = []
        for v in norm_v_all:
            peak_v.append(v.max())

        # isolate peak
        norm_peak_v = []
        for v in peak_v:
            norm_peak_v.append(v/max(peak_v))

        # only do if want to view each sim or save sim graphs
        if Nsim_plot or Nsim_save:
            fig1 = plot5(X=norm_v_all, dt=dt, dists=dists1, xaxis_range=xaxis_range,
                         yaxis_range=[-0.2, math.ceil(max(peak_v))], normalised=False,
                         palette=palette)

            fig2 = plot6(y=norm_peak_v, x=dists1, xaxis_range=[300,0], yaxis_range=[0,1.01],
                         palette=palette)

            if Nsim_plot:
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=('Dendritic voltage traces', 'Peak voltage by distance'),
                    horizontal_spacing=0.12
                )

                for trace in fig1.data:
                    trace.showlegend = True
                    trace.legendgroup = trace.name
                    fig.add_trace(trace, row=1, col=1)

                for trace in fig2.data:
                    trace.showlegend = False
                    trace.legendgroup = trace.name
                    fig.add_trace(trace, row=1, col=2)

                fig.update_layout(
                    width=width,
                    height=height,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    template='plotly_white',
                    font=dict(family='Calibri', size=10, color=plot_color),
                    showlegend=True,
                    legend=dict(
                        title='distance (um)',
                        x=1.02,
                        y=1,
                        xanchor='left',
                        yanchor='top',
                        font=dict(color=plot_color),
                        title_font=dict(color=plot_color),
                        bgcolor='rgba(0,0,0,0)',
                        borderwidth=0
                    )
                )

                fig.update_xaxes(showgrid=False, zeroline=False, ticks='outside', showline=True,
                                 mirror=False, linecolor=plot_color, tickcolor=plot_color,
                                 tickfont=dict(color=plot_color), title_font=dict(color=plot_color))

                fig.update_yaxes(showgrid=False, zeroline=False, ticks='outside', showline=True,
                                 mirror=False, linecolor=plot_color, tickcolor=plot_color,
                                 tickfont=dict(color=plot_color), title_font=dict(color=plot_color))

                fig.update_xaxes(title_text='time (ms)', range=xaxis_range, row=1, col=1)
                fig.update_yaxes(title_text='V (mV)', range=[-0.2, math.ceil(max(peak_v))], row=1, col=1)

                fig.update_xaxes(title_text='distance (um)', range=[300, 0], row=1, col=2)
                fig.update_yaxes(title_text='normalised amplitude', range=[0, 1.01], row=1, col=2)

                for annotation in fig.layout.annotations:
                    annotation.font.color = plot_color
                    annotation.font.size = 12

                fig.update_traces(line_simplify=False)
                fig.show()

            if Nsim_save:
                for fig, fig_name in [(fig1, 'fig1'), (fig2, 'fig2')]:
                    fig.write_html('{}/{}_{}.html'.format(sim_image_path, fig_name, time))

def setup_impedance_measurements(cell):
    impedance_locations = []
    cell_coordinates = []
    dists = []
    dends = []

    # Setup for somatic sections
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Middle segment of the soma

        # Store information for impedance measurement
        impedance_locations.append((sec, seg.x))

        x, y, z, diam = interpolate_3d(sec, seg.x)
        cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
        dends.append(sec.name())  # Soma could be distinguished here if needed
        dists.append(h.distance(seg.x, sec=sec))

    # Setup for dendritic sections
    for sec in cell.dendlist:
        h('access ' + sec.name())
        # sec.nseg = int(h.n3d())  # Ensure there's a segment at each 3D point

        for seg in sec:
            # Store information for impedance measurement
            impedance_locations.append((sec, seg.x))

            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return impedance_locations, cell_coordinates, dends, dists

def all_3D(cell):
    seg_x = [] 
    cell_coordinates = []
    dists = []
    dends = []

    # Record for somatic section, only at the center
    for sec in cell.somalist:
        h('access ' + sec.name())
        seg = sec(0.5)  # Access the middle segment of the soma
        seg_x.append(seg.x)

        x, y, z, diam = interpolate_3d(sec, 0.5)  # Use 0.5 to refer to the center of the section
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])
        dends.append(sec.name())  # You can distinguish soma from dendrites here if needed
        dists.append(h.distance(0.5, sec=sec))

    for sec in cell.dendlist:
        h('access ' + sec.name())
        # sec.nseg = int(h.n3d())

        for seg in sec:
            seg_x.append(seg.x)
            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])
            dends.append(sec.name())
            dists.append(h.distance(seg.x, sec=sec))
    
    return seg_x, cell_coordinates, dends, dists
   
    
def get_(sim, _dict):
    letter = sim[-1]
    key_pattern = f'{letter}'
    return _dict.get(key_pattern, "Default Value")
    
def compare_last_digit(string, integer):
    """
    Compare the digits *after* the first three digits in a string with one or more integers.

    :param string: The string whose digits (after the first three) will be compared.
    :param integer: A single integer or a list/tuple/set of integers to compare against.
    :return: True if the digits after the first three match any of the provided integers, False otherwise.
    """
    if not string:
        return False
    if ' ' in string:
        last_token = string.split()[-1]
        d = ''.join(ch for ch in last_token if ch.isdigit())
        if not d:
            return False
        value = int(d)
    else:
        digits = ''.join(ch for ch in string if ch.isdigit())
        if len(digits) <= 3:
            return False
        value = int(digits[3:])
    if isinstance(integer, (list, tuple, set)):
        return value in map(int, integer)
    return value == int(integer)
    
    
def max_and_sign(arrays):
    overall_max = 0
    overall_sign = 0
    for arr in arrays:
        # Find the maximum absolute value in this array
        current_max = np.max(np.abs(arr))
        if current_max > overall_max:
            overall_max = current_max
            # Get the index of that maximum absolute value in the current array
            idx = np.argmax(np.abs(arr))
            overall_sign = np.sign(arr[idx])
    return overall_max, overall_sign

def _spike_indices(y, dt, high_thr=0.0, low_thr=-5.0, refractory_ms=2.0):
    ref_n = max(1, int(round((refractory_ms/1000.0)/dt)))
    idx = []
    n = len(y)
    i = 0
    armed = False
    while i < n:
        if not armed and y[i] <= low_thr:
            armed = True
        if armed and y[i] > high_thr:
            idx.append(i)
            armed = False
            i += ref_n
            continue
        i += 1
    return np.array(idx, dtype=int)

def count_spikes(ydict, dt, high_thr=0.0, low_thr=-5.0, refractory_ms=2.0):
    spike_idx = [_spike_indices(y, dt, high_thr, low_thr, refractory_ms) for y in ydict]
    return [len(ix) for ix in spike_idx], spike_idx

# simple hex → rgba
def hex_to_rgba(hex_color, alpha=1):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return f'rgba({r},{g},{b},{alpha:.2f})'

# simple alpha wrapper
def alphaize(cols, alpha=1):
    return [hex_to_rgba(c, alpha) for c in cols]

# simple colour interpolation
def color_ramp_palette(colors, n):
    c = np.array([[int(c[i:i+2], 16) for i in (1, 3, 5)] for c in colors])
    x = np.linspace(0, 1, len(c))
    xi = np.linspace(0, 1, n)
    out = np.zeros((n, 3))
    for j in range(3):
        out[:, j] = np.interp(xi, x, c[:, j])
    return [f'#{int(r):02X}{int(g):02X}{int(b):02X}' for r, g, b in out]

def palette_cols(name, n, alpha=1, reverse=False):
    name = name.lower()
    palettes = {
        'jet': ['#00007F', '#0000FF', '#007FFF', '#00FFFF', '#7FFF7F', '#FFFF00', '#FF7F00', '#FF0000', '#7F0000'],
        'viridis': ['#440154','#3b528b','#21908d','#5ec962','#fde725'],
        'cividis': ['#00204c','#414487','#7e03a8','#a5f86b','#ffffc0'],
        'puor': ['#7f3b08','#fdb863','#f7f7f7','#b2abd2','#5e3c99'],
        'brbg': ['#543005','#bf812d','#f6e8c3','#c7eae5','#35978f','#003c30'],
        'roma': ['#7E1700','#8C360A','#984F13','#A3651E','#AD7B27','#B99235','#C4AB4A','#CEC56C','#D1DC94',
                 '#C9E8B5','#B4E9CC','#95E0D6','#73CED5','#54B8D0','#3DA2C9','#2F8CBF','#2677B7','#1F61AD','#023198'],
        'vik': ['#001260','#01276E','#023B7B','#055189','#166898','#3983AB','#629FBD','#8CB9CF','#B7D3E1','#DFE5E9',
                '#EDDCD2','#E5C2AE','#DAA78A','#CF8F69','#C6774A','#BA5F2C','#A5400F','#882506','#6F1107','#590007'],
        'batlow': ['#001959','#0B2D5D','#103D5E','#134B61','#195661','#25625F','#366A58','#49714E','#5F7842','#757E37',
                   '#8E852D','#A88B2B','#C29037','#D9954A','#ED9A63','#F9A382','#FDADA1','#FDB7BE','#FCC1DB','#F9CCF9'],
        'berlin': ['#9EB0FF','#7FABF0','#5DA5DD','#4093C0','#307A9E','#25617E','#1C4960','#153342','#101F27','#121112',
                   '#200A03','#2F0E00','#421300','#571B06','#742C16','#904430','#AB5D4E','#C6776C','#E2928C','#FFACAC'],
        'batlowk': ['#000000','#0B1720','#132738','#1A384E','#204960','#285A6E','#336B79','#437B81','#598A84','#729885',
                    '#8EA584','#A9B282','#C3BF81','#DACD80','#EFE27E','#FEF076','#FFF5A0','#FFFBD1','#FFFFF5'],
        'batloww': ['#FFFFFF','#F6F2E7','#E7D9B8','#D3C183','#B9AA57','#9C943A','#7E7F2E','#60702F','#436636','#275E3E',
                    '#0A5647','#004E4F','#004757','#00415E','#003A64','#003469','#002D6E','#002773','#002177','#001A7B'],
        'broc': ['#00224E','#063864','#185074','#2E697F','#498389','#6A9C8F','#8CB49A','#AAC9A9','#C6DCBC','#E1EDD2',
                 '#EFECD8','#EBD1C7','#E2B5B1','#D6989C','#C67A89','#B05E79','#954366','#762A50','#56183C','#380D29'],
        'cork': ['#2C004E','#3E1560','#512C71','#65407F','#7A568C','#906C98','#A782A2','#BD99AB','#D1B1B5','#E2C9C2',
                 '#F0E0D2','#EFE0D6','#E1C8CA','#CEB1BE','#B59BB1','#9787A3','#776693','#56457F','#3B2868','#241150'],
        'bam': ['#03006F','#17289C','#315BBE','#5292CF','#78BDC8','#9FD5AD','#C0E091','#E1D875','#F9C85D','#FFB44B',
                '#FEA13F','#F18E3E','#D97C40','#B76A45','#8E594B','#624A4F','#3C3B4E','#232A46','#0F1538','#000028'],
        'devon': ['#071C1E','#0B2E27','#12422D','#20552F','#36672F','#527730','#728533','#96923D','#BA9F4F','#D5B367',
                  '#E7C98D','#EFE0B5','#EEEBCB','#E0E2C5','#C6CDB1','#A3B39B','#7A9685','#557A72','#365F62','#204552'],
        'oleron': ['#001D47','#073463','#0F4B7E','#1C6296','#3179AB','#4D8FBB','#6AA3C6','#88B7CD','#A5C9D0','#C2D9D0',
                   '#E1E5CB','#F8EBC7','#FDE2B7','#F7D1A0','#ECB88C','#DC9E7B','#C9836D','#B16B5F','#965555','#7B3E49'],
        'nuuk': ['#00204C','#08315E','#14426E','#23527C','#35628A','#4A7197','#64809F','#7E8FA4','#999EA6','#B5ACA6',
                 '#D0BBA4','#E6CAA4','#F4DBA7','#FAEDB1','#FCFDC2','#F8FFDA','#F0FFF0','#E1FFF8','#C9FFFF','#AAFFFF'],
        'grayc': ['#000000','#1B1B1B','#343434','#4E4E4E','#686868','#828282','#9C9C9C','#B6B6B6','#D0D0D0','#EAEAEA','#FFFFFF']
    }
    if name not in palettes:
        raise ValueError(f"Palette must be one of: {', '.join(palettes.keys())}")
    cols = color_ramp_palette(palettes[name], n)
    if reverse:
        cols = list(reversed(cols))
    return alphaize(cols, alpha)
