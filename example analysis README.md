<h1 align="center">Example Analysis Notebook</h1>

This file summarizes the workflow in `analysis notebooks/example analysis.ipynb`.

<h2 align="center">Settings And Paths</h2>

The first code block selects the simulation, imports the shared project settings and helper functions, defines the simulation-data directory, and creates the analysis-output directory. Change `sim`, `cell_type`, `model`, `downsample`, `external`, and `external_name` as needed for a different analysis notebook.

The paths shown below are `macOS` examples. When `external = False`, data are read from the local repository under the `macOS` example path `~/Documents/Repositories/msNEURON_Belal2026`. When `external = True`, data are read from the `macOS` external-volume example path `/Volumes/<external_name>/Repositories/msNEURON_Belal2026`. On other systems, replace these examples with the path to your local or external copy of the repository. Analysis outputs are written inside the repository under `analysis/<cell_type>/<sim>`.

```python
sim = 'sim2256'

# settings
cell_type='ispn'
model = 3

import os, sys
# compute absolute path of main folder (once)
main_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# set CWD or add to path
os.chdir(main_dir)           # only once, to main folder
sys.path.insert(0, main_dir) # ensures imports work

# then import/run settings
%run -i settings.py

from analysis_functions import *

# Get the current working directory
current_wd = os.getcwd()
# 'sim' + str(sim)

downsample = True

home = os.path.expanduser('~')

# Set working directory
external = False
external_name = 'MacOS10'
target = f'model{model}'

if not external:
    if downsample:
        wd = home + '/Documents/Repositories/msNEURON_Belal2026/downsample/' + cell_type + '/' + target + '/physiological/simulations'
    else:
        wd = home + '/Documents/Repositories/msNEURON_Belal2026/' + cell_type + '/' + target + '/physiological/simulations'
else:
    if downsample:
        wd = '/Volumes/' + external_name + '/Repositories/msNEURON_Belal2026/downsample/' + cell_type + '/' + target + '/physiological/simulations'
    else:
        wd = '/Volumes/' + external_name + '/Repositories/msNEURON_Belal2026/' + cell_type + '/' + target + '/physiological/simulations'


# create path to directory to save analysis
base_path = os.path.join(home, 'Documents', 'Repositories', 'msNEURON_Belal2026', 'analysis')
sim_image_path = os.path.join(base_path, cell_type, sim)
os.makedirs(sim_image_path, exist_ok=True)

save = True
```

<h2 align="center">Basic Plot Settings</h2>

This block standardizes line width, figure dimensions, projection plane, dendrite selection, and downsampling. The iSPN and dSPN branches choose different projection settings and example dendrites.

```python
# Basic plot settings
s = 5 # splprep fits a parametric B-spline to the 2D points
lwd = 1 # standardise line width

height1 = 600
width1 = 600

height2 = 600
width2 = 800

if cell_type == 'ispn':
    plane='yx'
    mirror=False
    dend_name = 'dend[12]'

else:
    plane='xy'
    mirror=False
    dend_name = 'dend[7]'

if downsample:
    ds = 10
```

<h2 align="center">Load Simulation Data</h2>

This block loads the simulation pickle files into `sim_data`, checks the ordering of loaded simulation branches, and reports memory use. `load_data_dicts(...)` will use the local cache if the source is on an external drive and has already been copied.

```python
# load simulations
sims_dir = os.path.join(wd, sim)

sim_data = load_data_dicts(wd=wd, sim=sim, cell_type=cell_type, verbose=True)

# check files loaded in correct order
for name in sim_data.keys():
    print(name)

report_memory(verbose=True)
```

The resulting structure is:

```text
sim_data
  -> simulation branch, e.g. a, b, c
      -> variable name, e.g. metadata, vsoma, vdend, v_all_3D
          -> data object
```

<h2 align="center">Morphology Coordinates</h2>

This block obtains cell coordinates for plotting. If the simulation output already contains 3D coordinate data, those coordinates are used. If not, the cell is rebuilt and the soma and dendritic coordinates are reconstructed.

```python
# check if coordinates exist
coords_exist = summarise_cell_coordinates(sim_data)

# if so load from v_all_3D else retrieve from cell_build
# this is because the methods to retrieve the cell_coordinates differ so it is important when plotting 3D heatmaps that the correct coordinates are used
# the method to default will only be used when the simulations have no 3d data
if coords_exist:
    first_sim = next(iter(sim_data.values()))
    v_all_3D = first_sim['v_all_3D']
    first_key = next(iter(v_all_3D))

    cell_coordinates = np.array(v_all_3D[first_key]['cell_coordinates'], dtype=object)
    _, _, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, verbose=False, dend2remove=None)
else:
    cell, spines, dend_tree = cell_build(cell_type=cell_type, specs=specs, addSpines=True, spine_per_length=spine_per_length, frequency=frequency, d_lambda=d_lambda, verbose=False, dend2remove=None)
    cell_coordinates = []

    for sec in cell.somalist:
        h('access ' + sec.name())
        x, y, z, diam = interpolate_3d(sec, 0.5)
        cell_coordinates.append([sec.name(), 0.5, x, y, z, h.distance(0.5, sec=sec), diam])

    for sec in cell.dendlist:
        for seg in sec:
            x, y, z, diam = interpolate_3d(sec, seg.x)
            cell_coordinates.append([sec.name(), seg.x, x, y, z, h.distance(seg.x, sec=sec), diam])

    cell_coordinates = np.array(cell_coordinates, dtype=object)


fig_morphology = morphology_plot(cell_coordinates, dend_tree, lwd=lwd, s=s, color='grey', height=height1, width=width1, plane=plane, mirror=False)

fig_morphology.show(config={"toImageButtonOptions": {"format": "svg"}})
if save:
    fig_morphology.write_image(f"{sim_image_path}/morphology.svg", format='svg', scale=3)
```

Output:

```text
morphology.svg
```

<p align="center">
  <img src="./example%20images/example%20analysis/morphology.svg" width="650" alt="Cell morphology">
</p>

<h2 align="center">Synapse Locations</h2>

This block reads glutamatergic and GABAergic input locations from metadata, converts those locations into plotting coordinates, overlays them on the morphology, and optionally saves the annotated morphology.

```python
# Morphology plot with synaptic locations

# extract names and locations of GLUT and GABA inputs from metadata
# using last simulation to get all GABA locations used (fixed)
dend_gaba_all = list(sim_data.values())[-1]['metadata']['dend_gaba']
gaba_locations_all = list(sim_data.values())[-1]['metadata']['gaba_locations']

dend_glut = list(sim_data.values())[-1]['metadata']['dend_glut']
glut_locations = list(sim_data.values())[-1]['metadata']['glutamate_locations']

if len(dend_glut) != len(glut_locations):
    dend_glut = dend_glut * len(glut_locations)

# indices
idxs_gaba = get_coord_index(cell_coordinates=cell_coordinates, target_dendrite=dend_gaba_all, target_location=gaba_locations_all)
idxs_glut = get_coord_index(cell_coordinates=cell_coordinates, target_dendrite=dend_glut, target_location=glut_locations)

# coordinates
coords_gaba = get_coord_index_interp(cell_coordinates=cell_coordinates, target_dendrite=dend_gaba_all, target_location=gaba_locations_all)
coords_glut = get_coord_index_interp(cell_coordinates=cell_coordinates, target_dendrite=dend_glut, target_location=glut_locations)

# with interpolation
fig_morphology = morphology_plot(cell_coordinates=cell_coordinates, dend_tree=dend_tree, idxs=[coords_gaba, coords_glut],
                                 title='', idxs_colors = ['#5393CF', '#CD5C5C'], dot_size=[6,6], lwd=lwd, s=s, color='grey',
                                 height=height1, width=width1, plane=plane, mirror=mirror, alpha=[0.5, 0.25])

fig_morphology.show(config={"toImageButtonOptions": {"format": "svg"}})
if save:
    fig_morphology.write_image(f"{sim_image_path}/morphology2.svg", format='svg', scale=3)

compare_synapse_coords("Glutamatergic", dend_glut, glut_locations, coords_glut)
compare_synapse_coords("GABAergic", dend_gaba_all, gaba_locations_all, coords_gaba)
```

Output:

```text
morphology2.svg
```

<p align="center">
  <img src="./example%20images/example%20analysis/morphology2.svg" width="650" alt="Cell morphology with synapse locations">
</p>

<h2 align="center">Voltage Traces</h2>

This block collects somatic and dendritic voltage traces from `sim_data`, adjusts `dt` for downsampled simulations, plots the traces with `plot3_dt(...)`, and saves the soma and dendrite figures.

```python
# Simple output traces
# downsample plots if original simulation is sampled at high rate eg 0.025 ms
# if simulation was subsequently downsampled then no need to downsample plots

dt = next(iter(sim_data.values()))['metadata']['dt']
sim_time = next(iter(sim_data.values()))['metadata']['sim_time']
sim_time = 400 # override for plots

if downsample:
    ds_plot = 1
    dt = ds * dt

else:
    ds_plot = 10


Vsoma = []
Vdend = []

for variables in sim_data.values():
    if 'vsoma' in variables:
        Vsoma.append(variables['vsoma'])
    if 'vdend' in variables:
        Vdend.append(variables['vdend'])

Vsoma_fig = plot3_dt(Vsoma, yaxis='V (mV)', stim_time = 150, sim_time=sim_time, title='somatic voltage', lwd=lwd,
                     yrange=[-90, -59], yabline = [-86, -60], black_trace=0, gray_trace=1, black_shift=100,
                     x_err_bar=50, y_err_bar=5, baseline=50, dt=dt, ds=ds_plot, height=height2, width=width2)

Vdend_fig = plot3_dt(Vdend, yaxis='V (mV)', stim_time = 150, sim_time=sim_time, title='dendritic voltage', lwd=lwd,
                     yrange=[-86, -20], yabline = [-86, -30], black_trace=0, gray_trace=1, black_shift=100,
                     x_err_bar=50, y_err_bar=10, baseline=50, dt=dt, ds=ds_plot, height=height2, width=width2)

Vsoma_fig.show(config={"toImageButtonOptions": {"format": "svg"}})
Vdend_fig.show(config={"toImageButtonOptions": {"format": "svg"}})

if save:
    Vsoma_fig.write_image(f"{sim_image_path}/Vsoma.svg", format='svg', scale=3)
    Vdend_fig.write_image(f"{sim_image_path}/Vdend.svg", format='svg', scale=3)
```

Outputs:

```text
Vsoma.svg
Vdend.svg
```

<div align="center">
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/Vsoma.svg" alt="Somatic voltage traces">
  </div>
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/Vdend.svg" alt="Dendritic voltage traces">
  </div>
</div>

<h2 align="center">Stimulus Timing</h2>

This block reconstructs the stimulation timing for downsampled simulations, or reads it from metadata for non-downsampled simulations. It then plots the relative timing of glutamatergic and GABAergic inputs.

```python
if downsample:
    # reconstruct
    timing_range = range(120,301,10)
    if sim in ['2251', '2253', '2255', '2257', '2259']:
        timing_range = range(120,301,1)

    timing_range = np.insert(timing_range, 0, [150, 150])

else:
    timing_range = next(iter(sim_data.values()))['metadata']['timing_range']


stimulation_paradigm_fig = plot3_timing(timing_range=timing_range, sim_time=sim_time, black_trace=0, gray_trace=1,
                 palette='oleron', alpha=1, reverse=False, black_shift=100, baseline=0,
                 width=width1, height=200, lwd=lwd, tick_height=5, title='Stimulus timing (ms)')

stimulation_paradigm_fig.show(config={"toImageButtonOptions": {"format": "svg"}})
if save:
    stimulation_paradigm_fig.write_image(f"{sim_image_path}/stimulation_paradigm.svg", format='svg', scale=3)
```

Output:

```text
stimulation_paradigm.svg
```

<p align="center">
  <img src="./example%20images/example%20analysis/stimulation_paradigm.svg" width="650" alt="Stimulation timing">
</p>

<h2 align="center">Voltage From 3D Data</h2>

This block extracts voltage traces directly from `v_all_3D` using the soma coordinate and the recorded dendritic coordinate. It provides an independent way to plot soma and dendritic voltage from the 3D data arrays.

```python
record_dendrite = next(iter(sim_data.values()))['metadata']['record_dendrite']
record_location = next(iter(sim_data.values()))['metadata']['record_location']

# get indices for the recording location
record_coords = get_coord_index_interp(cell_coordinates=cell_coordinates, target_dendrite=record_dendrite, target_location=record_location)
idx = get_coord_index(cell_coordinates=cell_coordinates, target_dendrite=record_dendrite, target_location=record_location)

compare_synapse_coords("Glutamatergic", record_dendrite, record_location, record_coords)
compare_synapse_coords("Glutamatergic", record_dendrite, record_location, cell_coordinates[idx][0:5,])

# Simple output traces can also be obtained from 'v_all_3D'
idx1 = get_coord_index(cell_coordinates=cell_coordinates, target_dendrite='soma[0]', target_location=0.5)
idx2 = get_coord_index(cell_coordinates=cell_coordinates, target_dendrite=record_dendrite, target_location=record_location)

Vsoma1 = []
Vdend1 = []

for sim_vars in sim_data.values():
    v_all = sim_vars['v_all_3D']
    for sim_block in v_all.values():
        v_arrays = sim_block['v']
        Vsoma1.append(v_arrays[idx1])
        Vdend1.append(v_arrays[idx2])

Vsoma1_fig = plot3_dt(Vsoma1, yaxis='V (mV)', stim_time = 150, sim_time=sim_time, title='somatic voltage', lwd=lwd,
                     yrange=[-90, -55], yabline = [-86, -60], black_trace=0, gray_trace=1, black_shift=100,
                     x_err_bar=50, y_err_bar=5, baseline=50, dt=dt, ds=ds_plot, height=height2, width=width2)

Vdend1_fig = plot3_dt(Vdend1, yaxis='V (mV)', stim_time = 150, sim_time=sim_time, title='dendritic voltage', lwd=lwd,
                     yrange=[-86, -20], yabline = [-86, -30], black_trace=0, gray_trace=1, black_shift=100,
                     x_err_bar=50, y_err_bar=10, baseline=50, dt=dt, ds=ds_plot, height=height2, width=width2)

Vsoma1_fig.show(config={"toImageButtonOptions": {"format": "svg"}})
Vdend1_fig.show(config={"toImageButtonOptions": {"format": "svg"}})
```

<h2 align="center">3D Voltage Heat Maps</h2>

This block computes the peak voltage at every 3D coordinate, plots a full-cell voltage heat map, and plots a dendrite-restricted heat map. Downsampling is kept modest because it can visibly alter the heat-map geometry.

```python
# 3D heat map
idx = 0

Vpeaks_3D  = []

for sim_vars in sim_data.values():
    v_all = sim_vars['v_all_3D']
    max_vals = []
    for sim_block in v_all.values():
        v_arrays = sim_block['v']
        max_vals.extend([np.max(trace) for trace in v_arrays])
    Vpeaks_3D.append(np.array(max_vals))

# be careful with too much downsampling; it can really alter the image
V2D_fig  = heatmap2D(cell_coordinates=cell_coordinates, dend_tree=dend_tree, z=Vpeaks_3D[idx], palette='oleron', reverse=False, alpha=1,
                     lwd=lwd, show_bar=True, title='', zmin=-85, zmax=-30, height=height1, width=width1, scale_bar=50,
                     x_range=[-125, 175], y_range=[-150, 150], plane=plane, mirror=False, s=s, ds=2)

V2D_path_fig  = heatmap2D(cell_coordinates=cell_coordinates, dend_tree=dend_tree, dend_name=dend_name, z=Vpeaks_3D[idx], palette='oleron', reverse=False, alpha=1,
                     lwd=lwd, show_bar=True, title='', zmin=-85, zmax=-30, height=height1, width=width1, scale_bar=50,
                     x_range=[-125, 175], y_range=[-150, 150], plane=plane, mirror=False, s=s, ds=2)


V2D_fig.show(config={"toImageButtonOptions": {"format": "svg"}})
V2D_path_fig.show(config={"toImageButtonOptions": {"format": "svg"}})

if save:
    V2D_fig.write_image(f"{sim_image_path}/V2D_{idx}.svg", format='svg', scale=3)
    V2D_path_fig.write_image(f"{sim_image_path}/V2D_path_{idx}.svg", format='svg', scale=3)
```

Outputs:

```text
V2D_<idx>.svg
V2D_path_<idx>.svg
```

<p align="center"><strong>GLUT only</strong></p>

<div align="center">
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/V2D_0.svg" alt="Full-cell voltage heat map, timing index 0">
  </div>
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/V2D_path_0.svg" alt="Dendritic path voltage heat map, timing index 0">
  </div>
</div>

<p align="center"><strong>GLUT + GABA &Delta;t = t<sub>GLUT</sub> - t<sub>GABA</sub> = -10 ms</strong></p>

<div align="center">
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/V2D_6.svg" alt="Full-cell voltage heat map, timing index 6">
  </div>
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/V2D_path_6.svg" alt="Dendritic path voltage heat map, timing index 6">
  </div>
</div>

<h2 align="center">Voltage Versus Distance</h2>

This block extracts a dendritic path, collects coordinate distances along that path, maps voltage peaks onto those distances, and plots voltage as a function of distance from soma.

```python
root_name = dend_name
target_dend = 'dend[15]'
extracted = extract_dend_to_target(dend_tree, root_name, target_dend)
extracted

# Rebuild idxs from the new extracted structure
idxs = []
for outer_list in extracted:
    outer_idxs = []
    for path in outer_list:
        path_idxs = []
        for dend in path:
            dendname = dend.name()
            idx = get_coord_index(cell_coordinates, dendname, None)
            path_idxs.extend(idx)
        outer_idxs.append(path_idxs)
    idxs.append(outer_idxs)

# Rebuild dists
dists = []
for outer_list in idxs:
    outer_dists = []
    for path_idxs in outer_list:
        path_dists = cell_coordinates[path_idxs, 5].astype(float)
        outer_dists.append(path_dists)
    dists.append(outer_dists)


# Build Vpeaks (use 0 for first simulation, not idx)
sim_idx = 0

Vpeaks = [[Vpeaks_3D[sim_idx][np.array(path_idxs)] for path_idxs in outer_list] for outer_list in idxs]

titles = [path[-1].name() for outer_list in extracted for path in outer_list]

fig = plot_v_mpl(dists[0], Vpeaks[0], titles=titles, colors='grey', xrange=[-2, 225], yrange=[-85, -20], height=400, width=700,  points_size=4)

fig

if save:
    fig.savefig(f"{sim_image_path}/voltage_vs_distance_{sim_idx}.svg", format='svg', dpi=300, bbox_inches='tight')
```

The notebook also repeats this workflow for a selected timing offset:

```python
delta_t = +10

tgaba = next(iter(sim_data.values()))['metadata']['stim_time']

sim_idx = np.argmin(np.abs(np.array(timing_range) - (tgaba + delta_t)))

Vpeaks = [[Vpeaks_3D[sim_idx][np.array(path_idxs)] for path_idxs in outer_list] for outer_list in idxs]

titles = [path[-1].name() for outer_list in extracted for path in outer_list]

fig = plot_v_mpl(dists[0], Vpeaks[0], titles=titles, colors='grey', xrange=[-2, 225], yrange=[-85, -20], height=400, width=700,  points_size=4)

fig

if save:
    fig.savefig(f"{sim_image_path}/voltage_vs_distance_{sim_idx}.svg", format='svg', dpi=300, bbox_inches='tight')
```

Output:

```text
voltage_vs_distance_<sim_idx>.svg
```

<div align="center">
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <p><strong>GLUT only</strong></p>
    <img src="./example%20images/example%20analysis/voltage_vs_distance_0.svg" alt="Voltage versus distance, timing index 0">
  </div>
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <p><strong>GLUT + GABA &Delta;t = t<sub>GLUT</sub> - t<sub>GABA</sub> = -10 ms</strong></p>
    <img src="./example%20images/example%20analysis/voltage_vs_distance_6.svg" alt="Voltage versus distance, timing index 6">
  </div>
</div>

<h2 align="center">Mechanism Current Density</h2>

This block summarizes available ionic mechanism currents, extracts each mechanism at soma and dendrite, converts current density from mA/cm^2 to uA/cm^2, plots each mechanism, and saves soma and dendrite current-density figures.

```python
# units mA/cm^2
summarise_sim_entry(sim_data, 'i_mechs_3D')

mechs2display = get_mechs2display(sim_data)
name_map = {
    'kaf': 'K<sub>v</sub>4',
    'kas': 'K<sub>v</sub>1',
    'kir': 'K<sub>ir</sub>',
    'kcnq': 'K<sub>v</sub>7'
}


idx1 = get_coord_index(cell_coordinates, 'soma[0]', 0.5)
idx2 = get_coord_index(cell_coordinates, record_dendrite, record_location)

for mech_name in mechs2display:
    I_all = extract_mech_currents(sim_data, mech_name, [idx1, idx2])
    soma_traces = I_all[idx1]
    dend_traces = I_all[idx2]

    # convert to uA/cm^2
    soma_traces = [trace * 1e3 for trace in soma_traces]
    dend_traces = [trace * 1e3 for trace in dend_traces]

    # compute yrange across both soma & dend
    yrange_soma = get_y_range(soma_traces)
    yrange_dend = get_y_range(dend_traces)

    span_soma = yrange_soma[1] - yrange_soma[0]
    y_err_bar_soma = span_soma * 0.10
    y_err_bar_soma = roundup(y_err_bar_soma)

    span_dend = yrange_dend[1] - yrange_dend[0]
    y_err_bar_dend = span_dend * 0.10
    y_err_bar_dend = roundup(y_err_bar_dend)

    mech_label = name_map.get(mech_name, f'K<sub>{mech_name}</sub>')

    soma_fig = plot3_dt(soma_traces, yaxis='', yrange=yrange_soma, stim_time=150, sim_time=sim_time, height=height2, width=width2,
                        title=f'somatic {mech_label} density (uA/cm^2)', y_err_bar_units='uA/cm^2',
                        black_trace=0, gray_trace=1, black_shift=100, y_err_bar=y_err_bar_soma,
                        x_err_bar=50, baseline=50, dt=dt, ds=ds_plot)

    dend_fig = plot3_dt(dend_traces, yaxis='', yrange=yrange_dend, stim_time=150, sim_time=sim_time, height=height2, width=width2,
                        title=f'dendritic {mech_label} density (uA/cm^2)', y_err_bar_units='uA/cm^2',
                        black_trace=0, gray_trace=1, black_shift=100, y_err_bar=y_err_bar_dend,
                        x_err_bar=50, baseline=50, dt=dt, ds=ds_plot)

    soma_fig.show(config={"toImageButtonOptions": {"format": "svg"}})
    dend_fig.show(config={"toImageButtonOptions": {"format": "svg"}})

    if save:
        soma_fig.write_image(f"{sim_image_path}/current_density_soma_{mech_name}.svg", format='svg', scale=3)
        dend_fig.write_image(f"{sim_image_path}/current_density_dend_{mech_name}.svg", format='svg', scale=3)
```

Outputs:

```text
current_density_soma_<mech_name>.svg
current_density_dend_<mech_name>.svg
```

<div align="center">
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/current_density_soma_kaf.svg" alt="Somatic Kaf current density">
  </div>
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/current_density_dend_kaf.svg" alt="Dendritic Kaf current density">
  </div>
</div>

<div align="center">
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/current_density_soma_kas.svg" alt="Somatic Kas current density">
  </div>
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/current_density_dend_kas.svg" alt="Dendritic Kas current density">
  </div>
</div>

<div align="center">
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/current_density_soma_kir.svg" alt="Somatic Kir current density">
  </div>
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/current_density_dend_kir.svg" alt="Dendritic Kir current density">
  </div>
</div>

<div align="center">
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/current_density_soma_kcnq.svg" alt="Somatic KCNQ current density">
  </div>
  <div style="display:inline-block; width:48%; text-align:center; vertical-align:top;">
    <img src="./example%20images/example%20analysis/current_density_dend_kcnq.svg" alt="Dendritic KCNQ current density">
  </div>
</div>

<h2 align="center">Expected Outputs</h2>

For `sim2256`, saved outputs are written to `analysis/ispn/sim2256`. Typical files include the morphology figures, voltage traces, stimulation timing, heat maps, voltage-distance plots, and mechanism current-density plots.

```text
morphology.svg
morphology2.svg
Vsoma.svg
Vdend.svg
stimulation_paradigm.svg
V2D_<idx>.svg
V2D_path_<idx>.svg
voltage_vs_distance_<sim_idx>.svg
current_density_soma_<mech_name>.svg
current_density_dend_<mech_name>.svg
```
