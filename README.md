<h1 align="center">Repository for <code>NEURON + Python</code> Model of Striatal Projection Neurons</h1>

This repository contains a `NEURON + Python` model of striatal projection neurons (or SPNs) designed to simulate the interaction between GABAergic and glutamatergic synaptic inputs. <a href="https://doi.org/10.5281/zenodo.20705696"><img src="example%20images/doi_zenodo.svg" alt="DOI: 10.5281/zenodo.20705696" height="20"></a>

It also provides the `Python` code used to analyse the resulting `NEURON + Python` output to generate all graph and figure outputs.

The `NEURON + Python` model is built on top of the `striatal_SPN_lib` repository created by Lindroos and Kotaleski, 2020:

Lindroos R, Kotaleski JH. Predicting complex spikes in striatal projection neurons of the direct pathway following neuromodulation by acetylcholine and dopamine. Eur J Neurosci. 2020. <a href="https://doi.org/10.1111/ejn.14891"><img src="example%20images/doi_lindroos_kotaleski.svg" alt="DOI: 10.1111/ejn.14891"></a>

The original model can be found here [modelDB](https://modeldb.science/266775) or [`GitHub`](https://github.com/ModelDBRepository/266775)

## Manuscript

The code presented here was used to generate model simulations in the following manuscript:

**Cholinergic interneuron control of intrastriatal GABAergic circuits targeting spiny projection neurons is disrupted in Parkinson’s disease models**

Belal, M. <sup>1, 2</sup> <a href="https://orcid.org/0000-0001-8778-0617"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Perez-Rosello, T. <sup>1, 2</sup> <a href="https://orcid.org/0009-0007-8952-2276"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Guven E. B. <sup>3</sup> <a href="https://orcid.org/0000-0002-9634-0485"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Kocaturk, S. <sup>4</sup> <a href="https://orcid.org/0000-0002-0248-4825"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Xie, Z. <sup>1, 2</sup> <a href="https://orcid.org/0000-0002-8348-4455"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Ilijic, E. <sup>1, 2</sup> <a href="https://orcid.org/0000-0002-7112-4401"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Tkatch, T. <sup>1, 2</sup> <a href="https://orcid.org/0000-0001-6626-7435"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Li, J. <sup>5</sup>,
Dauer, W. <sup>6</sup> <a href="https://orcid.org/0000-0003-1775-7504"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Assous, M. <sup>4</sup> <a href="https://orcid.org/0000-0001-6039-816X"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Tepper, J. M. <sup>3</sup> <a href="https://orcid.org/0000-0002-8643-4082"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Clarke, V. R. J. <sup>1, 2</sup> <a href="https://orcid.org/0000-0002-6154-6555"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>,
Surmeier, D. J. <sup>1, 2</sup> <a href="https://orcid.org/0000-0002-6376-5225"><img src="example%20images/orcid_16x16.png" width="16" height="16" alt="ORCID iD"></a>

**Affiliations**

<sup>1</sup> Department of Neuroscience, Feinberg School of Medicine, Northwestern University, Chicago, Illinois 60611, USA  
<sup>2</sup> Aligning Science Across Parkinson's (ASAP) Collaborative Research Network, Chevy Chase, MD 20815  
<sup>3</sup> Molecular and Behavioral Neuroscience, Rutgers University, Newark, NJ USA  
<sup>4</sup> School of Biosciences, Cardiff University, Cardiff, UK  
<sup>5</sup> Department of Internal Medicine, University of Michigan Medical School, Ann Arbor, MI, USA  
<sup>6</sup> Peter O'Donnell Jr. Brain Institute, Departments of Neurology and Neuroscience, University of Texas Southwestern Medical Center, Dallas, TX, USA.

## Funding

This research was funded by grants to DJS from:

Aligning Science Across Parkinson’s [ASAP020551] through the Michael J. Fox Foundation for Parkinson’s Research (MJFF); Aligning Science
Across Parkinson’s Collaborative Research Network, Chevy Chase, MD, 20815; https://parkinsonsroadmap.org.  

Freedom Together Foundation [MR-2021-2960], 875 Third Avenue, 29th Floor, New York, NY 10022; https://www.freedomtogether.org.    

National Institute of Neurological Disorders and Stroke [R37 NS034696], P.O. Box 5801. Bethesda, MD 20824; https://www.ninds.nih.gov.



## Table of Contents
- [Manuscript](#manuscript)
- [Funding](#funding)
- [Initial Set Up](#initial-set-up)
- [Running the Models](#running-the-models)
  - [Getting Started](#getting-started)
  - [Simulations](#running-simulations-in-jupyter-notebook)
- [Repository Structure](#repository-structure)
- [Figure mapping](#figure-mapping)
- [Downsampling](#downsampling)
- [Figure simulations](./simulations.md)
- [Data Analysis](#data-analysis)
  - [Setting up](#setting-up)
  - [Using `Python` to analyse a simulation](#using-python-to-analyse-a-simulation)
- [`Anaconda` vs `Miniconda`](#anaconda-vs-miniconda)
- [Virtual Environments](#virtual-environments)
- [Updating `YAML` and `Conda` environment](#updating-yaml-and-conda-environment)
- [Exporting a working environment](#exporting-a-working-environment)
- [GitHub](#github)
- [References](#references)
- [Contact](#contact)


## Initial Set Up

### Prerequisites
- [`Conda`](https://docs.conda.io/projects/conda/en/stable)
- `NEURON` (tested on versions 8.1 and 8.2). Install from the [official website](https://www.neuronsimulator.org). Older `NEURON` releases, including 8.2.x installers, are available on [GitHub](https://github.com/neuronsimulator/nrn/releases).

  Reference: Hines ML, Carnevale NT. The `NEURON` Simulation Environment. Neural Comput. 1997. <a href="https://doi.org/10.1162/neco.1997.9.6.1179"><img src="example%20images/doi_neuron.svg" alt="DOI: 10.1162/neco.1997.9.6.1179"></a>
- `Python` (tested using version 3.11.11)

### Steps
1. **Install [`Conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)** (`Python` package manager)
   
   `Conda` should include a version of `Python`. 

   `Conda` is simply a package manager and environment management system that is used to install, run and update packages and their dependencies.

   Either `Anaconda` or `Miniconda` will work; see [comparison](#anaconda-vs-miniconda).

2. **Install [`Jupyter Notebook`](https://jupyter.org)**

   The simplest method to install `Jupyter Notebook` is via `Conda` using the command in `terminal`:

   ```bash
   conda install -c conda-forge notebook
   ```

3. **Install [`NEURON`](https://www.neuronsimulator.org)**

  Follow the guide at [`NEURON`](https://www.neuronsimulator.org/en/latest/install/install.html)

## Running the Models

### Quick setup after creating the environment

Use these commands after completing the environment creation steps in [Getting Started](#getting-started).

These commands assume the repository was cloned to the `Documents/Repositories/msNEURON_Belal2026` path shown below. If you cloned it elsewhere, use your own repository path in the `cd` command.

On `macOS` / `Linux`:
   ```bash
   cd ~/Documents/Repositories/msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   python -m ipykernel install --user --name msNEURON_Belal2026 --display-name "Python (msNEURON_Belal2026)"
   jupyter notebook
   ```

On `Windows` using `Command Prompt`, `Anaconda Prompt`, or `Miniconda Prompt`:
   ```bat
   cd %USERPROFILE%\Documents\Repositories\msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   python -m ipykernel install --user --name msNEURON_Belal2026 --display-name "Python (msNEURON_Belal2026)"
   jupyter notebook
   ```

On `Windows` using `PowerShell`:
   ```powershell
   cd $HOME\Documents\Repositories\msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   python -m ipykernel install --user --name msNEURON_Belal2026 --display-name "Python (msNEURON_Belal2026)"
   jupyter notebook
   ```

The following sections explain the initial set up required and instructions to create simulations subsequently used to generate figures.

### Getting Started

1. **Confirm `NEURON` is installed with `Python` support** (see setup instructions)

2. **Open `Terminal`**:
   - On `macOS`: Press `cmd + space` to open spotlight search and type `terminal`.
   - On `Linux`: Search for `terminal` in your applications menu or press `ctrl + alt + T`.
   - On `Windows`: Search for `command prompt` or `PowerShell` in the start menu.

3. **Create a `Conda` environment**

   There is a `YAML` file in the main directory called `environment.yml` for `macOS`/`Linux`. This can be used to create a `Conda` environment called `msNEURON_Belal2026`. This `macOS`/`Linux` environment file does not work on `Windows`. For further information see [Virtual Environments](#virtual-environments).

   Make sure your terminal is in the repository's main directory before running the setup commands.

   Check that the environment installed correctly using the `conda list` command.

   On `macOS` / `Linux`:
   ```bash
   cd ~/Documents/Repositories/msNEURON_Belal2026 
   conda env create -f environment.yml
   conda activate msNEURON_Belal2026
   conda list
   ```

   On `Windows`, the provided `environment.yml` fails because it was exported from `macOS`/`Linux` and contains platform-specific packages. For `Windows` smoke testing, create a clean `Python` 3.11 `Conda` environment manually, then install `NEURON` separately with the `Windows` installer.

   `Command Prompt`, `Anaconda Prompt`, or `Miniconda Prompt`:

   ```bat
   cd %USERPROFILE%\Documents\Repositories\msNEURON_Belal2026
   conda create -n msNEURON_Belal2026 python=3.11 -y
   conda activate msNEURON_Belal2026
   conda install -c conda-forge notebook ipykernel ipywidgets numpy=1.26 scipy pandas matplotlib seaborn plotly scikit-learn numba tqdm pyyaml requests openpyxl -y
   pip install colorednoise hmmlearn kaleido
   ```

   `PowerShell`:

   ```powershell
   cd $HOME\Documents\Repositories\msNEURON_Belal2026
   conda create -n msNEURON_Belal2026 python=3.11 -y
   conda activate msNEURON_Belal2026
   conda install -c conda-forge notebook ipykernel ipywidgets numpy=1.26 scipy pandas matplotlib seaborn plotly scikit-learn numba tqdm pyyaml requests openpyxl -y
   pip install colorednoise hmmlearn kaleido
   ```
  
   `NEURON` cannot be installed on `Windows` using the `macOS`/`Linux` `pip install neuron` workflow. Instead, download and run a `Windows` installer from [GitHub releases](https://github.com/neuronsimulator/nrn/releases). For this repository, choose a `NEURON` 8.2.x installer compatible with `Python` 3.11 if available.

   After installing `NEURON`, confirm the active `Conda` environment can import it:

   ```bash
   conda activate msNEURON_Belal2026
   python -c "from neuron import h; print(h.nrnversion())"
   ```

   Because the `conda env create -f environment.yml` command fails on `Windows`, the `conda list` command will show the base `Conda` environment rather than the project environment until the manual environment has been created and activated. This repository does not currently include `environment_pc.yml`, so use the manual `Windows` setup commands above unless you have generated or received a tested `Windows` environment file separately.
     
4. **Compile mechanisms**:

   In order to compile the mechanism mod files for the version of `NEURON` used, compile AFTER activating the environment `msNEURON_Belal2026`.

   If you have opened a new `terminal` window since creating the `Conda` environment, first navigate back to the repository and activate the environment again. `Conda` activation applies only to the current `terminal` session.

   ```bash
   cd ~/Documents/Repositories/msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   ```

   On `Windows` using `Command Prompt`, `Anaconda Prompt`, or `Miniconda Prompt`:
   ```bat
   cd %USERPROFILE%\Documents\Repositories\msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   ```

   On `Windows` using `PowerShell`:
   ```powershell
   cd $HOME\Documents\Repositories\msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   ```

   Navigate to directory containing `NEURON` mechanisms.

   For instance, if `msNEURON_Belal2026` is in the `Documents` folder on `macOS`, mechanisms are located in `~/Documents/Repositories/msNEURON_Belal2026/mechanisms/single3`.

   On `macOS`, `Linux`, and `Windows`, compile mechanisms with `nrnivmodl`.

   On `macOS` / `Linux`:
   ```bash
   cd mechanisms/single3
   nrnivmodl
   ```

   On `Windows` using `Command Prompt`, `Anaconda Prompt`, or `Miniconda Prompt`:
   ```bat
   cd %USERPROFILE%\Documents\Repositories\msNEURON_Belal2026\mechanisms\single3
   nrnivmodl
   ```

   On `Windows` using `PowerShell`:
   ```powershell
   cd $HOME\Documents\Repositories\msNEURON_Belal2026\mechanisms\single3
   nrnivmodl
   ```
     
5. **Quit `Terminal` / `Command Prompt` / `PowerShell`**

   ```bash
   exit
   ```
   
### Running simulations in `Jupyter Notebook`

  The following steps 1-4 must be performed every time a new `Jupyter Notebook` session is started.

1. **Open `Terminal`**:
   - On `macOS`: Press `cmd + space` to open spotlight search and type `terminal`.
   - On `Linux`: Search for `terminal` in your applications menu or press `ctrl + alt + T`.
   - On `Windows`: Search for `command prompt`, `PowerShell` or the appropriate `Miniconda`/`Anaconda` prompt in the start menu.

2. **Activate `Conda` environment `msNEURON_Belal2026`**

   Navigate back to the main directory

   On `macOS` / `Linux`:
   ```bash
   cd ~/Documents/Repositories/msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   ```

   On `Windows` using `Command Prompt`, `Anaconda Prompt`, or `Miniconda Prompt`:
   ```bat
   cd %USERPROFILE%\Documents\Repositories\msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   ```

   On `Windows` using `PowerShell`:
   ```powershell
   cd $HOME\Documents\Repositories\msNEURON_Belal2026
   conda activate msNEURON_Belal2026
   ```
3. **Run `Jupyter Notebook`**

   Add `msNEURON_Belal2026` environment then open `Jupyter Notebook`
   ```bash
   python -m ipykernel install --user --name msNEURON_Belal2026 --display-name "Python (msNEURON_Belal2026)"
   jupyter notebook
   ```

4. **Run a simulation**

   `Jupyter Notebook` should now be open in the default browser.

   To run simulations, open one of the notebooks in the `simulations/` directory. To run analysis, open one of the notebooks in the `analysis notebooks/` directory.
   
   Ensure kernel is set to `Python` 3 (ipykernel).

   From the Kernel dropdown menu, choose `Restart & Run All` (if running again then it's good practice to run `Restart and Clear Output` first).

   Code should run and generate raw data used to generate figures.

   If option `save = True` in the Notebook then the raw figures and pickled data are stored in a subdirectory within the main one.

## Repository Structure

This repository contains the model 3 implementation used for the manuscript. Model-specific files include:

`MSN_builder3.py`  
`params_dMSN3.json`  
`params_iMSN3.json`

```bash

msNEURON_Belal2026/
├── README.md
├── sim_descriptions.json
├── figure map.json
├── settings.py
├── master_functions.py
├── analysis_functions.py
├── MSN_builder3.py
├── params_dMSN3.json
├── params_iMSN3.json
├── environment.yml
├── simulations.md
├── LICENSE
├── example analysis README.md
├── update_pip_packages.py
├── .gitignore
│
├── simulations/
│   ├── sim225xx downsample.ipynb          Fig 4b-d, S4b-d
│   ├── sim227xx downsample.ipynb          Fig 4e-f, 6b-d
│   ├── sim241x downsample.ipynb           Fig S3a-b
│   ├── sim251x downsample.ipynb           Fig S3c-d
│   ├── sim261xx downsample.ipynb          Fig S2a-b
│   ├── sim271xx downsample.ipynb          Fig S2c-d
│   ├── sim281xx downsample.ipynb          Fig 5b-d, S4b,e-f
│   ├── sim291xx downsample.ipynb          Fig 5e-f, 6e-f
│   ├── passive tune downsample.ipynb      Fig S1 a,d-g,j-l
│   └── passive tune IR downsample.ipynb   Fig S1 b,c,h,i
│
├── Morphologies/
│   ├── WT-dMSN_P270-20_1.02_SGA1-m24.swc
│   └── WT-iMSN_P270-09_1.01_SGA2-m1.swc
│
├── mechanisms/
│   └── single3/
│       ├── naf.mod
│       ├── nap.mod
│       ├── kaf.mod
│       ├── kas.mod
│       ├── kdr.mod
│       ├── kir.mod
│       ├── kcnq.mod
│       ├── sk.mod
│       ├── bk.mod
│       ├── cal12.mod
│       ├── cal13.mod
│       ├── car.mod
│       ├── can.mod
│       ├── caq.mod
│       ├── cav32.mod
│       ├── cav33.mod
│       ├── cadyn.mod
│       ├── caldyn.mod
│       ├── glutsynapse.mod
│       ├── gabasynapse.mod
│       ├── tonicgaba1.mod
│       ├── tonicgaba2.mod
│       └── vecevent.mod
│
├── downsample/                            generated simulation output; ignored by git
│   ├── dspn/
│   │   └── model3/
│   │       └── physiological/
│   │           ├── simulations/
│   │           └── images/
│   └── ispn/
│       └── model3/
│           └── physiological/
│               ├── simulations/
│               └── images/
│
├── analysis/                              generated analysis output; ignored by git
│   ├── dspn/
│   └── ispn/
│
├── analysis notebooks/
│   ├── sim2256 analysis.ipynb
│   ├── sim2257 analysis.ipynb
│   ├── sim22512 analysis.ipynb
│   ├── sim22513 analysis.ipynb
│   ├── sim2276 analysis.ipynb
│   ├── sim2277 analysis.ipynb
│   ├── sim22712 analysis.ipynb
│   ├── sim22713 analysis.ipynb
│   ├── sim2410 iSPN analysis.ipynb
│   ├── sim2411 iSPN analysis.ipynb
│   ├── sim2418 iSPN analysis.ipynb
│   ├── sim2419 iSPN analysis.ipynb
│   ├── sim2510 iSPN analysis.ipynb
│   ├── sim2511 iSPN analysis.ipynb
│   ├── sim2518 iSPN analysis.ipynb
│   ├── sim2519 iSPN analysis.ipynb
│   ├── sim2610 iSPN analysis.ipynb
│   ├── sim2611 iSPN analysis.ipynb
│   ├── sim2612 iSPN analysis.ipynb
│   ├── sim2613 iSPN analysis.ipynb
│   ├── sim2616 iSPN analysis.ipynb
│   ├── sim2617 iSPN analysis.ipynb
│   ├── sim2710 iSPN analysis.ipynb
│   ├── sim2711 iSPN analysis.ipynb
│   ├── sim2712 iSPN analysis.ipynb
│   ├── sim2713 iSPN analysis.ipynb
│   ├── sim2716 iSPN analysis.ipynb
│   ├── sim2717 iSPN analysis.ipynb
│   ├── sim2811 dSPN analysis.ipynb
│   ├── sim2811 iSPN analysis.ipynb
│   ├── sim2813 dSPN analysis.ipynb
│   ├── sim2819 iSPN analysis.ipynb
│   ├── sim28111 dSPN analysis.ipynb
│   ├── sim28111 iSPN analysis.ipynb
│   ├── sim28113 dSPN analysis.ipynb
│   ├── sim28119 iSPN analysis.ipynb
│   ├── sim2911 dSPN analysis.ipynb
│   ├── sim2911 iSPN analysis.ipynb
│   ├── sim2913 dSPN analysis.ipynb
│   ├── sim2919 iSPN analysis.ipynb
│   ├── sim29111 dSPN analysis.ipynb
│   ├── sim29111 iSPN analysis.ipynb
│   ├── sim29113 dSPN analysis.ipynb
│   ├── sim29119 iSPN analysis.ipynb
│   ├── passive tune analysis.ipynb
│   ├── passive tune analysis Matplotlib.ipynb
│   ├── passive tune IR analysis Matplotlib.ipynb
│   └── example analysis.ipynb
│
├── example images/
│   └── example analysis/
│
└── animations/
```

## Figure Mapping

The file `figure map.json` links each manuscript figure panel to the simulation notebook, analysis notebook, and simulation identifiers used to generate it.

All simulations are run from the notebooks listed below. Each notebook generates a family of simulation output files. The entries below list the simulation IDs used for each figure.

- **Figure 4**: iSPN, subthreshold clustered glutamatergic input with fast synaptic or slow extrasynaptic GABAergic input, under control conditions and M<sub>1</sub> receptor activation.  
  Simulation notebooks: `sim225xx downsample.ipynb`, `sim227xx downsample.ipynb`  
  Simulations: `sim2256`, `sim22512`, `sim2257`, `sim22513`, `sim2276`, `sim22712`, `sim2277`, `sim22713`

- **Figure 5**: iSPN, suprathreshold clustered glutamatergic input with fast synaptic or slow extrasynaptic GABAergic input, under control conditions and M<sub>1</sub> receptor activation.  
  Simulation notebooks: `sim281xx downsample.ipynb`, `sim291xx downsample.ipynb`  
  Simulations: `sim2811`, `sim2819`, `sim28111`, `sim28119`, `sim2911`, `sim2919`, `sim29111`, `sim29119`

- **Figure 6**: dSPN, slow extrasynaptic GABAergic interaction with subthreshold and suprathreshold clustered glutamatergic input, under control conditions and M<sub>1</sub> receptor activation.  
  Simulation notebooks: `sim227xx downsample.ipynb`, `sim291xx downsample.ipynb`  
  Simulations: `sim2276`, `sim22712`, `sim2277`, `sim22713`, `sim2911`, `sim2913`, `sim29111`, `sim29113`

- **Figure S1**: Passive membrane properties, morphology, input resistance, conductance density, rheobase, and spike output for dSPN and iSPN models under control conditions and K<sub>v</sub>7 blockade.  
  Simulation notebooks: `passive tune downsample.ipynb`, `passive tune IR downsample.ipynb`  
  Simulations: `sim1001`, `sim1002`, `sim1003`, `sim1004`, `sim1005`, `sim1006`

- **Figure S2**: iSPN, varying the number of glutamatergic synapses with fixed fast synaptic or slow extrasynaptic GABAergic input, under control conditions and M<sub>1</sub> receptor activation.  
  Simulation notebooks: `sim261xx downsample.ipynb`, `sim271xx downsample.ipynb`  
  Simulations: `sim2610`, `sim2611`, `sim2612`, `sim2613`, `sim2616`, `sim2617`, `sim2710`, `sim2711`, `sim2712`, `sim2713`, `sim2716`, `sim2717`

- **Figure S3**: iSPN, varying the number of fast synaptic or slow extrasynaptic GABAergic inputs with fixed suprathreshold clustered glutamatergic input, under control conditions and M<sub>1</sub> receptor activation.  
  Simulation notebooks: `sim241x downsample.ipynb`, `sim251x downsample.ipynb`  
  Simulations: `sim2410`, `sim2411`, `sim2418`, `sim2419`, `sim2510`, `sim2511`, `sim2518`, `sim2519`

- **Figure S4**: dSPN, fast synaptic GABAergic interaction with subthreshold and suprathreshold clustered glutamatergic input, under control conditions and M<sub>1</sub> receptor activation.  
  Simulation notebooks: `sim225xx downsample.ipynb`, `sim281xx downsample.ipynb`  
  Simulations: `sim2256`, `sim22512`, `sim2257`, `sim22513`, `sim2811`, `sim2813`, `sim28111`, `sim28113`

---

## Downsampling

Simulations require a small time step (`dt`) for accurate numerical solutions. Saving full-resolution output can produce very large files, so the simulation data are downsampled after each simulation run.

These downsampled outputs are used for figure generation and analysis workflows in this repository.

## Data Analysis

The final analysis and manuscript figures are generated using `Python`.

The analysis code is stored in the `analysis notebooks` directory.

### Setting up

Use the `msNEURON_Belal2026` `Conda` environment described above, then launch `Jupyter Notebook` with the `Python (msNEURON_Belal2026)` kernel.

### Using `Python` to analyse a simulation

First, run the relevant simulation notebook with `save = True`. Then open and run the corresponding `Python` analysis notebook. Each simulation has a unique identifier, for example `sim2257` or `sim22713`. 

When a simulation notebook is run with `save = True`, the raw output is written to a simulation-specific folder under `downsample/<cell_type>/model<model>/physiological/simulations/<sim>`, for example `downsample/dspn/model3/physiological/simulations/sim2257`.

Each analysis notebook defines the variables `sim`, `cell_type`, `model`, `downsample`, and `external` near the start. These variables specify where the notebook looks for simulation output. 

The analysis notebook loads the pickled simulation data using `load_data_dicts(...)`. 

If the simulation data are stored on an external drive, they are copied once into a local cache at `~/Documents/simcache/<cell_type>/<sim>`. Subsequent analysis runs use this cached copy instead of copying the external-drive data again. 

If the source simulation folder is not found, the loader checks the cache before raising an error.

Analysis outputs are not saved back into the original simulation folder. Figures, animations, spreadsheets, and other analysis-generated files are written inside the repository under `analysis/<cell_type>/<sim>`, for example `~/Documents/Repositories/msNEURON_Belal2026/analysis/ispn/sim2257`. 

Images generated directly by the simulation workflow may also be found under the corresponding model image directory, for example `downsample/dspn/model3/physiological/images`.

### Example analysis notebook

The notebook [`example analysis.ipynb`](analysis%20notebooks/example%20analysis.ipynb) demonstrates the analysis workflow and saves example `SVG` outputs in the `example images/example analysis` directory.

![Example morphology output](example%20images/example%20analysis/morphology2.svg)

Morphology of reconstructed iSPN illustrating clustered glutamatergic synaptic activation with 11 glutamatergic inputs (red) and the location of fast GABAergic synaptic activation (blue). This subthreshold condition produces no upstates with or without GABAergic input in control, and no upstates for glutamatergic input alone after 50% reduction of K<sub>v</sub>7, K<sub>ir</sub>, and K<sub>v</sub>4 conductances.

![Example voltage output condition 0](example%20images/example%20analysis/V2D_0.svg)

Heat map of peak voltage for the `GLUT`-only response at 150 ms across the reconstructed dendritic morphology. This condition is subthreshold for dendritic spike generation.

![Example voltage output condition 6](example%20images/example%20analysis/V2D_6.svg)

Heat map of peak voltage for the `GABA` + `GLUT` timing condition where `GLUT` is activated at 150 ms and `GABA` is activated at 160 ms (Δt = t<sub>GLUT</sub> - t<sub>GABA</sub> = -10 ms). This condition is suprathreshold for dendritic spike generation.

### Movie simulations

Movie versions of selected simulations are provided in [Figure Simulations](./simulations.md). These animations illustrate how dendritic or somatic voltage changes as `GABA` timing is varied relative to clustered `GLUT` input, comparing control conditions with M<sub>1</sub> receptor activation for the simulations used in Figures 4, 5, and 6.

## `Anaconda` vs `Miniconda`

`Anaconda` and `Miniconda` are both popular distributions for `Python` programming in data science. They include the `Conda` package manager and aim to simplify package management and deployment.

**`Anaconda`** is a full-featured distribution that includes:

- `Python` language
- `Conda` package manager
- Over 1,500 pre-installed scientific packages
- Tools like `Jupyter Notebook` and other scientific computing applications.

`Anaconda` provides an out-of-the-box setup for data science and scientific computing.

**`Miniconda`** offers a minimalistic approach:

- `Python` language
- `Conda` package manager
- No pre-installed packages

`Miniconda` provides a lightweight base to start with but packages must be installed if needed.

**Advantages of `Anaconda`**:

- Quick, easy setup with a comprehensive suite of scientific packages and tools.
- Wide array of data science tools readily available within a single application.

**Advantages of `Miniconda`**:

- Lightweight, minimal base installation.
- Control over which packages are installed.
- Requires limited disk space or bandwidth.
- Clean environment that only includes packages required.

**Installation**

- For `Anaconda`, download the installer from [`Anaconda`](https://www.anaconda.com/products/individual#Downloads).
- For `Miniconda`, visit [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html).

Follow the installation instructions provided on the respective download pages.

**Additional Resources**

- [`Anaconda`](https://docs.anaconda.com/)
- [`Miniconda`](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- [`Conda`](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)


## Virtual Environments

The `environment.yml` file defines the `msNEURON_Belal2026` `Conda` environment used to run the model and analysis notebooks. It is configured for `Python` 3.11.11.

The file intentionally does not include a machine-specific `prefix`, so `Conda` creates the environment in the default location for your installation and uses the `name: msNEURON_Belal2026` entry for activation.
  

## Updating `YAML` and `Conda` environment

To update an existing working environment, activate it, update packages, test the notebooks or scripts you rely on, then re-export the environment file.

On `macOS` / `Linux`:

```bash
cd ~/Documents/Repositories/msNEURON_Belal2026
conda activate msNEURON_Belal2026
conda update --all
python update_pip_packages.py
python -c "import neuron, numpy, scipy, seaborn; print('Environment works!')"
conda env export --name msNEURON_Belal2026 --no-builds --file updated_environment.yml
```

Review `updated_environment.yml` before replacing `environment.yml`.

## Exporting a working environment

After the environment is working, it can be exported for future use.

On `macOS` / `Linux`:

```bash
conda activate msNEURON_Belal2026
conda env export --no-builds > environment.yml
pip freeze > requirements.txt
```

On `Windows`, you can export separate PC-specific files from a working manually created environment:

```bat
conda activate msNEURON_Belal2026
conda env export --no-builds > environment_pc.yml
pip freeze > requirements_pc.txt
```

The `--no-builds` option keeps package versions but removes platform-specific build strings, such as `bzip2=1.0.8=h80987f9_6`. These build strings can prevent an environment exported on one operating system from installing on another, even when the same package version is available for that platform.

However, `--no-builds` does not guarantee that a `Conda` environment file is fully cross-platform. Some packages are still specific to `macOS`, `Linux`, or `Windows`. For this reason, `Windows` users may need a separate `environment_pc.yml`. This repository does not currently ship that file.

To recreate the `Conda` environment from the `macOS` / `Linux` `YAML` file:

```bash
conda env create -f environment.yml
conda activate msNEURON_Belal2026
```

On `Windows`, run the following only if you have already generated or received a tested `Windows`-specific `environment_pc.yml` file:

```bat
conda env create -f environment_pc.yml
conda activate msNEURON_Belal2026
```

If the `pip:` section in a `Conda` environment file causes problems on another operating system, create the `Conda` environment first and then install packages separately:

```bash
conda create -n msNEURON_Belal2026 python=3.11 -y
conda activate msNEURON_Belal2026
conda install -c conda-forge notebook ipykernel ipywidgets numpy=1.26 scipy pandas matplotlib seaborn plotly scikit-learn numba tqdm pyyaml requests openpyxl -y
pip install colorednoise hmmlearn kaleido
```

On `Windows`, install `NEURON` separately using the `Windows` installer.


## `GitHub`

For beginners, the [`GitHub` Desktop GUI](https://desktop.github.com/) is recommended. 

Instructions for cloning a repository using `GitHub` Desktop can be found [here](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop).

## References

Day M, Belal M, Surmeier WC, Melendez A, Wokosin D, Tkatch T, et al. GABAergic regulation of striatal spiny projection neurons depends upon their activity state. PLoS Biol. 2024;22: e3002483. <a href="https://doi.org/10.1371/journal.pbio.3002483"><img src="example%20images/doi_day_belal.svg" alt="DOI: 10.1371/journal.pbio.3002483"></a>

Du K, Wu Y-W, Lindroos R, Liu Y, Rózsa B, Katona G, et al. Cell-type–specific inhibition of the dendritic plateau potential in striatal spiny projection neurons. Proceedings of the National Academy of Sciences. 2017;114: E7612–E7621. <a href="https://doi.org/10.1073/pnas.1704893114"><img src="example%20images/doi_du_lindroos.svg" alt="DOI: 10.1073/pnas.1704893114"></a>

Hines ML, Carnevale NT. The `NEURON` Simulation Environment. Neural Comput. 1997;9: 1179–1209. <a href="https://doi.org/10.1162/neco.1997.9.6.1179"><img src="example%20images/doi_neuron.svg" alt="DOI: 10.1162/neco.1997.9.6.1179"></a>

Lindroos R, Dorst MC, Du K, Filipović M, Keller D, Ketzef M, et al. Basal Ganglia Neuromodulation Over Multiple Temporal and Structural Scales-Simulations of Direct Pathway MSNs Investigate the Fast Onset of Dopaminergic Effects and Predict the Role of K<sub>v</sub>4.2. Frontiers in neural circuits. 2018;12: 3. <a href="https://doi.org/10.3389/fncir.2018.00003"><img src="example%20images/doi_lindroos_dorst.svg" alt="DOI: 10.3389/fncir.2018.00003"></a>

Lindroos R, Kotaleski JH. Predicting complex spikes in striatal projection neurons of the direct pathway following neuromodulation by acetylcholine and dopamine. Eur J Neurosci. 2020. <a href="https://doi.org/10.1111/ejn.14891"><img src="example%20images/doi_lindroos_kotaleski.svg" alt="DOI: 10.1111/ejn.14891"></a>



## Contact

This repository adapts the publicly available Lindroos and Kotaleski / ModelDB codebase, with adaptations by Vernon Clarke.

The provided code was executed on a `MacBook M2 Pro 32GB` and a `Mac mini M4 Pro 64 GB`. I have tried to ensure that the code works on other operating systems but it's inevitable that some errors and bugs exist. 

If any bug fixes are necessary (most likely related to providing help on other operating systems), an update will be provided on the parent [`GitHub`](https://github.com/vernonclarke/msNEURON_Belal2026).

For queries related to this repository, please [open an issue](https://github.com/vernonclarke/msNEURON_Belal2026/issues) or [email](mailto:WOPR2@proton.me) directly.
