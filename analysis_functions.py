'''
analysis functions
'''

import os
import re
import sys
import time
import copy
import shutil
import pickle
import psutil
import subprocess
import datetime

import io
import xml.etree.ElementTree as ET

from collections import OrderedDict

import numpy as np
import pandas as pd

from scipy.interpolate import (
    UnivariateSpline,
    splprep,
    splev,
    interp1d
)

from scipy.interpolate import PchipInterpolator

from neuron import h
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ipywidgets as widgets
from IPython.display import HTML, clear_output, display

from master_functions import palette_cols, extract_mechs_dict2array, round_up_to_sig


################### save / load ###################
def save_df(df, filename, sim_image_path=None, index=False):
    save_dir = sim_image_path if sim_image_path is not None else '.'
    os.makedirs(save_dir, exist_ok=True)
    
    filepath = os.path.join(save_dir, filename)
    
    if filename.endswith('.csv'):
        df.to_csv(filepath, index=index)
    elif filename.endswith('.xlsx'):
        try:
            df.to_excel(filepath, index=index, engine='openpyxl')
        except ModuleNotFoundError:
            print("Warning: openpyxl not installed. Saving as CSV instead.")
            csv_filepath = filepath.replace('.xlsx', '.csv')
            df.to_csv(csv_filepath, index=index)
            filepath = csv_filepath
    else:
        raise ValueError(f"Unsupported file format. Use .csv or .xlsx, got: {filename}")
    
    print(f"Saved: {filepath}")
    return filepath

def load_df(filename, sim_image_path=None):

    # file directory
    load_dir = sim_image_path if sim_image_path is not None else '.'
    
    # full file path
    filepath = os.path.join(load_dir, filename)
    
    # check file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # load based on file extension
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file format. Use .csv or .xlsx, got: {filename}")
    
    print(f"Loaded: {filepath}")
    return df

################### memory usage ###################

def report_memory(verbose=False):
    """Report memory usage for current notebook and system."""
    process = psutil.Process(os.getpid())
    used_mb = process.memory_info().rss / 1e6
    mem = psutil.virtual_memory()
    total_gb = mem.total / 1e9
    used_gb = mem.used / 1e9
    avail_gb = mem.available / 1e9
    print(f"Notebook: {used_mb:.2f} MB | System: {used_gb:.2f} / {total_gb:.2f} GB used ({avail_gb:.2f} GB free)")
    if verbose:
        print(f"PID: {process.pid}")
        print(f"Memory percent (system): {mem.percent:.1f}%")
        print(f"Memory percent (notebook): {process.memory_percent():.2f}%")  

        
############## data loading functions ##############

def is_external_drive(path):
    try:
        path = os.path.abspath(os.path.expanduser(path))
        mount_point = subprocess.run(["df", path], capture_output=True, text=True).stdout.splitlines()[-1].split()[0]
        result = subprocess.run(["diskutil", "info", mount_point], capture_output=True, text=True)
        return "Device Location:           External" in result.stdout
    except Exception:
        return False

def get_total_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

def copy_with_progress(src, dst, verbose=False, update_interval=0.5):
    total_size = get_total_size(src)
    copied = 0
    start = time.time()
    last_update = start
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        dest_dir = os.path.join(dst, rel)
        os.makedirs(dest_dir, exist_ok=True)
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(dest_dir, f)
            size = os.path.getsize(s)
            shutil.copy2(s, d)
            copied += size
            now = time.time()
            if verbose and (now - last_update) >= update_interval:
                elapsed = now - start
                pct = copied / total_size * 100
                speed = copied / elapsed if elapsed > 0 else 0
                remaining = (total_size - copied) / speed if speed > 0 else 0
                sys.stdout.write(
                    f"\rCopying: {copied/1e6:7.1f} / {total_size/1e6:7.1f} MB "
                    f"({pct:5.1f}%)  Elapsed: {elapsed:6.1f}s  Remaining: {remaining:6.1f}s"
                )
                sys.stdout.flush()
                last_update = now
    if verbose:
        sys.stdout.write(f"\nCopy complete in {time.time() - start:.1f} s\n")
        sys.stdout.flush()

def load_data_dicts(wd, sim, cell_type=None, copy_to_cache=True, cache_dir=None, verbose=False):
    sim_dir = os.path.join(wd, sim)
    parent_dir = sim_dir

    # If the original path doesn't exist, try cache first
    if not os.path.exists(parent_dir):
        if verbose:
            print(f"Path not found: {parent_dir}")
        base_cache = os.path.join(os.path.expanduser("~/Documents"), "simcache")
        cache_dir = os.path.join(base_cache, cell_type) if cell_type else base_cache
        candidate = os.path.join(cache_dir, os.path.basename(parent_dir))
        if os.path.exists(candidate):
            parent_dir = candidate
            if verbose:
                print(f"Using cached copy at {parent_dir}\n")
        else:
            raise FileNotFoundError(f"Neither {parent_dir} nor cached copy at {candidate} found.")

    # If external, copy to cache
    if copy_to_cache and is_external_drive(parent_dir):
        if cache_dir is None:
            base_cache = os.path.join(os.path.expanduser("~/Documents"), "simcache")
            cache_dir = os.path.join(base_cache, cell_type) if cell_type else base_cache
        os.makedirs(cache_dir, exist_ok=True)
        target = os.path.join(cache_dir, os.path.basename(parent_dir))
        if not os.path.exists(target):
            if verbose:
                print(f"External drive detected — copying to cache...")
                print(f"From: {parent_dir}")
                print(f"To:   {target}")
            copy_with_progress(parent_dir, target, verbose=verbose)
        else:
            if verbose:
                print(f"Using cached copy at {target}\n")
        parent_dir = target
    elif verbose:
        print(f"Loading directly from {parent_dir}\n")

    combined_data = {}
    sim_folders = sorted(
        [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))],
        key=lambda x: [int(text) if text.isdigit() else text for text in re.split('(\d+)', x)]
    )
    if sim_folders:
        for sim_name in sim_folders:
            sim_path = os.path.join(parent_dir, sim_name)
            pickle_files = [f for f in os.listdir(sim_path) if f.endswith('.pickle')]
            if not pickle_files:
                continue
            combined_data[sim_name] = {}
            start_load = time.time()
            for f in pickle_files:
                key = f.replace('.pickle', '')
                with open(os.path.join(sim_path, f), 'rb') as handle:
                    combined_data[sim_name][key] = pickle.load(handle)
            if verbose:
                print(f"{sim_name}: {len(pickle_files)} files loaded ({time.time() - start_load:.1f} s)")
    else:
        pickle_files = [f for f in os.listdir(parent_dir) if f.endswith('.pickle')]
        if pickle_files:
            start_load = time.time()
            for f in pickle_files:
                key = f.replace('.pickle', '')
                with open(os.path.join(parent_dir, f), 'rb') as handle:
                    combined_data[key] = pickle.load(handle)
            if verbose:
                print(f"{len(pickle_files)} files loaded from parent ({time.time() - start_load:.1f} s)")

    if verbose:
        print(f"\nAll data loaded from {parent_dir}")
    return combined_data


############# cell type characteristics ############

def summarise_cell_coordinates(sim_data):
    coord_vars = set()
    coords_all = {}

    def is_valid_coords(x):
        if x is None:
            return False
        if hasattr(x, 'empty') and x.empty:
            return False
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 0:
                return False
            flat = np.array(x, dtype=object).ravel()
            if all(
                (el is None)
                or (isinstance(el, (list, tuple, np.ndarray)) and len(el) == 0)
                for el in flat
            ):
                return False
            return True
        return False

    for sim_name, sim_vars in sim_data.items():
        for var_name, var_data in sim_vars.items():
            if var_name == 'metadata':
                continue
            
            if not isinstance(var_data, dict):
                continue

            if 'cell_coordinates' in var_data and is_valid_coords(var_data['cell_coordinates']):
                arr = np.array(var_data['cell_coordinates'], dtype=object)
                coord_vars.add(var_name)
                coords_all.setdefault(var_name, []).append(arr)

            for subname, subdata in var_data.items():
                if isinstance(subdata, dict) and 'cell_coordinates' in subdata and is_valid_coords(subdata['cell_coordinates']):
                    arr = np.array(subdata['cell_coordinates'], dtype=object)
                    coord_vars.add(var_name)
                    coords_all.setdefault(var_name, []).append(arr)

    if not coords_all:
        print("No 3D coordinate data found in simulations — building coordinates from cell morphology.")
        return False

    print("Variables containing cell_coordinates:")
    for v in sorted(coord_vars):
        print(f"  - {v}")

    print("\nCross-simulation consistency check:")
    all_ok = True
    for var, all_coords in coords_all.items():
        if not all_coords or any(c.size == 0 for c in all_coords):
            print(f"{var}: empty coordinate array detected")
            all_ok = False
            continue

        ref = all_coords[0]
        if ref.size == 0:
            print(f"{var}: empty coordinate array detected")
            all_ok = False
            continue

        matches = all(
            c.shape == ref.shape and np.allclose(
                c[:, 2:5].astype(float), ref[:, 2:5].astype(float), atol=1e-6
            )
            for c in all_coords[1:]
        )
        if matches:
            print(f"{var}: all coordinate sets match")
        else:
            print(f"{var}: coordinate mismatch detected")
            all_ok = False

    if all_ok:
        print("\nAll coordinate sets match across simulations")
    else:
        print("\nOne or more coordinate sets differ")

    return all_ok
    
def compare_synapse_coords(title, dend_locs, rel_locs, coords):
    """Display side-by-side comparison of simulation and interpolated synapse coordinates."""
    
    # Handle single coordinate (flat list or 1D array)
    if isinstance(coords, (list, tuple, np.ndarray)) and len(coords) in [5, 6] and not isinstance(coords[0], (list, tuple, np.ndarray)):
        coords = [coords]
    
    # Wrap single dendrite/location into lists
    if not isinstance(dend_locs, (list, tuple)):
        dend_locs = [dend_locs]
    if not isinstance(rel_locs, (list, tuple, np.ndarray)):
        rel_locs = [rel_locs]

    # Build text blocks
    lines_locs = [f"{str(d):<8}: {float(loc):>8.6f}" for d, loc in zip(dend_locs, rel_locs)]
    block_locs = "\n".join(lines_locs)

    # Handle coords with or without distance
    lines_coords = []
    for coord in coords:
        if len(coord) == 6:
            d, loc, x, y, z, dist = coord
            lines_coords.append(
                f"{str(d):<8}: {float(loc):>8.6f}, x={float(x):>7.3f}, y={float(y):>7.3f}, z={float(z):>7.3f}, dist={float(dist):>7.3f}"
            )
        else:
            d, loc, x, y, z = coord
            lines_coords.append(
                f"{str(d):<8}: {float(loc):>8.6f}, x={float(x):>7.3f}, y={float(y):>7.3f}, z={float(z):>7.3f}"
            )
    block_coords = "\n".join(lines_coords)

    # Display title and formatted HTML
    display(HTML(f"""
    <h3 style="text-align:center;">Comparison of Simulation and Interpolated {title} Coordinates</h3>
    <div style="display:flex; gap:40px; align-items:flex-start;">
      <div>
        <div style="font-weight:700; margin-bottom:6px;">{title} locations (simulation)</div>
        <pre style="margin:0; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size:13px; line-height:1.35;">
{block_locs}
        </pre>
      </div>
      <div>
        <div style="font-weight:700; margin-bottom:6px;">{title} synapse coordinates (interpolation)</div>
        <pre style="margin:0; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; font-size:13px; line-height:1.35;">
{block_coords}
        </pre>
      </div>
    </div>
    """))
def surface_area(cell, spines=None):
    h.define_shape()
    dend_secs = list(cell.dendlist)
    soma_secs = [cell.soma] if hasattr(cell, 'soma') else []

    dend_length = sum(sec.L for sec in dend_secs)
    dend_area = sum(h.area(seg.x, sec=sec) for sec in dend_secs for seg in sec)
    soma_area = sum(h.area(seg.x, sec=sec) for sec in soma_secs for seg in sec)
    area = dend_area + soma_area

    if spines is not None:
        spine_area = 0.0
        for sec in dend_secs:
            for _, spine in spines.get(sec.name(), {}).items():
                r_head = spine.head.diam / 2.0
                r_neck = spine.neck.diam / 2.0
                A_head = 4 * math.pi * (r_head ** 2)
                A_neck_lat = 2 * math.pi * r_neck * spine.neck.L
                A_junctions = 2 * math.pi * (r_neck ** 2)
                spine_area += A_head + A_neck_lat - A_junctions
        area += spine_area

    return area, dend_length

def get_plane_coords(coords, plane='xy', mirror=False):
    arr = np.asarray(coords, dtype=object)

    # ensure 2D
    if arr.ndim == 1:
        arr = arr[None, :]

    # accept [x,y,z], [sec,loc,x,y,z], or any shape with x,y,z at cols 2:5 (e.g., 7 cols)
    if arr.shape[1] >= 5:
        xyz = arr[:, 2:5].astype(float)
    elif arr.shape[1] == 3:
        xyz = arr.astype(float)
    else:
        raise ValueError("coords must have 3 columns (x,y,z) or ≥5 columns with x,y,z at positions 2:4")

    plane_map = {'x': 0, 'y': 1, 'z': 2}
    i1, i2 = plane_map[plane[0]], plane_map[plane[1]]
    coords_2d = xyz[:, [i1, i2]]

    if mirror:
        coords_2d[:, 0] = -coords_2d[:, 0]

    return coords_2d
    
    
def morphology_plot(cell_coordinates, dend_tree, dend_name=None, lwd=0.8, color='black', s=None,
                    height=600, width=600, scale_bar=50, title='morphology',
                    x_range=[-125, 175], y_range=[-150, 150],
                    plane='xy', mirror=False,
                    idxs=None, idxs_colors=None, alpha=0.5, dot_size=4):
    
    # # If dend_name is specified, filter dend_tree to only paths containing that dendrite
    # if dend_name is not None:
    #     dend_tree = extract_dend_lists(dend_tree, dend_name)


    # If dend_name is specified, filter dend_tree to only paths containing that dendrite
    if dend_name is not None:
        dend_tree = extract_dend_lists(dend_tree, dend_name)
        
        # Get all section names in filtered dend_tree
        valid_sections = set()
        for branch in dend_tree:
            for path in (branch if isinstance(branch[0], list) else [branch]):
                for sec in path:
                    valid_sections.add(sec.name())
        
        # Filter idxs to only include coordinates on valid sections
        if idxs is not None:
            if isinstance(idxs, list) and all(isinstance(i, (list, np.ndarray)) for i in idxs):
                # Multiple groups of coordinates
                filtered_idxs = []
                for coord_group in idxs:
                    coord_group = np.array(coord_group)
                    if coord_group.ndim == 2 and coord_group.shape[1] >= 5:
                        # Has section names in first column
                        mask = np.array([sec in valid_sections for sec in coord_group[:, 0]])
                        filtered_idxs.append(coord_group[mask])
                    else:
                        filtered_idxs.append(coord_group)  # Keep as is if format unknown
                idxs = filtered_idxs
            else:
                # Single group of coordinates
                idxs = np.array(idxs)
                if idxs.ndim == 2 and idxs.shape[1] >= 5:
                    mask = np.array([sec in valid_sections for sec in idxs[:, 0]])
                    idxs = idxs[mask]

                    
    # choose morphology base plot type
    if s is None:
        fig = morphology_plot1(cell_coordinates=cell_coordinates, dend_tree=dend_tree, lwd=lwd, color=color,
                               height=height, width=width, scale_bar=scale_bar, title=title,
                               x_range=x_range, y_range=y_range, plane=plane, mirror=mirror)
    else:
        fig = morphology_plot2(cell_coordinates=cell_coordinates, dend_tree=dend_tree, s=s, lwd=lwd, color=color,
                               height=height, width=width, scale_bar=scale_bar, title=title,
                               x_range=x_range, y_range=y_range, plane=plane, mirror=mirror)

    # add optional highlight points (supports both indices and coordinate arrays)
    if idxs is not None:
        if isinstance(idxs, list) and all(isinstance(i, (list, np.ndarray)) for i in idxs):
            if idxs_colors is None:
                idxs_colors = ['black'] * len(idxs)

            # handle per-group dot sizes and alpha values
            if isinstance(dot_size, (int, float)):
                dot_sizes = [dot_size] * len(idxs)
            else:
                dot_sizes = dot_size

            if isinstance(alpha, (int, float)):
                alphas = [alpha] * len(idxs)
            else:
                alphas = alpha

            for inds, c, size, a in zip(idxs, idxs_colors, dot_sizes, alphas):
                inds = np.array(inds)
                if inds.ndim == 2:
                    if inds.shape[1] >= 5:
                        coords_3d = inds[:, 2:5].astype(float)
                    elif inds.shape[1] == 3:
                        coords_3d = inds.astype(float)
                    else:
                        raise ValueError("Coordinate array must have 3 or ≥5 columns.")
                    coords_2d = get_plane_coords(coords_3d, plane, mirror)
                else:
                    coords_2d = get_plane_coords(cell_coordinates[np.array(inds, dtype=int)], plane, mirror)

                fig.add_trace(go.Scatter(
                    x=coords_2d[:, 0],
                    y=coords_2d[:, 1],
                    mode='markers',
                    marker=dict(color=c, size=size, opacity=a),
                    hoverinfo='none'
                ))

        else:
            idxs = np.array(idxs)
            if idxs.ndim == 2:
                if idxs.shape[1] >= 5:
                    coords_3d = idxs[:, 2:5].astype(float)
                elif idxs.shape[1] == 3:
                    coords_3d = idxs.astype(float)
                else:
                    raise ValueError("Coordinate array must have 3 or ≥5 columns.")
                coords_2d = get_plane_coords(coords_3d, plane, mirror)
            else:
                coords_2d = get_plane_coords(cell_coordinates[np.array(idxs, dtype=int)], plane, mirror)

            fig.add_trace(go.Scatter(
                x=coords_2d[:, 0],
                y=coords_2d[:, 1],
                mode='markers',
                marker=dict(color='black', size=dot_size, opacity=alpha),
                hoverinfo='none'
            ))

    return fig
    
def morphology_plot1(cell_coordinates, dend_tree, lwd=0.8, color='black',
                     height=600, width=600, scale_bar=50, title='morphology',
                     x_range=[-125, 175], y_range=[-150, 150], plane='xy', mirror=False):
    fig = go.Figure()
    simplified_dend_tree = [sub_branch for branch in dend_tree for sub_branch in (branch if isinstance(branch[0], list) else [branch])]
    connections = {}
    for branch in simplified_dend_tree:
        for i in range(len(branch) - 1):
            sec_name = branch[i].name()
            next_sec_name = branch[i + 1].name()
            if sec_name not in connections:
                connections[sec_name] = set()
            connections[sec_name].add(next_sec_name)

    section_coords = cell_coordinates[cell_coordinates[:, 0] == 'soma[0]']
    coords_2d = get_plane_coords(section_coords, plane, mirror)
    x_coords, y_coords = coords_2d[:, 0], coords_2d[:, 1]
    center_x, center_y = x_coords.mean(), y_coords.mean()
    radius = np.mean([coord[6] for coord in section_coords]) / 2

    fig.add_shape(type="circle", xref="x", yref="y",
                  x0=center_x - radius, y0=center_y - radius,
                  x1=center_x + radius, y1=center_y + radius,
                  line=dict(color=color, width=lwd), fillcolor=color)

    primaries = list(set(tree[0].name() for tree in simplified_dend_tree))

    def point_on_circle(cx, cy, angle, radius):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        return x, y

    for primary in primaries:
        section_coords = cell_coordinates[cell_coordinates[:, 0] == primary]
        if section_coords.size > 0:
            coords_2d = get_plane_coords(section_coords, plane, mirror)
            start_x, start_y = coords_2d[0, 0], coords_2d[0, 1]
            angle = np.arctan2(start_y - center_y, start_x - center_x)
            perimeter_x, perimeter_y = point_on_circle(center_x, center_y, angle, radius)
            fig.add_trace(go.Scatter(x=[perimeter_x, start_x], y=[perimeter_y, start_y],
                                     mode='lines', line=dict(color=color, width=lwd), hoverinfo='none'))
            hover_texts = ['dist: {:.2f}'.format(coord[5]) for coord in section_coords]
            fig.add_trace(go.Scatter(x=coords_2d[:, 0], y=coords_2d[:, 1],
                                     mode='lines', line=dict(color=color, width=lwd),
                                     text=hover_texts, hoverinfo='text'))

    for parent, children in connections.items():
        section_coords = cell_coordinates[cell_coordinates[:, 0] == parent]
        coords_2d = get_plane_coords(section_coords, plane, mirror)
        hover_texts = [f'{parent}, dist: {coord[5]:.2f}' for coord in section_coords]
        fig.add_trace(go.Scatter(x=coords_2d[:, 0], y=coords_2d[:, 1],
                                 mode='lines', line=dict(color=color, width=lwd),
                                 text=hover_texts, hoverinfo='text'))
        end_x, end_y = coords_2d[-1, 0], coords_2d[-1, 1]
        for child in children:
            section_coords = cell_coordinates[cell_coordinates[:, 0] == child]
            coords_2d = get_plane_coords(section_coords, plane, mirror)
            start_x, start_y = coords_2d[0, 0], coords_2d[0, 1]
            fig.add_trace(go.Scatter(x=[end_x, start_x], y=[end_y, start_y],
                                     mode='lines', line=dict(color=color, width=lwd), hoverinfo='none'))
            hover_texts = [f'{child}, dist: {coord[5]:.2f}' for coord in section_coords]
            fig.add_trace(go.Scatter(x=coords_2d[:, 0], y=coords_2d[:, 1],
                                     mode='lines', line=dict(color=color, width=lwd),
                                     text=hover_texts, hoverinfo='text'))

    fig.update_layout(title=title, title_x=0.5,
                      autosize=False, width=width, height=height,
                      xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False,
                                 scaleanchor='x', scaleratio=1),
                      showlegend=False, dragmode=False, hovermode='closest',
                      modebar_remove=['select2d', 'lasso2d'],
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    x1 = x_range[1]
    x0 = x1 - scale_bar
    y0 = y_range[0] + (y_range[1] - y_range[0]) / 3
    fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y0, line=dict(color=color, width=2))
    fig.add_annotation(x=(x0 + x1) / 2, y=y0 - 10, text=f'{scale_bar} µm', showarrow=False, font=dict(color=color, size=12))

    # makes svg better in Adobe Illustrator
    # fig.update_layout(font=dict(family='Myriad Pro', size=10, color='black'))
    fig.update_layout(font=dict(family='Myriad Pro-Regular', size=10, color=color))
    fig.update_traces(line_simplify=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    
    fig.update_traces(cliponaxis=False)  # keep this only
    fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)
    fig.update_layout(annotationdefaults=dict(visible=True))

    return fig

def morphology_plot2(cell_coordinates, dend_tree, lwd=0.8, color='black', s=2,
                     height=600, width=600, scale_bar=50, title='morphology',
                     x_range=[-125, 175], y_range=[-150, 150], plane='xy', mirror=False):
    fig = go.Figure()
    section_coords = cell_coordinates[cell_coordinates[:, 0] == 'soma[0]']
    coords_2d = get_plane_coords(section_coords, plane, mirror)
    x_coords, y_coords = coords_2d[:, 0], coords_2d[:, 1]
    center_x, center_y = x_coords.mean(), y_coords.mean()
    radius = np.mean([coord[6] for coord in section_coords]) / 2
    fig.add_shape(type="circle", xref="x", yref="y",
                  x0=center_x - radius, y0=center_y - radius,
                  x1=center_x + radius, y1=center_y + radius,
                  line=dict(color=color, width=lwd), fillcolor=color)

    def point_on_circle(cx, cy, angle, radius):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        return x, y

    def smooth_and_plot(coords):
        coords_2d = get_plane_coords(coords, plane, mirror)
        x_coords, y_coords = coords_2d[:, 0], coords_2d[:, 1]
        dists = [coord[5] for coord in coords]
        names = [coord[0] for coord in coords]
    
        # Remove consecutive duplicate points
        xy = np.column_stack((x_coords, y_coords))
        _, unique_idx = np.unique(xy.round(6), axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        x_coords, y_coords = xy[unique_idx].T
        dists = [dists[i] for i in unique_idx]
        names = [names[i] for i in unique_idx]
        
        # Create hover texts with CORRECT section names for each point
        hover_texts = [f'{name}, dist: {d:.2f}' for name, d in zip(names, dists)]
        unique_xy = np.unique(np.column_stack((x_coords, y_coords)), axis=0)
    
        if len(unique_xy) > 4 and len(x_coords) > 4:
            try:
                tck, _ = splprep([x_coords, y_coords], s=s)
                u_fine = np.linspace(0, 1, 300)
                x_smooth, y_smooth = splev(u_fine, tck)
    
                # Plot smoothed line WITHOUT hover
                fig.add_trace(go.Scatter(
                    x=x_smooth, y=y_smooth, mode='lines',
                    line=dict(color=color, width=lwd),
                    hoverinfo='none'
                ))
                
                # Add original points as markers WITH hover
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords, mode='markers',
                    marker=dict(size=0.1, color='grey', opacity=0),  # Invisible
                    text=hover_texts, hoverinfo='text',
                    showlegend=False
                ))
                return
            except Exception as e:
                print(f"\nSpline failed for {names[0]}: {e}")
                print("Coordinates after cleaning:")
                for x, y in zip(x_coords, y_coords):
                    print(f"({x:.3f}, {y:.3f})")
                print("—" * 40)
    
        # Fallback: non-smoothed line with hover
        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords, mode='lines',
            line=dict(color=color, width=lwd),
            text=hover_texts, hoverinfo='text'
        ))

    
    simplified_dend_tree = [sub_branch for branch in dend_tree for sub_branch in (branch if isinstance(branch[0], list) else [branch])]
    sorted_dend_tree = sorted(dend_tree, key=lambda sublist: -len(sublist) if isinstance(sublist, list) else -1)

    for path_group in sorted_dend_tree:
        if not isinstance(path_group, list):
            path_group = [path_group]
        for path in path_group:
            if not isinstance(path, list):
                path = [path]
            coords = []
            for sec in path:
                name = sec.name()
                cell_coords = cell_coordinates[cell_coordinates[:, 0] == name]
                coords.append(cell_coords)
            if coords:
                combined_coords = np.vstack(coords)
                smooth_and_plot(combined_coords)

    primaries = list(set(tree[0].name() for tree in simplified_dend_tree))
    for primary in primaries:
        section_coords = cell_coordinates[cell_coordinates[:, 0] == primary]
        if section_coords.size > 0:
            coords_2d = get_plane_coords(section_coords, plane, mirror)
            start_x, start_y = coords_2d[0, 0], coords_2d[0, 1]
            angle = np.arctan2(start_y - center_y, start_x - center_x)
            perimeter_x, perimeter_y = point_on_circle(center_x, center_y, angle, radius)
            fig.add_trace(go.Scatter(x=[perimeter_x, start_x], y=[perimeter_y, start_y],
                                     mode='lines', line=dict(color=color, width=lwd), hoverinfo='none'))

    fig.update_layout(title=title, title_x=0.5,
                      autosize=False, width=width, height=height,
                      xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False,
                                 scaleanchor='x', scaleratio=1),
                      showlegend=False, dragmode=False, hovermode='closest',
                      modebar_remove=['select2d', 'lasso2d'],
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    x1 = x_range[1]
    x0 = x1 - scale_bar
    y0 = y_range[0] + (y_range[1] - y_range[0]) / 3
    fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y0, line=dict(color=color, width=2))
    fig.add_annotation(x=(x0 + x1) / 2, y=y0 - 10, text=f'{scale_bar} µm', showarrow=False, font=dict(color=color, size=12))

    # makes svg better in Adobe Illustrator
    # fig.update_layout(font=dict(family='Myriad Pro', size=10, color='black'))
    fig.update_layout(font=dict(family='Myriad Pro-Regular', size=10, color=color))
    fig.update_traces(line_simplify=False)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    
    fig.update_traces(cliponaxis=False)  # keep this only
    fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)
    fig.update_layout(annotationdefaults=dict(visible=True))

    return fig

def get_coord_index(cell_coordinates, target_dendrite, target_location):
    # get cell_coordinates index for this location
    if isinstance(target_dendrite, (list, np.ndarray)):
        indices = []
        for dend, loc in zip(target_dendrite, target_location):
            # filter rows
            rows = np.where(cell_coordinates[:, 0] == dend)[0]
            # extract locations
            locations = cell_coordinates[rows, 1].astype(float)
            # find the closest index
            closest_idx = rows[np.argmin(np.abs(locations - loc))]
            indices.append(closest_idx)
        return indices
    else:
        # filter rows
        rows = np.where(cell_coordinates[:, 0] == target_dendrite)[0]
        
        # if target_location is None, return all indices for this dendrite
        if target_location is None:
            return rows
        
        # extract locations
        locations = cell_coordinates[rows, 1].astype(float)
        # find the closest index
        closest_idx = rows[np.argmin(np.abs(locations - target_location))]
        return closest_idx

def get_coord_index_interp(cell_coordinates, target_dendrite, target_location):
    # Extract soma radius from cell_coordinates (soma diameter is in column 6)
    soma_rows = np.where(np.char.find(cell_coordinates[:, 0].astype(str), 'soma') >= 0)[0]
    if len(soma_rows) > 0:
        soma_radius = float(cell_coordinates[soma_rows[0], 6]) / 2.0
    else:
        soma_radius = 0.0  # Fallback if no soma found
    
    def single_interp(dend, loc):
        if isinstance(loc, (list, np.ndarray)):
            loc = float(loc[0])
        rows = np.where(cell_coordinates[:, 0] == dend)[0]
        locs = cell_coordinates[rows, 1].astype(float)
        
        if loc <= locs.min():
            x, y, z = cell_coordinates[rows[0], 2:5].astype(float)
            dist_from_edge = float(cell_coordinates[rows[0], 5])
            dist_from_center = dist_from_edge + soma_radius
            return np.array([dend, loc, x, y, z, dist_from_center], dtype=object)
        
        if loc >= locs.max():
            x, y, z = cell_coordinates[rows[-1], 2:5].astype(float)
            dist_from_edge = float(cell_coordinates[rows[-1], 5])
            dist_from_center = dist_from_edge + soma_radius
            return np.array([dend, loc, x, y, z, dist_from_center], dtype=object)
        
        i2 = np.searchsorted(locs, loc)
        i1 = i2 - 1
        w = (loc - locs[i1]) / (locs[i2] - locs[i1])
        x = (1 - w) * float(cell_coordinates[rows[i1], 2]) + w * float(cell_coordinates[rows[i2], 2])
        y = (1 - w) * float(cell_coordinates[rows[i1], 3]) + w * float(cell_coordinates[rows[i2], 3])
        z = (1 - w) * float(cell_coordinates[rows[i1], 4]) + w * float(cell_coordinates[rows[i2], 4])
        dist_from_edge = (1 - w) * float(cell_coordinates[rows[i1], 5]) + w * float(cell_coordinates[rows[i2], 5])
        dist_from_center = dist_from_edge + soma_radius
        return np.array([dend, loc, x, y, z, dist_from_center], dtype=object)

    if isinstance(target_dendrite, (list, np.ndarray)):
        return np.array([single_interp(d, l) for d, l in zip(target_dendrite, target_location)], dtype=object)
    else:
        return single_interp(target_dendrite, target_location)
        
################## image clean-up ##################

import os, re

def clean_svg_directory(sim_image_path):    
    for file in os.listdir(sim_image_path):
        if not file.endswith(".svg"):
            continue
        full_path = os.path.join(sim_image_path, file)
        with open(full_path, "r", encoding="utf-8") as f:
            svg_data = f.read()

        # remove invisible rectangles and hidden groups
        svg_data = re.sub(r'<rect[^>]*(fill="none"[^>]*)>', '', svg_data)
        svg_data = re.sub(r'<rect[^>]*(opacity="0"[^>]*)>', '', svg_data)
        svg_data = re.sub(r'<g[^>]*display="none"[^>]*>.*?</g>', '', svg_data, flags=re.DOTALL)

        # rename key groups for Illustrator
        svg_data = re.sub(r'class="legend"', 'id="Legend_Group"', svg_data)
        svg_data = re.sub(r'class="xaxislayer-above"', 'id="X_Axis_Group"', svg_data)
        svg_data = re.sub(r'class="yaxislayer-above"', 'id="Y_Axis_Group"', svg_data)
        svg_data = re.sub(r'class="cartesianlayer"', 'id="Plot_Group"', svg_data)
        svg_data = re.sub(r'class="subplot xy"', 'id="Subplot_Group"', svg_data)

        # merge axis line into tick group (for Illustrator grouping)
        svg_data = re.sub(
            r'(<g id="X_Axis_Group".*?</g>)(\s*<path[^>]*stroke="black"[^>]*>)',
            r'<g id="X_Axis_Group_Combined">\1\2</g>',
            svg_data,
            flags=re.DOTALL
        )
        svg_data = re.sub(
            r'(<g id="Y_Axis_Group".*?</g>)(\s*<path[^>]*stroke="black"[^>]*>)',
            r'<g id="Y_Axis_Group_Combined">\1\2</g>',
            svg_data,
            flags=re.DOTALL
        )

        # enforce correct font family for Illustrator
        svg_data = re.sub(r'font-family:[^;"]*[";]', 'font-family:Myriad Pro;', svg_data)
        svg_data = re.sub(r'font-family="[^"]+"', 'font-family="Myriad Pro"', svg_data)

        # overwrite the original SVG
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(svg_data)

    print(f"SVG cleanup complete in: {sim_image_path}")

# for MatlibPlot

def save_clean_svg(fig, path):
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', transparent=True)
    buf.seek(0)
    tree = ET.parse(buf)
    root = tree.getroot()
    for elem in root.iter():
        if 'clip-path' in elem.attrib:
            del elem.attrib['clip-path']
    tree.write(path, encoding='utf-8', xml_declaration=True)
    
 # plot functions 

def plot9_MLP(x, ydict, yaxis='', xaxis='', _range=None, _range_subset=None,
              yaxis_range=[-110, 30], xaxis_range=[200, 1500],
              y_err_bar=10, x_err_bar=100, y_err_bar_shift=5,
              ybar_units='mV', xbar_units='ms',
              palette='oleron', reverse=False, col=None, alpha=1,
              lw=1, width=6, height=5, fig_title='',
              ds=10, offset=False, offset_ms=20, yabline=None,
              text_color='grey',
              sim_image_path=None, save_name='figure.svg', save=False):

    def _rgba_to_mpl(col):
        if isinstance(col, str) and col.startswith('rgba'):
            nums = col.replace('rgba(', '').replace(')', '').split(',')
            r, g, b, a = [float(x) for x in nums]
            return (r/255, g/255, b/255, a)
        return col

    def _set_font():
        try:
            font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            has_myriad = any('Myriad' in os.path.basename(f) for f in font_paths)
            plt.rcParams['font.family'] = 'Myriad Pro' if has_myriad else 'Calibri'
        except Exception:
            plt.rcParams['font.family'] = 'Calibri'

    _set_font()
    plt.rcParams['lines.solid_capstyle'] = 'round'
    plt.rcParams['lines.dash_capstyle'] = 'round'

    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    N = len(ydict)
    if col is None:
        cols = palette_cols(palette, N, alpha=alpha, reverse=reverse)
    else:
        cols = [col.replace('1.00', f'{alpha:.2f}')] * N
    cols = [_rgba_to_mpl(c) for c in cols]

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
        ax.plot(xvals, y_trim, color=cols[ii], lw=lw, label=name)

    all_x = np.concatenate(x_all)
    all_y = np.concatenate(y_all)
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    xaxis_range_extended = [x_min - x_pad, x_max + x_pad]
    yaxis_range_auto = [y_min - y_pad, y_max + y_pad]
    ax.set_xlim(xaxis_range_extended)
    ax.set_ylim(yaxis_range if yaxis_range is not None else yaxis_range_auto)

    ax.set_xlabel(xaxis, fontsize=10, color=text_color)
    ax.set_ylabel(yaxis, fontsize=10, color=text_color)
    ax.set_title(fig_title, fontsize=12, color=text_color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, colors=text_color)

    if yabline is not None:
        for yv in yabline:
            ax.hlines(y=yv, xmin=xaxis_range[0], xmax=xaxis_range[1],
                      color=text_color, lw=lw, ls=(0, (1, 2)))
            ax.text(xaxis_range[1] + (xaxis_range_extended[1] - xaxis_range_extended[0]) * 0.015,
                    yv, f'{round(yv, 3):g}', color=text_color, fontsize=10, va='center', ha='left')

    x0 = xaxis_range_extended[1] + (xaxis_range_extended[1] - xaxis_range_extended[0]) * 0.12
    y0 = yaxis_range[0] + (yaxis_range[1] - yaxis_range[0]) * 0.1

    ax.plot([x0, x0 + x_err_bar], [y0, y0], color=text_color, lw=lw, clip_on=False)
    ax.text(x0 + x_err_bar / 2, y0 - (yaxis_range[1] - yaxis_range[0]) * 0.015,
            f'{x_err_bar:g} {xbar_units}', ha='center', va='top', fontsize=10, color=text_color)

    ax.plot([x0, x0], [y0, y0 + y_err_bar], color=text_color, lw=lw, clip_on=False)
    ax.text(x0 - (xaxis_range_extended[1] - xaxis_range_extended[0]) * 0.005,
            y0 + y_err_bar / 2, f'{y_err_bar:g} {ybar_units}',
            ha='right', va='center', fontsize=10, color=text_color, rotation=90)

    leg = ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    for txt in leg.get_texts():
        txt.set_color(text_color)

    fig.tight_layout()
    plt.rcParams['svg.fonttype'] = 'none'

    if save:
        save_dir = sim_image_path if sim_image_path is not None else '.'
        os.makedirs(save_dir, exist_ok=True)
        save_clean_svg(fig, os.path.join(save_dir, save_name))

    plt.close(fig)
    return fig


def plot_mech_current_MLP(mech, data_dict, _range=None, _range_subset=None,
                          sim_image_path=None, sim_time=None, step_start=0,
                          width=6, height=5, dt=0.1, ds=10,
                          text_color='grey', save=False):

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

    fig = plot9_MLP(x=x, ydict=out, _range=_range, _range_subset=_range_subset,
                    xaxis_range=[step_start-100, sim_time], ds=ds,
                    yaxis_range=yaxis_range, y_err_bar=y_err_bar,
                    y_err_bar_shift=y_err_bar_shift, ybar_units='mA/cm²',
                    width=width, height=height, fig_title=mech,
                    text_color=text_color)

    if save:
        save_dir = sim_image_path if sim_image_path is not None else '.'
        os.makedirs(save_dir, exist_ok=True)
        save_clean_svg(fig, os.path.join(save_dir, f"fig_{mech}_summary.svg"))

    display(fig)

    return fig

def plot_xy_MLP(x, y, sim_image_path=None, save=False,
                xrange=(-100, -40), yrange=None,
                lw=1, width=6, height=4, marker_size=3,
                open_circles=False, dashed_lines=False,
                y_title='Y-axis', x_title='X-axis',
                title='XY Plot', save_name='plot_xy_MLP.svg',
                markers=True, col='#595959'):

    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ls = ':' if dashed_lines else '-'
    if open_circles:
        marker_style = {'marker': 'o', 'mfc': 'none', 'mec': col,
                        'ms': marker_size, 'lw': 0, 'linestyle': 'none'}
    else:
        marker_style = {'marker': 'o', 'mfc': col, 'mec': col,
                        'ms': marker_size, 'lw': 0, 'linestyle': 'none'}

    line_style = {'color': col, 'lw': lw, 'ls': ls}

    if markers:
        spline = UnivariateSpline(x, y, s=0.5 * len(x))
        x_fit = np.linspace(min(x), max(x), 300)
        y_fit = spline(x_fit)
        ax.plot(x_fit, y_fit, **line_style, label='Spline fit', zorder=1)
        ax.plot(x, y, **marker_style, label='Data', zorder=2)
    else:
        ax.plot(x, y, **line_style, label='Line', zorder=1)

    ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)

    ax.set_xlabel(x_title, color=col)
    ax.set_ylabel(y_title, color=col)
    ax.set_title(title, color=col)

    ax.spines['left'].set_color(col)
    ax.spines['bottom'].set_color(col)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(direction='out', length=3, width=0.8, colors=col)

    leg = ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.9))
    for txt in leg.get_texts():
        txt.set_color(col)

    plt.tight_layout()

    if save:
        save_dir = sim_image_path if sim_image_path is not None else '.'
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, save_name), format='svg', dpi=300, transparent=True)

    plt.close(fig)
    return fig
    
def plot_df(df, x_col, y_cols, sim_image_path=None, save=False,
            xrange=(-100, -40), yrange=None, 
            xtick_interval=None, ytick_interval=None,
            transparent=True,  lw=1, width=6, height=4,
            x_title='X-axis', y_title='Y-axis', title='Plot',
            palette='berlin', plot_color=None, alpha=1,
            save_name='plot_df.svg', log_y=False,
            open_circles=False, dashed_lines=False,
            marker_size=6, markers=True,
            interp='pchip', legend=True, text_color='darkgray', line_colors=None):

    fig, ax = plt.subplots(figsize=(width, height), dpi=100)
    
    # Set transparent background
    if transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    
    # Get base colors from palette
    if plot_color is not None:
        col_dict = {col_name: plot_color for col_name in y_cols}
    else:
        cols = [rgba_to_mpl(c) for c in palette_cols(palette, len(y_cols), alpha=alpha)]
        col_dict = dict(zip(y_cols, cols))
        
        if line_colors is not None:
            col_dict.update(line_colors)

    for col_name in y_cols:
        if col_name in df.columns:
            col = col_dict[col_name]
            xdata = df[x_col].to_numpy()
            ydata = df[col_name].to_numpy()
            
            if xrange is not None:
                mask = (xdata >= xrange[0]) & (xdata <= xrange[1])
                xdata = xdata[mask]
                ydata = ydata[mask]
            
            if not np.all(np.isnan(ydata)):
                ls = ':' if dashed_lines else '-'

                if markers:
                    x_fit = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 300)

                    if interp == 'pchip':
                        interp_func = PchipInterpolator(
                            xdata,
                            np.log10(ydata) if log_y else ydata
                        )
                        y_fit = 10 ** interp_func(x_fit) if log_y else interp_func(x_fit)

                    elif interp == 'spline':
                        spline = UnivariateSpline(xdata, ydata, s=0.5 * len(xdata))
                        y_fit = spline(x_fit)

                    elif interp == 'none':
                        y_fit = np.interp(x_fit, xdata, ydata)

                    ax.plot(x_fit, y_fit, color=col, lw=lw, ls=ls, label=col_name, zorder=1)

                    if open_circles:
                        ax.plot(xdata, ydata, marker='o', mfc='white', mec=col,
                            ms=marker_size, markeredgewidth=1, lw=0, linestyle='none', zorder=2)
                    else:
                        ax.plot(xdata, ydata, marker='o', mfc=col, mec=col,
                            ms=marker_size, markeredgewidth=1, lw=0, linestyle='none', zorder=2)
                else:
                    ax.plot(xdata, ydata, color=col, lw=lw, ls=ls, label=col_name, zorder=1)

    if log_y:
        ax.set_yscale('log')
    if yrange is not None:
        ax.set_ylim(yrange)
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 0.9, ymax * 1.1)

    if xrange is not None:
        ax.set_xlim(xrange)
    
    # Set custom x-tick interval if specified
    if xtick_interval is not None:
        if xrange is not None:
            x_start = np.ceil(xrange[0] / xtick_interval) * xtick_interval
            x_end = np.floor(xrange[1] / xtick_interval) * xtick_interval
        else:
            current_xlim = ax.get_xlim()
            x_start = np.ceil(current_xlim[0] / xtick_interval) * xtick_interval
            x_end = np.floor(current_xlim[1] / xtick_interval) * xtick_interval
        xticks = np.arange(x_start, x_end + xtick_interval, xtick_interval)
        ax.set_xticks(xticks)

    # Set custom y-tick interval if specified
    if ytick_interval is not None:
        current_yrange = ax.get_ylim()
        y_start = np.ceil(current_yrange[0] / ytick_interval) * ytick_interval
        y_end = np.floor(current_yrange[1] / ytick_interval) * ytick_interval
        yticks = np.arange(y_start, y_end + ytick_interval, ytick_interval)
        ax.set_yticks(yticks)
    
    ax.set_xlabel(x_title, fontsize=10, color=text_color)
    ax.set_ylabel(y_title, fontsize=10, color=text_color)
    ax.set_title(title, fontsize=10, color=text_color)

    # FIXED: Use subplots_adjust with consistent spacing instead of tight_layout
    if legend:
        leg = ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.9), fontsize=10)
        plt.setp(leg.get_texts(), color=text_color)
        plt.subplots_adjust(left=0.18, right=0.82, top=0.92, bottom=0.18)
    else:
        plt.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.18)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(text_color)
    ax.spines['bottom'].set_color(text_color)
    ax.tick_params(direction='out', length=3, width=0.8, colors=text_color)

    if save:
        save_dir = sim_image_path if sim_image_path is not None else '.'
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, save_name), format='svg', dpi=300, transparent=transparent)

    plt.close(fig)
    return fig
    
def plot4(rel_timing, P3_P1_soma, P3_P1_dend, sim_image_path=None, save=False,
          width=8, height=6, marker_size=6, lwd=1.0,
          x_title=r'$\mathrm{t}_{\mathrm{GLUT}} - \mathrm{t}_{\mathrm{GABA}}\;\mathrm{(ms)}$',
          y_title='P₂/P₁ ratio', title='', save_name='P3_P1_plot.svg',
          xrange=(-31, 131), yrange=(0, 3.5), x_tick_step=10, y_tick_step=1.0,
          palette=('slateblue', 'indianred'), markers=True, open_circles=False,
          text_color='gray', transparent=True):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots(figsize=(width, height))

    if transparent:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    else:
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

    soma_color, dend_color = palette

    if markers:
        if open_circles:
            marker_face = 'none' if transparent else 'white'

            ax.plot(rel_timing, P3_P1_soma,
                    color=soma_color, linewidth=lwd,
                    marker='o', markersize=marker_size,
                    markerfacecolor=marker_face,
                    markeredgecolor=soma_color,
                    markeredgewidth=lwd)

            ax.plot(rel_timing, P3_P1_dend,
                    color=dend_color, linewidth=lwd,
                    marker='o', markersize=marker_size,
                    markerfacecolor=marker_face,
                    markeredgecolor=dend_color,
                    markeredgewidth=lwd)
        else:
            ax.plot(rel_timing, P3_P1_soma,
                    color=soma_color, linewidth=lwd,
                    marker='o', markersize=marker_size)

            ax.plot(rel_timing, P3_P1_dend,
                    color=dend_color, linewidth=lwd,
                    marker='o', markersize=marker_size)
    else:
        ax.plot(rel_timing, P3_P1_soma,
                color=soma_color, linewidth=lwd)

        ax.plot(rel_timing, P3_P1_dend,
                color=dend_color, linewidth=lwd)

    ax.set_xlabel(x_title, color=text_color)
    ax.set_ylabel(y_title, color=text_color)
    ax.set_title(title, color=text_color)

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.xaxis.set_major_locator(MultipleLocator(x_tick_step))
    ax.yaxis.set_major_locator(MultipleLocator(y_tick_step))

    ax.tick_params(axis='both', colors=text_color)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(lwd)
    ax.spines['left'].set_linewidth(lwd)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['left'].set_color(text_color)

    ax.spines['bottom'].set_position(('outward', 4))
    ax.spines['left'].set_position(('outward', 4))

    plt.tight_layout()

    if save and sim_image_path is not None:
        os.makedirs(sim_image_path, exist_ok=True)
        fig.savefig(
            os.path.join(sim_image_path, save_name),
            format='svg',
            dpi=300,
            transparent=transparent
        )

    return fig
    
def rgba_to_mpl(col):
    if isinstance(col, str) and col.startswith('rgba'):
        vals = re.findall(r'[\d.]+', col)
        if len(vals) == 4:
            r, g, b, a = map(float, vals)
            return (r/255, g/255, b/255, a)
    return col

def plot3_dt(Varray, title=None, yaxis='V (mV)', yrange=[-86, 40], yabline=None,
             stim_time=150, sim_time=400, black_trace=None, gray_trace=None,
             x_err_bar=100, y_err_bar=10, rel_shift=0.3,
             palette='oleron', alpha=0.8, reverse=False,
             baseline=20, dt=0.025, width=1000, height=400,
             black_shift=200, ds=10, offset=False, offset_ms=None, offset_y=None, lwd=1,
             y_err_bar_units='mV', xrange=None, legend=True, err_bar_color='gray', abline_color='gray', text_color='gray'):

    if offset and offset_ms is None:
        offset_ms = 20

    n = len(Varray)
    if black_trace is None and gray_trace is None:
        cols = palette_cols(palette, n, alpha=alpha, reverse=reverse)
    elif black_trace is not None and gray_trace is None:
        cols = ['#000000'] + palette_cols(palette, n - 1, alpha=alpha, reverse=reverse)
    elif black_trace is not None and gray_trace is not None:
        base_cols = palette_cols(palette, n - 2, alpha=alpha, reverse=reverse)
        cols = ['#000000', '#C0C0C0'] + base_cols
    else:  # black_trace is None, gray_trace is not None
        base_cols = palette_cols(palette, n - 1, alpha=alpha, reverse=reverse)
        cols = base_cols[:gray_trace] + ['#C0C0C0'] + base_cols[gray_trace:]

    def update_layout(fig, title_str, yaxis_label, yrange_vals, width_val, height_val):
        fig.update_layout(
            autosize=False,
            width=width_val,
            height=height_val,
            margin=dict(l=20, r=90, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title=title_str,
            title_x=0.45,
            title_font=dict(family='Calibri', size=14, color=text_color),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(
                side='left',
                anchor='free',
                position=0,
                tick0=yrange_vals[0],
                dtick=(yrange_vals[1] - yrange_vals[0]),
                showticklabels=False,
                showgrid=False,
                zeroline=False
            ),
            showlegend=legend,
            legend=dict(title='sim', x=1.1, y=0.95, font=dict(color=text_color), title_font=dict(color=text_color)),
            font=dict(family='Calibri', size=14, color=text_color)
        )

    def add_corner_errbars(fig, x_right, x_err_bar_val, y_err_bar_val, units, rel_shift_val, yrange_vals):
        y0_data = yrange_vals[0] + rel_shift_val * (yrange_vals[1] - yrange_vals[0])
        x_left = x_right - x_err_bar_val

        fig.add_shape(
            type='line',
            x0=x_left, x1=x_right,
            y0=y0_data, y1=y0_data,
            xref='x', yref='y',
            line=dict(color=err_bar_color, width=lwd)
        )
        fig.add_annotation(
            x=x_left + x_err_bar_val / 2,
            y=y0_data,
            text=f'{x_err_bar_val:g} ms',
            showarrow=False,
            font=dict(color=err_bar_color, size=12),
            xref='x', yref='y',
            yanchor='top',
            yshift=-5
        )

        fig.add_shape(
            type='line',
            x0=x_left, x1=x_left,
            y0=y0_data, y1=y0_data + y_err_bar_val,
            xref='x', yref='y',
            line=dict(color=err_bar_color, width=lwd)
        )
        fig.add_annotation(
            x=x_left,
            y=y0_data + y_err_bar_val / 2,
            text=f'{round(y_err_bar_val, 3):g} {units}',
            showarrow=False,
            font=dict(color=err_bar_color, size=12),
            xref='x', yref='y',
            xanchor='right',
            xshift=-4,
            textangle=270
        )

    def plot3_(Varray_vals, title_str, yaxis_label, cols_vals, yrange_vals, x_err_bar_val, y_err_bar_val, bl,
               yabline_vals, units, rel_shift_val, width_val, height_val):
        fig = go.Figure()
        npts = len(Varray_vals[0][0]) if isinstance(Varray_vals[0], (list, tuple)) else len(Varray_vals[0])
        x_full = np.arange(npts) * dt
        ind1 = 0
        ind2 = int(round((sim_time - stim_time + bl) / dt))
        ind3 = int(round((stim_time - bl) / dt))
        ind4 = int(round(sim_time / dt))
        x_min_all, x_max_all = np.inf, -np.inf
        offset_val = 0
        offset_val_y = 0
        last_trace_offset = 0

        def apply_xrange_mask(xv, yv):
            if xrange is None:
                return xv, yv
            mask = (xv >= xrange[0]) & (xv <= xrange[1])
            return xv[mask], yv[mask]

        for ii in range(len(Varray_vals)):
            if gray_trace is not None and ii == gray_trace:
                continue
            yvals = Varray_vals[ii][0] if isinstance(Varray_vals[ii], (list, tuple)) else Varray_vals[ii]
            xvals = x_full[ind1:ind2][::ds]
            y_data = yvals[ind3:ind4][::ds]
            if black_trace is not None and ii == black_trace:
                xvals = xvals - black_shift
            if offset:
                if offset_ms is not None:
                    xvals = xvals + offset_val
                    offset_val += offset_ms
                if black_trace is None and offset_y is not None:
                    y_data = np.array(y_data, dtype=float) + offset_val_y
                    last_trace_offset = offset_val_y
                    offset_val_y += offset_y
            xvals, y_data = apply_xrange_mask(xvals, np.asarray(y_data))
            if len(xvals) == 0:
                continue
            x_min_all = min(x_min_all, xvals.min())
            x_max_all = max(x_max_all, xvals.max())
            fig.add_trace(go.Scatter(
                x=xvals, y=y_data, mode='lines',
                line=dict(color=cols_vals[ii], width=lwd),
                showlegend=legend
            ))

        if gray_trace is not None:
            yvals = Varray_vals[gray_trace][0] if isinstance(Varray_vals[gray_trace], (list, tuple)) else Varray_vals[gray_trace]
            xvals = x_full[ind1:ind2][::ds]
            y_data = yvals[ind3:ind4][::ds]
            if black_trace is not None and gray_trace == black_trace:
                xvals = xvals - black_shift
            xvals, y_data = apply_xrange_mask(xvals, np.asarray(y_data))
            if len(xvals) > 0:
                x_min_all = min(x_min_all, xvals.min())
                x_max_all = max(x_max_all, xvals.max())
                fig.add_trace(go.Scatter(
                    x=xvals, y=y_data, mode='lines',
                    line=dict(color=cols_vals[gray_trace], width=lwd),
                    showlegend=legend
                ))

        y_lines = yabline_vals if yabline_vals is not None else yrange_vals

        if offset_y is not None and y_lines is not None and len(y_lines) > 1:
            adjusted_y_lines = [y_lines[0], y_lines[1] + last_trace_offset]
            original_labels = y_lines
        else:
            adjusted_y_lines = y_lines
            original_labels = y_lines

        x_lo, x_hi = (xrange[0], xrange[1]) if xrange is not None else (x_min_all, x_max_all)

        for yv in adjusted_y_lines:
            fig.add_shape(
                type='line',
                x0=x_lo, x1=x_hi,
                y0=yv, y1=yv,
                line=dict(color=abline_color, width=lwd, dash='dot'),
                xref='x', yref='y'
            )

        for i, yv in enumerate(adjusted_y_lines):
            fig.add_annotation(
                x=1.01, y=yv,
                text=f'{round(original_labels[i], 3):g} {units}',
                showarrow=False,
                font=dict(color=abline_color, size=12),
                xref='paper', yref='y',
                xanchor='left', yanchor='middle'
            )

        add_corner_errbars(fig, x_hi, x_err_bar_val, y_err_bar_val, units, rel_shift_val, yrange_vals)
        update_layout(fig, title_str or 'response', yaxis_label, yrange_vals, width_val, height_val)
        return fig

    fig = plot3_(Varray_vals=Varray, title_str=title, yaxis_label=yaxis, cols_vals=cols, yrange_vals=yrange,
                 x_err_bar_val=x_err_bar, y_err_bar_val=y_err_bar, bl=baseline,
                 yabline_vals=yabline, units=y_err_bar_units, rel_shift_val=rel_shift,
                 width_val=width, height_val=height)

    fig.update_layout(font=dict(family='Myriad Pro-Regular', size=10, color=text_color),
                      legend=dict(font=dict(color=text_color), title_font=dict(color=text_color)))
    fig.update_traces(line_simplify=False, cliponaxis=False)
    fig.update_xaxes(showgrid=False, zeroline=False)

    y_pad = max(0.8, 0.03 * (yrange[1] - yrange[0]))
    fig.update_yaxes(showgrid=False, zeroline=False, range=[yrange[0] - y_pad, yrange[1]], autorange=False)

    fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)
    fig.update_layout(annotationdefaults=dict(visible=True))

    if xrange is not None:
        fig.update_xaxes(range=xrange)

    return fig

def extract_dend_lists(dend_tree, dend_name):
        def contains_dend_name(subtree, name):
            if isinstance(subtree, list):
                return any(contains_dend_name(item, name) for item in subtree)
            return subtree.name() == name
        
        extracted_lists = []
        for sublist in dend_tree:
            filtered_paths = [path for path in sublist if contains_dend_name(path, dend_name)]
            if filtered_paths:
                extracted_lists.append(filtered_paths)
        
        return extracted_lists

    
def heatmap2D(cell_coordinates, dend_tree, z, dend_name=None, palette='oleron', reverse=False, alpha=0.6, lwd=0.8,
          show_bar=True, title='', zmin=None, zmax=None, height=600, width=600, text_color='grey',
          scale_bar=50, x_range=[-125, 175], y_range=[-150, 150], plane='xy', mirror=False, s=None, ds=1):
    
    # if dend_name is specified, filter dend_tree to only paths containing that dendrite
    if dend_name is not None:
        dend_tree = extract_dend_lists(dend_tree, dend_name)
    
    if s is None:
        fig = heatmap2D_1(cell_coordinates=cell_coordinates, dend_tree=dend_tree, z=z, palette=palette, reverse=reverse,
                          alpha=alpha, lwd=lwd, show_bar=show_bar, title=title, zmin=zmin, zmax=zmax, text_color=text_color,
                          height=height, width=width, scale_bar=scale_bar, x_range=x_range, y_range=y_range,
                          plane=plane, mirror=mirror, ds=ds)
    else:
        fig = heatmap2D_2(cell_coordinates=cell_coordinates, dend_tree=dend_tree, z=z, palette=palette, reverse=reverse,
                          alpha=alpha, lwd=lwd, show_bar=show_bar, title=title, zmin=zmin, zmax=zmax, text_color=text_color,
                          height=height, width=width, scale_bar=scale_bar, x_range=x_range, y_range=y_range,
                          plane=plane, mirror=mirror, s=s, ds=ds)
    return fig

    
def heatmap2D_1(cell_coordinates, dend_tree, z, palette='oleron', reverse=False, alpha=0.6, lwd=0.8,
                show_bar=True, title='', zmin=None, zmax=None, height=600, width=600, text_color='grey',
                scale_bar=50, x_range=[-125, 175], y_range=[-150, 150], plane='xy', mirror=False, ds=1):
    # palette for continuous mapping
    Npal = 256
    cols = palette_cols(palette, Npal, alpha=alpha, reverse=reverse)

    def pick_color(val, lo, hi):
        if hi == lo:
            idx = 0
        else:
            t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
            idx = int(round(t * (Npal - 1)))
        return cols[idx]

    if zmin is None: zmin = float(np.min(z))
    if zmax is None: zmax = float(np.max(z))

    # build parent to children (same logic as morphology_plot1)
    simplified = [sub for branch in dend_tree for sub in (branch if isinstance(branch[0], list) else [branch])]
    connections = {}
    for branch in simplified:
        for i in range(len(branch) - 1):
            p = branch[i].name()
            c = branch[i + 1].name()
            connections.setdefault(p, set()).add(c)

    # quick lookup: each row’s z by its exact row tuple
    coord_to_z = {tuple(row.tolist()): float(val) for row, val in zip(cell_coordinates, z)}

    fig = go.Figure()

    # soma disk colored by its first row's z
    soma_rows = cell_coordinates[cell_coordinates[:, 0] == 'soma[0]'][::ds]
    if len(soma_rows):
        soma_xy = get_plane_coords(soma_rows, plane, mirror)
        cx, cy = soma_xy[:, 0].mean(), soma_xy[:, 1].mean()
        radius = float(np.mean(soma_rows[:, 6].astype(float)) / 2.0)
        soma_val = coord_to_z.get(tuple(soma_rows[0].tolist()), zmin)
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=cx - radius, y0=cy - radius, x1=cx + radius, y1=cy + radius,
                      line=dict(color='rgba(0,0,0,0)', width=0),
                      fillcolor=pick_color(soma_val, zmin, zmax))

    # primaries: colored link from soma edge to first primary point
    primaries = list(set([path[0].name() for path in simplified]))
    if len(soma_rows):
        soma_xy = get_plane_coords(soma_rows, plane, mirror)
        cx, cy = soma_xy[:, 0].mean(), soma_xy[:, 1].mean()
        radius = float(np.mean(soma_rows[:, 6].astype(float)) / 2.0)

        def point_on_circle(cx, cy, angle, radius):
            return cx + radius * np.cos(angle), cy + radius * np.sin(angle)

        for primary in primaries:
            sec_rows = cell_coordinates[cell_coordinates[:, 0] == primary][::ds]
            if sec_rows.size == 0:
                continue
            sec_xy = get_plane_coords(sec_rows, plane, mirror)
            x0, y0 = float(sec_xy[0, 0]), float(sec_xy[0, 1])
            ang = np.arctan2(y0 - cy, x0 - cx)
            px, py = point_on_circle(cx, cy, ang, radius)
            cval = coord_to_z.get(tuple(sec_rows[0].tolist()), zmin)
            fig.add_trace(go.Scatter(x=[px, x0], y=[py, y0], mode='lines',
                                     line=dict(color=pick_color(cval, zmin, zmax), width=lwd),
                                     hoverinfo='none'))

            # draw the primary section polyline itself
            for i in range(len(sec_xy) - 1):
                v = coord_to_z.get(tuple(sec_rows[i].tolist()), zmin)
                fig.add_trace(go.Scatter(x=[sec_xy[i, 0], sec_xy[i + 1, 0]],
                                         y=[sec_xy[i, 1], sec_xy[i + 1, 1]],
                                         mode='lines',
                                         line=dict(color=pick_color(v, zmin, zmax), width=lwd),
                                         hoverinfo='none'))

    # draw each section polyline as colored segments (color by start-point z)
    for parent, children in connections.items():
        parent_rows = cell_coordinates[cell_coordinates[:, 0] == parent][::ds]
        if parent_rows.size:
            pxy = get_plane_coords(parent_rows, plane, mirror)
            for i in range(len(pxy) - 1):
                cval = coord_to_z.get(tuple(parent_rows[i].tolist()), zmin)
                fig.add_trace(go.Scatter(x=[pxy[i, 0], pxy[i + 1, 0]],
                                         y=[pxy[i, 1], pxy[i + 1, 1]],
                                         mode='lines',
                                         line=dict(color=pick_color(cval, zmin, zmax), width=lwd),
                                         hoverinfo='none'))
        for child in children:
            child_rows = cell_coordinates[cell_coordinates[:, 0] == child][::ds]
            if child_rows.size == 0:
                continue
            # connector from end of parent to start of child
            if parent_rows.size:
                pxy_end = get_plane_coords(parent_rows[-2:], plane, mirror) if len(parent_rows) > 1 else get_plane_coords(parent_rows, plane, mirror)
                cxy0 = get_plane_coords(child_rows[:1], plane, mirror)
                cval0 = coord_to_z.get(tuple(child_rows[0].tolist()), zmin)
                fig.add_trace(go.Scatter(x=[pxy_end[-1, 0], cxy0[0, 0]], y=[pxy_end[-1, 1], cxy0[0, 1]],
                                         mode='lines',
                                         line=dict(color=pick_color(cval0, zmin, zmax), width=lwd),
                                         hoverinfo='none'))
            # child segments
            cxy = get_plane_coords(child_rows, plane, mirror)
            for i in range(len(cxy) - 1):
                cval = coord_to_z.get(tuple(child_rows[i].tolist()), zmin)
                fig.add_trace(go.Scatter(x=[cxy[i, 0], cxy[i + 1, 0]],
                                         y=[cxy[i, 1], cxy[i + 1, 1]],
                                         mode='lines',
                                         line=dict(color=pick_color(cval, zmin, zmax), width=lwd),
                                         hoverinfo='none'))

    # layout (match morphology_plot1)
    fig.update_layout(title=title, title_x=0.5,
                      autosize=False, width=width, height=height,
                      xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False,
                                 scaleanchor='x', scaleratio=1),
                      showlegend=False, dragmode=False, hovermode='closest',
                      modebar_remove=['select2d', 'lasso2d'],
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Myriad Pro', size=10, color=text_color))

    # scale bar
    x1 = x_range[1]; x0 = x1 - scale_bar
    y0 = y_range[0] + (y_range[1] - y_range[0]) / 3
    fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y0, line=dict(color=text_color, width=2))
    fig.add_annotation(x=(x0 + x1) / 2, y=y0 - 10, text=f'{scale_bar} µm', showarrow=False, font=dict(color=text_color, size=12))

    # colorbar (short, top, no box; only zmin/zmax labels)
    if show_bar:
        colorscale = [(i / (Npal - 1), cols[i]) for i in range(Npal)]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                colorscale=colorscale,
                cmin=zmin, cmax=zmax,
                color=[zmin, zmax],
                size=0, opacity=0,
                colorbar=dict(
                    title='',
                    thickness=10,
                    len=0.25, lenmode='fraction',
                    y=0.95, yanchor='top',
                    outlinewidth=0, borderwidth=0,
                    bgcolor='rgba(0,0,0,0)',
                    tickmode='array',
                    tickvals=[zmin, zmax],
                    ticktext=[f'{zmin:g}', f'{zmax:g}']
                ),
                showscale=True
            ),
            hoverinfo='none',
            showlegend=False
        ))

    fig.update_traces(line_simplify=False)
    return fig

def heatmap2D_2(cell_coordinates, dend_tree, z, palette='oleron', reverse=False, alpha=0.6, lwd=0.8,
                show_bar=True, title='', zmin=None, zmax=None, height=600, width=600, text_color='grey',
                scale_bar=50, x_range=[-125, 175], y_range=[-150, 150], plane='xy', mirror=False, s=2, ds=1):
    # palette for continuous mapping
    Npal = 256
    cols = palette_cols(palette, Npal, alpha=alpha, reverse=reverse)

    def pick_color(val, lo, hi):
        if hi == lo:
            idx = 0
        else:
            t = max(0.0, min(1.0, (val - lo) / (hi - lo)))
            idx = int(round(t * (Npal - 1)))
        return cols[idx]

    if zmin is None: zmin = float(np.min(z))
    if zmax is None: zmax = float(np.max(z))

    # keep the morphology_plot2 traversal order
    simplified = [sub for branch in dend_tree for sub in (branch if isinstance(branch[0], list) else [branch])]
    sorted_dend_tree = sorted(dend_tree, key=lambda sub: -len(sub) if isinstance(sub, list) else -1)

    # z by row lookup
    coord_to_z = {tuple(row.tolist()): float(val) for row, val in zip(cell_coordinates, z)}

    fig = go.Figure()

    # colored soma
    soma_rows = cell_coordinates[cell_coordinates[:, 0] == 'soma[0]']
    if len(soma_rows):
        soma_xy = get_plane_coords(soma_rows, plane, mirror)
        cx, cy = soma_xy[:, 0].mean(), soma_xy[:, 1].mean()
        radius = float(np.mean(soma_rows[:, 6].astype(float)) / 2.0)
        soma_val = coord_to_z.get(tuple(soma_rows[0].tolist()), zmin)
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=cx - radius, y0=cy - radius, x1=cx + radius, y1=cy + radius,
                      line=dict(color='rgba(0,0,0,0)', width=0),
                      fillcolor=pick_color(soma_val, zmin, zmax))
    else:
        cx = cy = radius = 0.0  # fallback if soma not present

    # primary section names (first section of each simplified path)
    primaries = list(set((path[0].name() if isinstance(path, list) else path.name()) for path in simplified))

    # helper: smooth a path and color tiny segments by interpolated z
    def smooth_and_plot(coords, values):
        xy = get_plane_coords(coords, plane, mirror)
        x = xy[:, 0].astype(float)
        y = xy[:, 1].astype(float)
        vals = np.asarray(values, dtype=float)
    
        mask = np.isfinite(x) & np.isfinite(y)
        x, y, vals = x[mask], y[mask], vals[mask]
    
        # remove consecutive or identical duplicate (x, y) pairs
        xy_pairs = np.column_stack((x, y))
        _, unique_idx = np.unique(xy_pairs.round(6), axis=0, return_index=True)
        xy_pairs = xy_pairs[np.sort(unique_idx)]
        x, y = xy_pairs[:, 0], xy_pairs[:, 1]
    
        unique_xy = np.unique(xy_pairs, axis=0)
    
        if len(unique_xy) > 4 and len(x) > 4:
            try:
                tck, u = splprep([x, y], s=s)
                u_f = np.linspace(0, 1, 200)
                xs, ys = splev(u_f, tck)
                zf = np.interp(u_f, np.linspace(0, 1, len(vals)), vals)
                xs, ys, zf = xs[::ds], ys[::ds], zf[::ds]
                for i in range(len(xs) - 1):
                    fig.add_trace(go.Scatter(
                        x=[xs[i], xs[i + 1]], y=[ys[i], ys[i + 1]],
                        mode='lines',
                        line=dict(color=pick_color(zf[i], zmin, zmax), width=lwd),
                        hoverinfo='none'
                    ))
            except Exception as e:
                print(f"\nSpline failed: {e}")
                print("Coordinates after cleaning:")
                for x0, y0 in zip(x, y):
                    print(f"({x0:.3f}, {y0:.3f})")
                print("—" * 40)
                # fallback straight lines
                for i in range(len(x) - 1):
                    fig.add_trace(go.Scatter(
                        x=[x[i], x[i + 1]], y=[y[i], y[i + 1]],
                        mode='lines',
                        line=dict(color=pick_color(vals[i], zmin, zmax), width=lwd),
                        hoverinfo='none'
                    ))
        else:
            print(f"Skipping short/degenerate segment: len={len(x)}, unique={len(unique_xy)}")
            x, y, vals = x[::ds], y[::ds], vals[::ds]
            for i in range(len(x) - 1):
                fig.add_trace(go.Scatter(
                    x=[x[i], x[i + 1]], y=[y[i], y[i + 1]],
                    mode='lines',
                    line=dict(color=pick_color(vals[i], zmin, zmax), width=lwd),
                    hoverinfo='none'
                ))
                
    # traverse each path group and draw colored curves
    for group in sorted_dend_tree:
        if not isinstance(group, list):
            group = [group]
        for path in group:
            path = path if isinstance(path, list) else [path]
            rows_list = []
            vals = []
            for sec in path:
                name = sec.name()
                rows = cell_coordinates[cell_coordinates[:, 0] == name]
                if rows.size == 0:
                    continue
                rows_list.append(rows)
                vals.extend([coord_to_z.get(tuple(r.tolist()), zmin) for r in rows])
            if not rows_list:
                continue
            combined = np.vstack(rows_list)

            # compute straight connector endpoints if this is a primary path
            draw_connector = False
            if path and (path[0].name() in primaries) and (combined.shape[0] > 0) and radius > 0:
                first_row_xy = get_plane_coords(combined[:1], plane, mirror)[0]
                start_x = float(first_row_xy[0]); start_y = float(first_row_xy[1])
                cxcy = get_plane_coords(np.array([[None, None, 0, 0, 0, 0, 0]], dtype=object), plane, mirror)[0]
                cx_p, cy_p = float(cx), float(cy)
                ang = np.arctan2(start_y - cy_p, start_x - cx_p)
                peri_x = cx_p + radius * np.cos(ang)
                peri_y = cy_p + radius * np.sin(ang)
                conn_color = pick_color(vals[0], zmin, zmax)
                draw_connector = True

            # spline only the dendrite (do NOT prepend the soma anchor)
            smooth_and_plot(combined, vals)

            # add straight soma→primary connector on top
            if draw_connector:
                fig.add_trace(go.Scatter(x=[peri_x, start_x], y=[peri_y, start_y],
                                         mode='lines',
                                         line=dict(color=conn_color, width=lwd),
                                         hoverinfo='none'))

    # layout (match morphology_plot2)
    fig.update_layout(title=title, title_x=0.5,
                      autosize=False, width=width, height=height,
                      xaxis=dict(range=x_range, showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(range=y_range, showgrid=False, zeroline=False, visible=False,
                                 scaleanchor='x', scaleratio=1),
                      showlegend=False, dragmode=False, hovermode='closest',
                      modebar_remove=['select2d', 'lasso2d'],
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Myriad Pro', size=10, color=text_color))
    # scale bar
    x1 = x_range[1]; x0 = x1 - scale_bar
    y0 = y_range[0] + (y_range[1] - y_range[0]) / 3
    fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y0, line=dict(color=text_color, width=2))
    fig.add_annotation(x=(x0 + x1) / 2, y=y0 - 10, text=f'{scale_bar} µm', showarrow=False, font=dict(color=text_color, size=12))

    # colorbar (no marker box; only zmin/zmax labels)
    if show_bar:
        colorscale = [(i / (Npal - 1), cols[i]) for i in range(Npal)]
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(
                colorscale=colorscale,
                cmin=zmin, cmax=zmax,
                color=[zmin, zmax],
                size=0, opacity=0,
                colorbar=dict(
                    title='',
                    thickness=10,
                    len=0.25, lenmode='fraction',
                    y=0.95, yanchor='top',
                    outlinewidth=0,
                    borderwidth=0,
                    bgcolor='rgba(0,0,0,0)',
                    tickmode='array',
                    tickvals=[zmin, zmax],
                    ticktext=[f'{zmin:g}', f'{zmax:g}']
                ),
                showscale=True
            ),
            hoverinfo='none',
            showlegend=False
        ))

    fig.update_layout(font=dict(family='Myriad Pro-Regular', size=10, color=text_color))
    fig.update_traces(line_simplify=False)

    return fig

def summarise_sim_entry(sim_data, entry_name):
    sim_entry = next(iter(sim_data.values()))[entry_name]
    for sim_key, subentry in sim_entry.items():
        print(f"{sim_key}: {list(subentry.keys())}")
        for k, v in subentry.items():
            if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'shape'):
                print(f"{k} count = {len(v)}")
                print(f"Example {k}[0] shape = {v[0].shape}")
        break

def get_mechs2display(sim_data, entry_name='i_mechs_3D'):
    first_entry = next(iter(sim_data.values()))[entry_name]
    mechs2display = next(iter(first_entry.values()))['mechs']
    return mechs2display

def extract_mech_currents(sim_data, mech_name, idxs, entry_name='i_mechs_3D'):
    I_all = {idx: [] for idx in idxs}
    mechs2display = get_mechs2display(sim_data, entry_name)
    mech_index = mechs2display.index(mech_name)

    for sim_vars in sim_data.values():
        i_all = sim_vars[entry_name]
        for i_entry in i_all.values():
            for idx in idxs:
                I_all[idx].append(i_entry['i'][idx][mech_index])

    return I_all

def roundup(x):
    # Choose an order of magnitude for x:
    exp = math.floor(math.log10(x)) if x>0 else 0
    base = 5**exp
    # Round x up to nearest multiple of base/10 if needed for finer granularity
    # Determine multiplier = ceil(x / base)
    mult = math.ceil(x / base)
    return mult * base

def get_y_range(traces, include_zero=False, target_ticks=10, dp=4):
    ymin = min(tr.min() for tr in traces)
    ymax = max(tr.max() for tr in traces)
    if include_zero and ymin > 0:
        ymin = 0.0
    rng = ymax - ymin
    if rng <= 0:
        span = abs(ymin) if ymin != 0 else 1.0
        return [round(ymin - 0.1 * span, dp), round(ymax + 0.1 * span, dp)]
    raw_step = rng / (target_ticks - 1)
    exponent = math.floor(math.log10(raw_step))
    base = 10 ** exponent
    nice_steps = [1, 1.5, 2, 2.5, 5, 10]
    for mult in nice_steps:
        step_candidate = base * mult
        if raw_step <= step_candidate:
            step = step_candidate
            break
    else:
        step = base * 10
    ymin = math.floor(ymin / step) * step
    ymax = math.ceil(ymax / step) * step
    return [round(ymin, dp), round(ymax, dp)]

def peak_fun(V, timing_range, dt=0.025, start_time=120.0, baseline=20.0, dp=5, window=None):
    peaks = []
    idxs = []
    t_peaks = []

    i0 = int(round((start_time - baseline) / dt))
    i1 = int(round(start_time / dt))

    for v, t_event in zip(V, timing_range):
        y = v[0] if isinstance(v, (list, tuple)) else v
        yb = y - np.mean(y[i0:i1])

        i_start = int(round(t_event / dt))
        
        # Apply window if specified
        if window is None:
            seg = yb[i_start:]
        else:
            i_window = int(round(window / dt))
            i_end = min(i_start + i_window, len(yb))
            seg = yb[i_start:i_end]
        
        j = int(np.argmax(seg))
        peaks.append(seg[j])
        idxs.append(i_start + j)
        t_peaks.append((i_start + j) * dt)

    return (
        np.round(np.array(peaks), dp),
        np.array(idxs, dtype=int),
        np.round(np.array(t_peaks), dp)
    )
    
def rel_peak_fun(V, timing_range, dt=0.025, start_time=120.0, baseline=20.0,
                 omit_trace=0, sub_trace=1, dp=5):
    peaks = []
    idxs = []
    t_peaks = []

    i0 = int(round((start_time - baseline) / dt))
    i1 = int(round(start_time / dt))

    y_ref = V[sub_trace][0] if isinstance(V[sub_trace], (list, tuple)) else V[sub_trace]
    y_refb = y_ref - np.mean(y_ref[i0:i1])

    for i, (v, t_event) in enumerate(zip(V, timing_range)):
        y = v[0] if isinstance(v, (list, tuple)) else v
        yb = y - np.mean(y[i0:i1])
        
        # Handle different length traces by padding to match longest
        if i == omit_trace:
            diff = yb
        else:
            max_len = max(len(yb), len(y_refb))
            if len(yb) < max_len:
                yb_padded = np.pad(yb, (0, max_len - len(yb)), mode='edge')
            else:
                yb_padded = yb
            if len(y_refb) < max_len:
                y_refb_padded = np.pad(y_refb, (0, max_len - len(y_refb)), mode='edge')
            else:
                y_refb_padded = y_refb
            diff = yb_padded - y_refb_padded

        i_start = int(round(t_event / dt))
        seg = diff[i_start:]
        j = int(np.argmax(seg))
        peaks.append(seg[j])
        idxs.append(i_start + j)
        t_peaks.append((i_start + j) * dt)

    return (
        np.round(np.array(peaks), dp),
        np.array(idxs, dtype=int),
        np.round(np.array(t_peaks), dp)
    )
    
def count_spikes(V, dt=0.025, threshold=0.0, refractory_ms=2.0):

    if len(V) == 0:
        return 0
    
    refractory_steps = int(refractory_ms / dt)
    spike_count = 0
    last_spike_idx = -refractory_steps
    
    for i in range(1, len(V)):
        if V[i] > threshold and V[i-1] <= threshold:
            if i - last_spike_idx > refractory_steps:
                spike_count += 1
                last_spike_idx = i
    
    return spike_count
    
def downsample_and_save_data(wd, sim, output_wd=None, parent_suffix='_ds', ds=10,
                             cell_type=None, impedance_ignore=True, verbose=False):

    import gc, shutil

    def ds_array(x):
        return x[::ds] if isinstance(x, np.ndarray) and x.ndim > 0 else x

    def ds_dataframe(df):
        return df.iloc[::ds].reset_index(drop=True) if isinstance(df, pd.DataFrame) else df

    def ds_dict(d):
        if isinstance(d, dict):
            for k in list(d.keys()):
                v = d[k]
                if isinstance(v, pd.DataFrame):
                    d[k] = ds_dataframe(v)
                elif isinstance(v, np.ndarray):
                    d[k] = ds_array(v)
                elif isinstance(v, dict):
                    ds_dict(v)
                elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], np.ndarray):
                    d[k] = [ds_array(x) for x in v]
                del v
        return d

    sim_dir = os.path.join(wd, sim)
    if output_wd is None:
        new_parent = wd + parent_suffix
        output_dir = os.path.join(new_parent, sim)
    else:
        output_dir = os.path.join(output_wd, sim)

    if verbose:
        print("="*60)
        print("Downsampling and saving data")
        print("="*60)
        print(f"Source: {sim_dir}")
        print(f"Output: {output_dir}")
        print(f"Downsampling factor: {ds}")
        print(f"Impedance downsampling: {'skip' if impedance_ignore else 'yes'}")

    os.makedirs(output_dir, exist_ok=True)
    total_start = time.time()
    gc.disable()

    sim_folders = sorted(
        [d for d in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, d))],
        key=lambda x: [int(text) if text.isdigit() else text for text in re.split('(\d+)', x)]
    )

    if sim_folders:
        for sim_name in sim_folders:
            sim_path = os.path.join(sim_dir, sim_name)
            pickle_files = [f for f in os.listdir(sim_path) if f.endswith('.pickle')]
            if not pickle_files:
                continue

            sim_output_path = os.path.join(output_dir, sim_name)
            os.makedirs(sim_output_path, exist_ok=True)

            if verbose:
                print(f"\nProcessing {sim_name}...")
                start = time.time()

            for f in pickle_files:
                var_name = f.replace('.pickle', '')

                if var_name == 'metadata':
                    shutil.copy2(os.path.join(sim_path, f), os.path.join(sim_output_path, f))
                    continue

                with open(os.path.join(sim_path, f), 'rb') as handle:
                    var_data = pickle.load(handle)

                if impedance_ignore and 'imp' in var_name.lower():
                    downsampled = var_data
                elif isinstance(var_data, pd.DataFrame):
                    downsampled = ds_dataframe(var_data)
                elif isinstance(var_data, np.ndarray):
                    downsampled = ds_array(var_data)
                elif isinstance(var_data, dict):
                    downsampled = ds_dict(var_data)
                elif isinstance(var_data, (list, tuple)) and len(var_data) > 0 and isinstance(var_data[0], np.ndarray):
                    downsampled = [ds_array(x) for x in var_data]
                else:
                    downsampled = var_data

                output_file = os.path.join(sim_output_path, f)
                with open(output_file, 'wb') as handle:
                    pickle.dump(downsampled, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.flush()
                    os.fsync(handle.fileno())

                del var_data, downsampled
                gc.collect()

            if verbose:
                elapsed = time.time() - start
                print(f"  {len(pickle_files)} files processed in {elapsed:.1f} s")

    else:
        pickle_files = [f for f in os.listdir(sim_dir) if f.endswith('.pickle')]

        if verbose:
            print(f"\nProcessing flat structure...")
            start = time.time()

        for f in pickle_files:
            var_name = f.replace('.pickle', '')

            if var_name == 'metadata':
                shutil.copy2(os.path.join(sim_dir, f), os.path.join(output_dir, f))
                continue

            with open(os.path.join(sim_dir, f), 'rb') as handle:
                var_data = pickle.load(handle)

            if impedance_ignore and 'imp' in var_name.lower():
                downsampled = var_data
            elif isinstance(var_data, pd.DataFrame):
                downsampled = ds_dataframe(var_data)
            elif isinstance(var_data, np.ndarray):
                downsampled = ds_array(var_data)
            elif isinstance(var_data, dict):
                downsampled = ds_dict(var_data)
            elif isinstance(var_data, (list, tuple)) and len(var_data) > 0 and isinstance(var_data[0], np.ndarray):
                downsampled = [ds_array(x) for x in var_data]
            else:
                downsampled = var_data

            output_file = os.path.join(output_dir, f)
            with open(output_file, 'wb') as handle:
                pickle.dump(downsampled, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())

            del var_data, downsampled
            gc.collect()

        if verbose:
            elapsed = time.time() - start
            print(f"  {len(pickle_files)} files processed in {elapsed:.1f} s")

    gc.enable()
    total_elapsed = time.time() - total_start

    if verbose:
        print("\n" + "="*60)
        print(f"COMPLETE: All downsampled data saved to {output_dir}")
        print(f"Total time: {total_elapsed:.1f} s")
        print("="*60)

    return output_dir
    
def extract_dend_to_target(dend_tree, root_name, target_dend):

    def contains_dend_name(subtree, name):
        if isinstance(subtree, list):
            return any(contains_dend_name(item, name) for item in subtree)
        return subtree.name() == name
    
    # find the path with target_dend to get reference dendrites
    target_path_dends = set()
    for sublist in dend_tree:
        for path in sublist:
            if contains_dend_name(path, root_name) and contains_dend_name(path, target_dend):
                path_names = [item.name() for item in path]
                target_path_dends.update(path_names)
    
    extracted_lists = []
    already_included = set(target_path_dends)  # track dendrites already included
    
    for sublist in dend_tree:
        filtered_paths = []
        for path in sublist:
            if contains_dend_name(path, root_name):
                path_names = [item.name() for item in path]
                
                if root_name not in path_names:
                    continue
                    
                start_idx = path_names.index(root_name)
                
                if target_dend in path_names:
                    # full path from beginning to target
                    target_idx = path_names.index(target_dend)
                    segment = path[:target_idx + 1]
                    filtered_paths.append(segment)
                else:
                    # only dendrites after root_name that haven't been included yet
                    segment = []
                    for item in path[start_idx + 1:]:
                        if item.name() not in already_included:
                            segment.append(item)
                            already_included.add(item.name())
                    
                    if segment:
                        filtered_paths.append(segment)
        
        if filtered_paths:
            extracted_lists.append(filtered_paths)
    
    return extracted_lists
    
def plot_v(X, Y, titles, colors=None, palette='oleron', alpha=1, reverse=False, yname='', 
           xname='distance from soma (µm)', xrange=[-2, 275], yrange=[0, 70], ab1=None, 
           smooth=1, width=1000, height=600, points=True, ignore_first=False, ds=10, 
           legend=False, points_size=6):
    
    fig = go.Figure()

    # Handle colors
    if colors is None:
        colors = ['gray'] * len(X)
    elif isinstance(colors, str):
        colors = [colors] * len(X)
    elif len(colors) == 1 and len(X) > 1:
        colors = colors * len(X)
    
    # If palette is specified and colors is default, use palette
    if palette and colors == ['gray'] * len(X):
        import plotly.express as px
        color_scale = px.colors.get_colorscale(palette)
        n_colors = len(X)
        if reverse:
            color_scale = color_scale[::-1]
        colors = [f'rgba({int(c[1][4:-1].split(",")[0])},{int(c[1][4:-1].split(",")[1])},{int(c[1][4:-1].split(",")[2])},{alpha})' 
                  if 'rgba' in c[1] else c[1] 
                  for c in color_scale[:n_colors]]
        if n_colors == 1:
            colors = [color_scale[len(color_scale)//2][1]]

    for i, (x, y, title, color) in enumerate(zip(X, Y, titles, colors), start=1):
        if x is not None and y is not None:
            # Apply alpha to color if not already rgba
            if not color.startswith('rgba'):
                if color.startswith('rgb'):
                    color = color.replace('rgb', 'rgba').replace(')', f',{alpha})')
                elif color.startswith('#'):
                    # Convert hex to rgba
                    h = color.lstrip('#')
                    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
                    color = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})'
                else:
                    # Named color, just use it with marker opacity
                    pass
            
            # Spline fitting uses ALL points (not downsampled)
            x_for_spline, y_for_spline = (x[1:], y[1:]) if ignore_first else (x, y)

            # Perform spline fitting on the full data
            sorted_pairs = sorted(zip(x_for_spline, y_for_spline))
            x_for_spline, y_for_spline = zip(*sorted_pairs)
            s = UnivariateSpline(x_for_spline, y_for_spline, s=smooth)
            xnew = np.linspace(min(x_for_spline), max(x_for_spline), 1000)
            ynew = s(xnew)

            # Plot the spline fit FIRST so it appears below points
            fig.add_trace(go.Scatter(x=xnew, y=ynew, mode='lines', 
                                    line=dict(dash='dot', color=color), 
                                    name=f'{title} spline fit', opacity=alpha))
            
            # Plot downsampled data points SECOND so they appear above lines
            if points:
                indices = list(range(0, len(x), ds))
                if indices[-1] != len(x) - 1:
                    indices.append(len(x) - 1)
                x_ds = x[indices]
                y_ds = y[indices]
                fig.add_trace(go.Scatter(x=x_ds, y=y_ds, mode='markers', name=f'{title}', 
                                        marker=dict(color=color, opacity=alpha, size=points_size)))

    if ab1 is not None:
        fig.add_shape(type="line", x0=ab1, x1=ab1, y0=yrange[0], y1=yrange[1], 
                     line=dict(color="gray", width=1, dash="dot"))

    fig.update_layout(
        title={'text': '', 'x': 0.5, 'xanchor': 'center'},
        font=dict(family='Myriad Pro-Regular', size=10, color='black'),
        xaxis=dict(
            title=xname,
            range=xrange,
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            tickcolor='black',
            showgrid=False,
            zeroline=False,
            layer='below traces'
        ),
        yaxis=dict(
            title=yname,
            range=yrange,
            showline=True,
            linewidth=1,
            linecolor='black',
            ticks='outside',
            tickcolor='black',
            showgrid=False,
            zeroline=False,
            autorange=False,
            layer='below traces'
        ),
        width=width,
        height=height,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=legend,
        legend=dict(
            orientation="v",
            yanchor="auto",
            y=1,
            xanchor="left",
            x=1.05
        ),
        legend_traceorder='normal',
        legend_tracegroupgap=0,
        annotationdefaults=dict(visible=True),
        margin=dict(r=150)
    )

    fig.update_layout(font=dict(family='Myriad Pro-Regular', size=10, color='black'))
    fig.update_traces(line_simplify=False, cliponaxis=False)
    fig.update_xaxes(showgrid=False, zeroline=False, layer='below traces')
    fig.update_yaxes(showgrid=False, zeroline=False, range=yrange, autorange=False, layer='below traces')
    fig.update_layout(legend_traceorder='normal', legend_tracegroupgap=0)

    return fig


def plot_v_mpl(X, Y, titles, colors='darkgray', palette='oleron', alpha=1, reverse=False, yname='', 
               xname='distance from soma (µm)', xrange=[-2, 275], yrange=[0, 70], ab1=None, 
               smooth=1, width=1000, height=600, points=True, ignore_first=False, ds=10, legend=False, points_size=6):
    
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.interpolate import UnivariateSpline
    import numpy as np
    
    # Convert width/height from pixels to inches (assuming 100 dpi)
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    
    # Make background transparent
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    # Handle colors
    if colors is None:
        colors = ['gray'] * len(X)
    elif isinstance(colors, str):
        colors = [colors] * len(X)
    elif len(colors) == 1 and len(X) > 1:
        colors = colors * len(X)
    
    # If palette is specified, use matplotlib colormap
    if palette and colors == ['gray'] * len(X):
        try:
            cmap = cm.get_cmap(palette)
        except:
            cmap = cm.get_cmap('viridis')
        n_colors = len(X)
        color_indices = np.linspace(0, 1, n_colors)
        if reverse:
            color_indices = color_indices[::-1]
        colors = [cmap(idx) for idx in color_indices]
    
    for i, (x, y, title, color) in enumerate(zip(X, Y, titles, colors), start=1):
        if x is not None and y is not None:
            # Spline fitting uses ALL points (not downsampled)
            x_for_spline, y_for_spline = (x[1:], y[1:]) if ignore_first else (x, y)
            
            # Perform spline fitting on the full data
            sorted_pairs = sorted(zip(x_for_spline, y_for_spline))
            x_for_spline, y_for_spline = zip(*sorted_pairs)
            s = UnivariateSpline(x_for_spline, y_for_spline, s=smooth)
            xnew = np.linspace(min(x_for_spline), max(x_for_spline), 1000)
            ynew = s(xnew)
            
            # Plot the spline fit FIRST (so it appears below points)
            ax.plot(xnew, ynew, linestyle=':', color=color, alpha=alpha, label=f'{title} spline fit')
            
            # Plot downsampled data points SECOND (so they appear above lines)
            if points:
                indices = list(range(0, len(x), ds))
                if indices[-1] != len(x) - 1:
                    indices.append(len(x) - 1)
                x_ds = x[indices]
                y_ds = y[indices]
                ax.scatter(x_ds, y_ds, color=color, alpha=alpha, s=points_size**2, label=f'{title}')
    
    # Add vertical line if specified
    if ab1 is not None:
        ax.axvline(x=ab1, color='gray', linestyle=':', linewidth=1)
    
    # Set axis properties
    ax.set_xlabel(xname, fontfamily='sans-serif', fontsize=10, color='darkgray')
    ax.set_ylabel(yname, fontfamily='sans-serif', fontsize=10, color='darkgray')
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('darkgray')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_color('darkgray')
    ax.spines['bottom'].set_linewidth(1)
    
    # Style the ticks
    ax.tick_params(axis='both', which='major', labelsize=10, colors='darkgray', 
                   direction='out', length=4, width=1)
    ax.tick_params(axis='both', which='minor', colors='darkgray')
    
    # Remove grid
    ax.grid(False)
    
    # Legend
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False, fontsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

    
def extract_baseline_min_max(traces, dt, stim_time=150, baseline_period=50):
    """
    Extract baseline, min, and max values from traces.
    
    Parameters:
    - traces: list of traces (voltage, impedance, current, etc.)
    - dt: time step (ms)
    - stim_time: stimulation start time (ms)
    - baseline_period: duration of baseline before stim (ms)
    
    Returns:
    - baseline_vals: baseline (mean before stim)
    - min_vals: minimum value after stim
    - max_vals: maximum value after stim
    """
    baseline_vals = []
    min_vals = []
    max_vals = []
    
    # Calculate indices
    i_baseline_start = int((stim_time - baseline_period) / dt)
    i_baseline_end = int(stim_time / dt)
    i_stim = int(stim_time / dt)
    
    for trace in traces:
        # Handle if trace is wrapped in a list
        t = trace[0] if isinstance(trace, (list, tuple)) else trace
        
        # Baseline: mean before stimulation
        baseline = np.mean(t[i_baseline_start:i_baseline_end])
        
        # Min and max after stimulation
        t_after_stim = t[i_stim:]
        t_min = np.min(t_after_stim)
        t_max = np.max(t_after_stim)
        
        baseline_vals.append(baseline)
        min_vals.append(t_min)
        max_vals.append(t_max)
    
    return np.array(baseline_vals), np.array(min_vals), np.array(max_vals)

def extract_delta_min_max_range(traces, dt, stim_time=150, baseline_period=50):
    """
    Extract delta (change from baseline) min, max, and range.
    
    Parameters:
    - traces: list of traces (voltage, impedance, current, etc.)
    - dt: time step (ms)
    - stim_time: stimulation start time (ms)
    - baseline_period: duration of baseline before stim (ms)
    
    Returns:
    - delta_min: minimum change from baseline (most negative = largest drop)
    - delta_max: maximum change from baseline (largest increase)
    - delta_range: range of change (max - min)
    """
    delta_min = []
    delta_max = []
    delta_range = []
    
    # Calculate indices
    i_baseline_start = int((stim_time - baseline_period) / dt)
    i_baseline_end = int(stim_time / dt)
    i_stim = int(stim_time / dt)
    
    for trace in traces:
        # Handle if trace is wrapped in a list
        t = trace[0] if isinstance(trace, (list, tuple)) else trace
        
        # Baseline: mean before stimulation
        baseline = np.mean(t[i_baseline_start:i_baseline_end])
        
        # Get trace after stimulation
        t_after_stim = t[i_stim:]
        
        # Calculate delta (change from baseline)
        delta_t = t_after_stim - baseline
        
        delta_min.append(np.min(delta_t))
        delta_max.append(np.max(delta_t))
        delta_range.append(np.max(delta_t) - np.min(delta_t))
    
    return np.array(delta_min), np.array(delta_max), np.array(delta_range)

def animate_traces(fig, skip_indices=None, highlight_color='white', highlight_width=None,
                   frame_duration=60, transition_duration=0, highlighted_trace=2):
    import copy
    import plotly.graph_objects as go

    fig = copy.deepcopy(fig)

    orig_colors = [t.line.color for t in fig.data]
    orig_widths = [t.line.width or 1 for t in fig.data]
    skip = set(skip_indices or [])
    animated_indices = [i for i in range(len(fig.data)) if i not in skip]

    fig.add_trace(go.Scatter(
        x=[], y=[], mode='lines',
        line=dict(color=highlight_color, width=1),
        showlegend=False
    ))
    spotlight_idx = len(fig.data) - 1

    if highlighted_trace is not None:
        if highlighted_trace in skip:
            raise ValueError("highlighted_trace is in skip_indices.")
        if highlighted_trace < 0 or highlighted_trace >= spotlight_idx:
            raise ValueError("highlighted_trace is out of range.")

        # Start after initial highlight, then EXCLUDE it from playback so no return-to-start at end
        if highlighted_trace in animated_indices:
            k = animated_indices.index(highlighted_trace)
            animated_indices = animated_indices[k + 1:] + animated_indices[:k + 1]
            animated_indices = [i for i in animated_indices if i != highlighted_trace]

    def make_frame(highlight_idx=None, name='0'):
        frame_data = []
        for i in range(spotlight_idx):
            frame_data.append(go.Scatter(
                line=dict(color=orig_colors[i], width=orig_widths[i])
            ))

        if highlight_idx is not None:
            t = fig.data[highlight_idx]
            w = highlight_width if highlight_width is not None else (t.line.width or 1)
            frame_data.append(go.Scatter(
                x=t.x, y=t.y, mode='lines',
                line=dict(color=highlight_color, width=w),
                showlegend=False
            ))
        else:
            frame_data.append(go.Scatter(
                x=[], y=[], mode='lines',
                showlegend=False
            ))

        return go.Frame(data=frame_data, name=name)

    if highlighted_trace is not None:
        t0 = fig.data[highlighted_trace]
        w0 = highlight_width if highlight_width is not None else (t0.line.width or 1)
        fig.data[spotlight_idx].x = t0.x
        fig.data[spotlight_idx].y = t0.y
        fig.data[spotlight_idx].line.color = highlight_color
        fig.data[spotlight_idx].line.width = w0
        frames = [make_frame(highlight_idx=highlighted_trace, name='0')]
    else:
        frames = [make_frame(name='0')]

    for step, idx in enumerate(animated_indices, 1):
        frames.append(make_frame(highlight_idx=idx, name=str(step)))

    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                x=0.965, y=0.95, xanchor='right', yanchor='top',
                pad=dict(l=0, r=0, t=0, b=0),
                bgcolor='rgba(0,0,0,0)',   # transparent
                bordercolor='rgba(0,0,0,0)',
                borderwidth=0,
                font=dict(family='Calibri', size=10, color='white'),
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, dict(
                            frame=dict(duration=frame_duration, redraw=True),
                            transition=dict(duration=0),
                            fromcurrent=False,
                            mode='afterall'
                        )]
                    )
                ]
            ),
            dict(
                type='buttons',
                showactive=False,
                x=0.966, y=0.95, xanchor='left', yanchor='top',
                pad=dict(l=0, r=0, t=0, b=0),
                borderwidth=0,
                bgcolor='rgba(0,0,0,0)',
                font=dict(family='Calibri', size=10, color=highlight_color),
                buttons=[
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], dict(
                            frame=dict(duration=0, redraw=True),
                            transition=dict(duration=0),
                            mode='immediate'
                        )]
                    )
                ]
            )
        ],
        sliders=[dict(
            active=0,
            x=0.05, len=0.95,      # keep width
            y=0.0,
            pad=dict(t=2, b=0, l=0, r=0),
            ticklen=0,
            tickwidth=1,
            font=dict(family='Calibri', size=9),
            currentvalue=dict(
                prefix='trace: ',
                font=dict(family='Calibri', size=9),
                offset=4
            ),
            steps=[dict(
                method='animate',
                args=[[f.name], dict(
                    mode='immediate',
                    frame=dict(duration=frame_duration, redraw=True),
                    transition=dict(duration=0)
                )],
                label=f.name
            ) for f in frames]
        )]    
    )
    return fig
    
def export_html(fig, path):
    _dark_script = """
document.head.insertAdjacentHTML('beforeend',
  '<style>@media(prefers-color-scheme:dark){body{background:#1a1a1a}}' +
  '@media(prefers-color-scheme:light){body{background:#fff}}</style>'
);
(function(){
  var gd = document.querySelectorAll('.plotly-graph-div')[0];
  if (!gd) return;
  function applyTheme(dark) { Plotly.relayout(gd, {'font.color': dark ? '#fff' : '#000'}); }
  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  applyTheme(mq.matches);
  if (mq.addEventListener) mq.addEventListener('change', function(e){ applyTheme(e.matches); });
})();
"""
    fig.write_html(path, auto_play=False, post_script=_dark_script)


def export_gif(fig, path, fps=10, scale=1):
    import imageio.v2 as imageio, io, copy
    import plotly.graph_objects as go
    base = copy.deepcopy(fig)
    images = []
    for frame in fig.frames:
        for i, trace in enumerate(frame.data):
            base.data[i].update(trace)
        img_bytes = base.to_image(format='png', scale=scale)
        images.append(imageio.imread(io.BytesIO(img_bytes)))
    imageio.mimsave(path, images, fps=fps)

def export_mp4(fig, path, fps=5, scale=4, crf=10, preset="slow"):
    import io
    import copy
    import numpy as np
    import imageio.v2 as imageio
    import plotly.graph_objects as go

    src = copy.deepcopy(fig)
    src.layout.updatemenus = ()
    src.layout.sliders = ()

    frames = list(src.frames or [])
    if len(frames) <= 1:
        raise ValueError(f"Animation has {len(frames)} frame(s). Need > 1.")

    base_data = [t.to_plotly_json() for t in src.data]
    base_layout = src.layout.to_plotly_json()

    writer = imageio.get_writer(
        path,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=[
            "-crf", str(crf),
            "-preset", preset,
            "-tune", "animation",
            "-bf", "0",
            "-movflags", "+faststart",
        ],
    )

    try:
        for fr in frames:
            # Fresh figure each frame: eliminates cumulative update artifacts
            frame_fig = go.Figure(
                data=copy.deepcopy(base_data),
                layout=copy.deepcopy(base_layout),
            )

            idxs = list(fr.traces) if getattr(fr, "traces", None) else list(range(len(fr.data)))
            for ti, tr in zip(idxs, fr.data):
                upd = tr.to_plotly_json() if hasattr(tr, "to_plotly_json") else tr
                frame_fig.data[ti].update(upd, overwrite=True)

            img = imageio.imread(io.BytesIO(frame_fig.to_image(format="png", scale=scale)))

            # flatten alpha onto black
            if img.ndim == 3 and img.shape[2] == 4:
                a = img[..., 3:4].astype(np.float32) / 255.0
                img = (img[..., :3].astype(np.float32) * a).astype(np.uint8)

            writer.append_data(img)
    finally:
        writer.close()

def plot3_timing(timing_range, sim_time, black_trace=None, gray_trace=None,
                 palette='oleron', alpha=0.8, reverse=False,
                 width=1000, height=250, lwd=2, tick_height=5,
                 title='stimulus timing (ms)', black_shift=200, baseline=20, line_gap=10,
                 text_color='gray', line_color=None):

    if line_color is None:
        line_color = text_color

    n = len(timing_range)
    if black_trace is None and gray_trace is None:
        cols = palette_cols(palette, n, alpha=alpha, reverse=reverse)
    elif black_trace is not None and gray_trace is None:
        cols = [text_color] + palette_cols(palette, n - 1, alpha=alpha, reverse=reverse)
    elif black_trace is not None and gray_trace is not None:
        base_cols = palette_cols(palette, n - 2, alpha=alpha, reverse=reverse)
        cols = [text_color, '#C0C0C0'] + base_cols

    fig = go.Figure()

    if gray_trace is not None:
        # double the separation between the two lines
        y_top = line_gap
        y_bottom = -line_gap
        fig.add_shape(type='line', x0=0, x1=sim_time, y0=y_top, y1=y_top,
                      line=dict(color=line_color, width=lwd))
        fig.add_shape(type='line', x0=0, x1=sim_time, y0=y_bottom, y1=y_bottom,
                      line=dict(color=line_color, width=lwd))
        # non-gray ticks (down from top)
        for ii in range(n):
            if gray_trace is not None and ii == gray_trace:
                continue
            x = timing_range[ii] - baseline
            if black_trace is not None and ii == black_trace:
                x = (timing_range[ii] - baseline) - black_shift
            fig.add_shape(type='line', x0=x, x1=x, y0=y_top, y1=y_top - tick_height,
                          line=dict(color=cols[ii], width=lwd))
        # gray tick (up from bottom)
        x = timing_range[gray_trace] - baseline
        if black_trace is not None and gray_trace == black_trace:
            x = (timing_range[gray_trace] - baseline) - black_shift
        fig.add_shape(type='line', x0=x, x1=x, y0=y_bottom, y1=y_bottom + tick_height,
                      line=dict(color=cols[gray_trace], width=lwd))
    else:
        # single line, all ticks down
        y0 = 0
        fig.add_shape(type='line', x0=0, x1=sim_time, y0=y0, y1=y0,
                      line=dict(color=line_color, width=lwd))
        for ii in range(n):
            x = timing_range[ii] - baseline
            if black_trace is not None and ii == black_trace:
                x = (timing_range[ii] - baseline) - black_shift
            fig.add_shape(type='line', x0=x, x1=x, y0=y0, y1=y0 - tick_height,
                          line=dict(color=cols[ii], width=lwd))

    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,
        title_x=0.45,
        title_font=dict(family='Calibri', size=14, color=text_color),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, sim_time]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   range=[-tick_height - 2 * line_gap, tick_height + 2 * line_gap]),
        font=dict(family='Calibri', size=14, color=text_color)
    )
    
    return fig
