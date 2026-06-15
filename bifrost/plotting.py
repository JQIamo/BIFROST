import numpy as np
from numpy import linspace
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from utils import JonesMtoMuellerM, MuellertoAxisAngle


# Setup stuff
degrees = np.pi / 180

colors = matplotlib.colors.TABLEAU_COLORS
name_colors = list(colors)


# Helper functions
def azel_2_xyz(az, el, pol=1):
    phi = 2*az #NOTE: azimuth is multiplied by 2 to match the Poincare sphere convention
    chi = -2*el + 90*degrees
    x = pol * np.sin(chi) * np.cos(phi)
    y = pol * np.sin(chi) * np.sin(phi)
    z = pol * np.cos(chi)
    return x, y, z

def obj_2_xyz(S, in_degrees=False):
    az, el = S.parameters.azimuth_ellipticity(out_number=False, use_nan=False)
    x, y, z = azel_2_xyz(az, el)
    if in_degrees:
        ret = [x, y, z, az/degrees, el/degrees]
    else:
        ret = [x, y, z, az, el]
    return ret

def rotation_arc(axis, start_vec, angle, n_points=120):

    # Build an orthonormal frame: axis, u (start), v (u × axis)
    u = start_vec - np.dot(start_vec, axis) * axis
    u_norm = np.linalg.norm(u)
    if u_norm < 1e-10:
        perp = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
        u = perp - np.dot(perp, axis) * axis
        u /= np.linalg.norm(u)
    else:
        u /= u_norm
    v = np.cross(axis, u)

    ts = np.linspace(0, angle, n_points)
    arc = np.outer(np.cos(ts), u) + np.outer(np.sin(ts), v)
    # Project onto the sphere (radius = distance of start_vec from axis)
    r = np.linalg.norm(start_vec - np.dot(start_vec, axis) * axis)
    arc *= r
    arc += np.dot(start_vec, axis) * axis[np.newaxis, :]
    return arc[:, 0], arc[:, 1], arc[:, 2]

# Poincare Sphere Functions
def build_poincare_sphere(fig=None, figsize=(6, 6), draw_axes=True,
                           draw_guides=True, n_subplots=1):
    """
    Creates (or reuses) a Plotly figure and populates it with the Poincaré
    sphere surface, axis lines, and guide circles.
    """
    add_auxiliar = fig is None
    if add_auxiliar:
        specs = [[{"type": "surface"} for _ in range(n_subplots)]]
        fig = make_subplots(rows=1, cols=n_subplots, specs=specs,
                            horizontal_spacing=0.05)

    lighting = dict(ambient=0.9, diffuse=0., roughness=0.5,
                    specular=0.05, fresnel=0.2)
    hovertemplate = ("S1: %{x:.3f}<br>S2: %{y:.3f}<br>S3: %{z:.3f}"
                     "<br>Parameter: %{customdata:.3f}")

    # Axis lines
    axis_annotations = []
    axis_traces = []
    if draw_axes and add_auxiliar:
        marker = dict(size=1)
        size = 30
        for label, color, xs, ys, zs, xsh, ysh in [
            ("S1",  "red",   [0, 1.2],  [0, 0],    [0, 0],    15,  15),
            ("-S1", "red",   [0, -1.2], [0, 0],    [0, 0],   -15, -15),
            ("S2",  "green", [0, 0],    [0, 1.2],  [0, 0],    15,  15),
            ("-S2", "green", [0, 0],    [0, -1.2], [0, 0],   -15, -15),
            ("S3",  "blue",  [0, 0],    [0, 0],    [0, 1.2],  15,  15),
            ("-S3", "blue",  [0, 0],    [0, 0],    [0, -1.2],-15, -15),
        ]:
            dash = "dash" if label.startswith("-") else "solid"
            axis_traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, marker=marker, hoverinfo="skip",
                name=label, line=dict(width=6, color=color, dash=dash)))
            axis_annotations.append(dict(
                showarrow=False, x=xs[-1], y=ys[-1], z=zs[-1], text=label,
                font=dict(color=color, size=size, family="Times New Roman"),
                xshift=xsh, yshift=ysh))

    # Guide circles
    guide_traces = []
    if draw_guides and add_auxiliar:
        angle = np.linspace(0, 360 * degrees, 361)
        line = dict(width=2, color="darkslategrey", dash="dashdot")
        s, c, z = np.sin(angle), np.cos(angle), np.zeros_like(angle)
        guide_traces = [
            go.Scatter3d(x=s, y=c, z=z, line=line, mode="lines", name="S3=0"),
            go.Scatter3d(x=s, y=z, z=c, line=line, mode="lines", name="S2=0"),
            go.Scatter3d(x=z, y=s, z=c, line=line, mode="lines", name="S1=0"),
        ]

    # Sphere surface 
    if add_auxiliar:
        el, az = np.mgrid[-45 * degrees:45 * degrees:100j,
                           0:180 * degrees:100j]
        sx, sy, sz = azel_2_xyz(az, el)
        sphere_customdata = np.dstack((az / degrees, el / degrees))

    for col in range(1, n_subplots + 1):
        if axis_traces:
            fig.add_traces(axis_traces, rows=[1] * len(axis_traces),
                           cols=[col] * len(axis_traces))
        if guide_traces:
            fig.add_traces(guide_traces, rows=[1] * len(guide_traces),
                           cols=[col] * len(guide_traces))
        if add_auxiliar:
            fig.add_trace(go.Surface(
                x=sx, y=sy, z=sz,
                surfacecolor=np.ones_like(sx) * 0.2,
                cmin=0, cmax=1, opacity=0.7, colorscale="Blues",
                showscale=False, lighting=lighting,
                customdata=sphere_customdata,
                hovertemplate=hovertemplate,
                name="Sphere", showlegend=False,
            ), row=1, col=col)

    return fig, add_auxiliar, axis_annotations

def poincare_fig(fig, n_subplots, figsize, annotations_per_subplot):
    """
    Applies figure-wide layout settings and per-subplot annotations/camera.
    """
    axis_dict = dict(showbackground=False, showgrid=False,
                     zeroline=False, visible=False)
    camera = dict(up=dict(x=0, y=0, z=1),
                  center=dict(x=0, y=0, z=0),
                  eye=dict(x=0.85, y=0.85, z=0.85))
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        width=int(figsize[0] * 100 * n_subplots),
        height=int(figsize[1] * 100),
        showlegend=False,
    )
    for col in range(1, n_subplots + 1):
        scene_key = "scene" if col == 1 else f"scene{col}"
        fig.layout[scene_key].annotations = annotations_per_subplot.get(col, [])
        fig.layout[scene_key].xaxis = axis_dict
        fig.layout[scene_key].yaxis = axis_dict
        fig.layout[scene_key].zaxis = axis_dict
        fig.layout[scene_key].camera = camera
    return fig

# Main Functions #####

def plot_stokes(*datasets,
                  fig=None,
                  figsize=(6, 6),
                  draw_axes=True,
                  draw_guides=True,
                  param=None,
                  #in_degrees=False,
                  param_name=None,
                  #log=False,
                  colormap="Blackbody",
                  n_subplots=1,
                  datasets_per_subplot=None):

    n = len(datasets)
    if not isinstance(param, list):       param       = [param]       * n
    if not isinstance(param_name, list):  param_name  = [param_name]  * n
    if not isinstance(colormap, list):    colormap    = [colormap]    * n

    if datasets_per_subplot is None:
        datasets_per_subplot = [n // n_subplots] * n_subplots
        for i in range(n % n_subplots):
            datasets_per_subplot[i] += 1

    subplot_assignment = []
    for sp_idx, count in enumerate(datasets_per_subplot):
        subplot_assignment.extend([sp_idx + 1] * count)

    fig, _, axis_annotations = build_poincare_sphere(
        fig=fig, figsize=figsize, draw_axes=draw_axes,
        draw_guides=draw_guides, n_subplots=n_subplots)

    hovertemplate = ("S1: %{x:.3f}<br>S2: %{y:.3f}<br>S3: %{z:.3f}"
                     "<br>Parameter: %{customdata:.3f}")
    subplot_colorbar_count = [0] * (n_subplots + 1)

    for ds_idx, S in enumerate(datasets):
        S = S.copy()
        col          = subplot_assignment[ds_idx]
        ds_param     = param[ds_idx]
        ds_pname     = param_name[ds_idx] if param_name[ds_idx] is not None else S.name
        ds_colormap  = colormap[ds_idx]
        within_idx   = subplot_colorbar_count[col]
        subplot_colorbar_count[col] += 1

        x, y, z, az, el = obj_2_xyz(S, in_degrees=True)
        customdata = np.squeeze(np.dstack((az, el)))

        subplot_width = 1.0 / n_subplots
        colorbar_x = (col - 1) * subplot_width + subplot_width / 2
        colorbar_y = -0.1 - within_idx * 0.15

        if isinstance(ds_param, str):
            Scolor = eval("S.parameters." + ds_param + "(out_number=False)")
            colorbar = dict(title=ds_param, orientation="h",
                            ticklabelposition="outside bottom",
                            y=colorbar_y, x=colorbar_x,
                            len=subplot_width * 0.8, anchor="center")
        elif ds_param is not None:
            Scolor = np.array(ds_param).flatten()
            colorbar = dict(title=ds_pname, orientation="h",
                            ticklabelposition="outside bottom",
                            y=colorbar_y, x=colorbar_x,
                            len=subplot_width * 0.8, xanchor="center",
                            tickfont=dict(size=15))
        else:
            Scolor   = np.ones(len(x)) * (ds_idx + 1) / n
            colorbar = {}

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name=S.name,
            marker=dict(size=10, color=Scolor, colorbar=colorbar,
                        colorscale=ds_colormap),
            customdata=customdata, hovertemplate=hovertemplate,
        ), row=1, col=col)

    # Same axis annotations for every subplot
    annotations_per_subplot = {col: axis_annotations
                                for col in range(1, n_subplots + 1)}
    poincare_fig(fig, n_subplots, figsize, annotations_per_subplot)
    fig.show()
    return fig

def plot_jones_matrix(*datasets,
               fig=None,
               figsize=(6, 6),
               draw_axes=True,
               draw_guides=True,
               n_subplots=1,
               datasets_per_subplot=None,
               axis_color="black",
               arc_color="crimson",
               axis_length=1.25,
               arc_width=5,
               axis_width=8,
               n_arrows=3,
               start_vec=None,
               show_angle_label=True):

    n = len(datasets)

    if datasets_per_subplot is None:
        datasets_per_subplot = [n // n_subplots] * n_subplots
        for i in range(n % n_subplots):
            datasets_per_subplot[i] += 1

    subplot_assignment = []
    for sp_idx, count in enumerate(datasets_per_subplot):
        subplot_assignment.extend([sp_idx + 1] * count)

    fig, _, axis_annotations = build_poincare_sphere(
        fig=fig, figsize=figsize, draw_axes=draw_axes,
        draw_guides=draw_guides, n_subplots=n_subplots)

    # Per-subplot angle label annotations (added as each dataset is processed)
    subplot_angle_annotations = {col: [] for col in range(1, n_subplots + 1)}

    for ds_idx, J in enumerate(datasets):
        J   = J.copy()
        col = subplot_assignment[ds_idx]

        M = JonesMtoMuellerM(J)
        axis, angle = MuellertoAxisAngle(M)

        ax, ay, az = axis * axis_length
        fig.add_trace(go.Scatter3d(
            x=[-ax, ax], y=[-ay, ay], z=[-az, az], mode="lines",
            line=dict(color=axis_color, width=axis_width),
            name="Rotation axis", hoverinfo="skip",
        ), row=1, col=col)

        sv = start_vec
        if sv is None:
            perp = np.array([1, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1, 0])
            sv   = perp - np.dot(perp, axis) * axis
            sv  /= np.linalg.norm(sv)

        x_arc, y_arc, z_arc = rotation_arc(axis, sv, angle)

        fig.add_trace(go.Scatter3d(
            x=x_arc, y=y_arc, z=z_arc, mode="lines",
            line=dict(color=arc_color, width=arc_width),
            name="Rotation arc (%.1f°)" % (angle / degrees),
            hoverinfo="name",
        ), row=1, col=col)

        if n_arrows > 0:
            for i in np.linspace(10, len(x_arc) - 2, n_arrows, dtype=int):
                tangent = np.array([x_arc[i+1] - x_arc[i-1],
                                    y_arc[i+1] - y_arc[i-1],
                                    z_arc[i+1] - z_arc[i-1]])
                tangent /= np.linalg.norm(tangent)
                fig.add_trace(go.Cone(
                    x=[x_arc[i]], y=[y_arc[i]], z=[z_arc[i]],
                    u=[tangent[0]], v=[tangent[1]], w=[tangent[2]],
                    sizemode="absolute", sizeref=0.12,
                    colorscale=[[0, arc_color], [1, arc_color]],
                    showscale=False, anchor="tip", hoverinfo="skip",
                ), row=1, col=col)

        if show_angle_label:
            mid = len(x_arc) // 2
            subplot_angle_annotations[col].append(dict(
                x=x_arc[mid] * 1.1, y=y_arc[mid] * 1.1, z=z_arc[mid] * 1.1,
                text="%.1f°" % (angle / degrees),
                showarrow=False,
                font=dict(color=arc_color, size=18, family="Times New Roman"),
            ))

    # Combine shared axis labels with per-subplot angle labels
    annotations_per_subplot = {col: axis_annotations + subplot_angle_annotations[col]
                                for col in range(1, n_subplots + 1)}
    poincare_fig(fig, n_subplots, figsize, annotations_per_subplot)
    fig.show()
    return fig

def plot_ellipse(*datasets,
                 figsize=(6, 6),
                 N_angles=91,
                 draw_arrow=True,
                 draw_titles=True,
                 draw_labels=True,
                 limit=None,
                 param=None,
                 param_name=None,
                 colormap=None,
                 n_subplots=1,
                 datasets_per_subplot=None,
                 arrow_color=None,
                 **kwargs):
    
    n = len(datasets)
    if n == 0:
        raise ValueError("At least one dataset must be provided.")
 
    # Normalise list arguments to length n
    if not isinstance(param, list):
        param = [param] * n
    if not isinstance(param_name, list):
        param_name = [param_name] * n
    if not isinstance(colormap, list):
        colormap = [colormap] * n
 
    # Replace None colormaps with defaults
    for i in range(n):
        if colormap[i] is None:
            colormap[i] = 'viridis' if param[i] is not None else None
 
    # Distribute datasets across subplots
    if datasets_per_subplot is None:
        base = n // n_subplots
        rem  = n % n_subplots
        datasets_per_subplot = [base + (1 if i < rem else 0) for i in range(n_subplots)]
 
    subplot_assignment = []
    for sp_idx, count in enumerate(datasets_per_subplot):
        subplot_assignment.extend([sp_idx] * count)  # 0-indexed
 
    # Compute per-subplot x and y limits independently
    use_fixed_limit = limit not in [0, '', [], None]

    # Computing ellipse data
    all_Ex = []
    all_Ey = []
    for E in datasets:
        if E._type == 'Jones_vector':
            E0x, E0y = E.parameters.amplitudes(shape=False)
        else:
            raise ValueError("Only Jones_vector is supported.")

        delay = E.parameters.delay(shape=False)
        phase = E.parameters.global_phase(shape=False)
        if phase is None:
            phase = np.zeros_like(E0x)
        if np.isnan(phase).any():
            phase[np.isnan(phase)] = 0

        angles = linspace(0, 360 * degrees, N_angles)
        Angles, E0X = np.meshgrid(angles, E0x)
        _,      E0Y = np.meshgrid(angles, E0y)
        _,    Delay = np.meshgrid(angles, delay)
        _,    Phase = np.meshgrid(angles, phase)

        Ex = E0X * np.cos(Phase - Angles)
        Ey = E0Y * np.cos(Phase + Delay - Angles)
        all_Ex.append(Ex)
        all_Ey.append(Ey)

    # Per-subplot limits
    subplot_xlim = {}
    subplot_ylim = {}
    for sp_idx in range(n_subplots):
        ds_in_sp = [i for i in range(n) if subplot_assignment[i] == sp_idx]
        if use_fixed_limit:
            subplot_xlim[sp_idx] = (-limit, limit)
            subplot_ylim[sp_idx] = (-limit, limit)
        else:
            x_vals = np.concatenate([all_Ex[i].ravel() for i in ds_in_sp])
            y_vals = np.concatenate([all_Ey[i].ravel() for i in ds_in_sp])
            x_max = np.abs(x_vals).max() * 1.2 if x_vals.size else 1.0
            y_max = np.abs(y_vals).max() * 1.2 if y_vals.size else 1.0
            subplot_xlim[sp_idx] = (-x_max, x_max)
            subplot_ylim[sp_idx] = (-y_max, y_max)

    # Making arrow heads, size relative to subplot limits
    def arrow_head(sp_idx):
        xrange = subplot_xlim[sp_idx][1] - subplot_xlim[sp_idx][0]
        yrange = subplot_ylim[sp_idx][1] - subplot_ylim[sp_idx][0]
        return 0.05 * max(xrange, yrange) / 2

    # n_subplots columns, 1 row
    fig, axes_grid = plt.subplots(1, n_subplots, figsize=(figsize[0] * n_subplots, figsize[1]),
                                  squeeze=False)
    axes_grid = axes_grid[0]  
 
    dif_index_arrow = 4
 
    for ds_idx, E in enumerate(datasets):
        sp_idx   = subplot_assignment[ds_idx]
        axis     = axes_grid[sp_idx]
        ds_param = param[ds_idx]
        ds_pname = param_name[ds_idx] if param_name[ds_idx] is not None else E.name
        ds_cmap  = colormap[ds_idx]
 
        # Field stuff
        if E._type == 'Jones_vector':
            E0x, E0y = E.parameters.amplitudes(shape=False)
        else:
            raise ValueError(f"Dataset {ds_idx}: only Jones_vector is supported.")
 
        delay = E.parameters.delay(shape=False)
        phase = E.parameters.global_phase(shape=False)
        if phase is None:
            phase = np.zeros_like(E0x)
        if np.isnan(phase).any():
            phase[np.isnan(phase)] = 0
 
        is_linear = E.checks.is_linear(shape=False, out_number=False)
        is_pol    = np.ones(E.size, dtype=bool) if E.size >= 2 else np.array([True])
 
        Ex = all_Ex[ds_idx]
        Ey = all_Ey[ds_idx]
 
        # Arrows
        if draw_arrow:
            dx = np.diff(Ex)
            dy = np.diff(Ey)
            dif_sq = dx**2 + dy**2
            index_arrow = np.argmax(dif_sq, axis=-1)
            cond = index_arrow >= N_angles - dif_index_arrow
            if np.any(cond):
                index_arrow[cond] -= N_angles
 
        # Color mapping 
        if ds_param is not None:
            param_arr = np.asarray(ds_param).flatten()
            norm      = mcolors.Normalize(vmin=param_arr.min(), vmax=param_arr.max())
            cmap_obj  = cm.get_cmap(ds_cmap)
            sm        = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
 
            def get_color(ind):
                return cmap_obj(norm(param_arr[ind]))
        else:
            def get_color(ind):
                return colors[name_colors[(ind + ds_idx * 10) % 10]]
 
        # Plot dataset ellipses
        for ind in range(E.size):
            if not is_pol[ind]:
                continue
 
            color  = get_color(ind)
            label  = f"{ds_pname}: {ds_param[ind]:.3g}" if ds_param is not None else (
                     E.name if E.size == 1 else f"{E.name}[{ind}]")
 
            axis.plot(Ex[ind, :], Ey[ind, :], color=color, label=label, zorder=2, **kwargs)
 
            if draw_arrow and not is_linear[ind]:
                axis.arrow(
                    Ex[ind, index_arrow[ind]],
                    Ey[ind, index_arrow[ind]],
                    Ex[ind, index_arrow[ind] + dif_index_arrow] - Ex[ind, index_arrow[ind]],
                    Ey[ind, index_arrow[ind] + dif_index_arrow] - Ey[ind, index_arrow[ind]],
                    width=0,
                    head_width=arrow_head(sp_idx),
                    linewidth=0,
                    color=arrow_color if arrow_color is not None else color,
                    length_includes_head=True,
                    zorder=3
                )
 
        # colorbars
        if ds_param is not None:
            cbar = fig.colorbar(sm, ax=axis, orientation='horizontal',
                                pad=0.12, fraction=0.046)
            cbar.set_label(ds_pname, fontsize=11)
 
    # Axis formatting and titles
    for sp_idx, axis in enumerate(axes_grid):
        axis.grid(True)
        axis.set_xlim(*subplot_xlim[sp_idx])
        axis.set_ylim(*subplot_ylim[sp_idx])
 
        if draw_titles:
            names = [datasets[i].name for i in range(n) if subplot_assignment[i] == sp_idx]
            axis.set_title(', '.join(names), fontsize=14)
 
        if draw_labels:
            axis.set_xlabel('$E_x$', fontsize=14)
            axis.set_ylabel('$E_y$', fontsize=14)
        else:
            axis.set_xticklabels([])
            axis.set_yticklabels([])
 
        ds_in_sp = [i for i in range(n) if subplot_assignment[i] == sp_idx]
        if any(param[i] is None for i in ds_in_sp):
            axis.legend(fontsize=8)
 
    plt.tight_layout()
    plt.show()
    return fig, list(axes_grid)
