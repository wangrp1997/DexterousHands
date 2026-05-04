#!/usr/bin/env python3

import argparse
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from tools import csv2numpy, find_all_files, group_files


def smooth(y, radius, mode='two_sided', valid_only=False):
    '''Smooth signal y, where radius is determines the size of the window.

    mode='twosided':
        average over the window [max(index - radius, 0), min(index + radius, len(y)-1)]
    mode='causal':
        average over the window [max(index - radius, 0), index]
    valid_only: put nan in entries where the full-sized window is not available
    '''
    assert mode in ('two_sided', 'causal')
    if len(y) < 2 * radius + 1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius + 1)
        out = np.convolve(y, convkernel, mode='same') / \
            np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel, mode='full') / \
            np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius + 1]
        if valid_only:
            out[:radius] = np.nan
    return out


def smooth_window_ymin_ymax(y, radius):
    """Per-index min/max of raw y over the same index window as smooth() (two_sided)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    lo = np.empty(n, dtype=float)
    hi = np.empty(n, dtype=float)
    if radius <= 0:
        lo[:] = hi[:] = y
        return lo, hi
    if n < 2 * radius + 1:
        lo[:] = np.min(y)
        hi[:] = np.max(y)
        return lo, hi
    for i in range(n):
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        lo[i] = np.min(y[a:b])
        hi[i] = np.max(y[a:b])
    return lo, hi


COLORS = (
    [
        # deepmind style
        '#0072B2',
        '#009E73',
        '#D55E00',
        '#CC79A7',
        # '#F0E442',
        '#d73027',  # RED
        # built-in color
        'blue',
        'red',
        'pink',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'brown',
        'orange',
        'teal',
        'lightblue',
        'lime',
        'lavender',
        'turquoise',
        'darkgreen',
        'tan',
        'salmon',
        'gold',
        'darkred',
        'darkblue',
        'green',
        # personal color
        '#313695',  # DARK BLUE
        '#74add1',  # LIGHT BLUE
        '#f46d43',  # ORANGE
        '#4daf4a',  # GREEN
        '#984ea3',  # PURPLE
        '#f781bf',  # PINK
        '#ffc832',  # YELLOW
        '#000000',  # BLACK
    ]
)


def plot_ax(
    ax,
    file_lists,
    legend_pattern=".*",
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    xkey='env_step',
    ykey='rew',
    smooth_radius=0,
    shaded_std=True,
    legend_outside=False,
):

    def legend_fn(x):
        # return os.path.split(os.path.join(
        #     args.root_dir, x))[0].replace('/', '_') + " (10)"
        return re.search(legend_pattern, x).group(0)

    legneds = map(legend_fn, file_lists)
    # sort filelist according to legends
    file_lists = [f for _, f in sorted(zip(legneds, file_lists))]
    legneds = list(map(legend_fn, file_lists))

    for index, csv_file in enumerate(file_lists):
        csv_dict = csv2numpy(csv_file)
        x = csv_dict[xkey]
        y_raw = np.asarray(csv_dict[ykey], dtype=float)
        y_line = smooth(y_raw, radius=smooth_radius)
        color = COLORS[index % len(COLORS)]
        # label= ties legend colors to this Line2D; legend(list_of_str) uses default colors.
        ax.plot(x, y_line, color=color, label=legneds[index])
        if not shaded_std:
            continue
        # Multi-seed: tools.py rew:shaded (cross-seed std). Single-seed + --smooth: fill
        # between rolling min/max of raw rew in the same window as smooth() — band width
        # is exactly the spread of original points in that window (not std).
        used_csv_shaded = False
        if ykey + ":shaded" in csv_dict:
            y_csv = np.asarray(csv_dict[ykey + ":shaded"], dtype=float)
            if np.nanmax(np.abs(y_csv)) > 1e-12:
                y_band = smooth(y_csv, radius=smooth_radius)
                ax.fill_between(x, y_line - y_band, y_line + y_band, color=color, alpha=0.2)
                used_csv_shaded = True
        if not used_csv_shaded and smooth_radius > 0:
            y_lo, y_hi = smooth_window_ymin_ymax(y_raw, smooth_radius)
            ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.2)

    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    else:
        ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mticker.EngFormatter())
    if xlim is not None:
        ax.set_xlim(xmin=0, xmax=xlim)
    # add title
    ax.set_title(title)
    # add labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_figure(
    file_lists,
    group_pattern=None,
    fig_length=6,
    fig_width=6,
    sharex=False,
    sharey=False,
    title=None,
    **kwargs,
):
    if not group_pattern:
        fig, ax = plt.subplots(figsize=(fig_length, fig_width))
        plot_ax(ax, file_lists, title=title, **kwargs)
    else:
        res = group_files(file_lists, group_pattern)
        row_n = int(np.ceil(len(res) / 3))
        col_n = min(len(res), 3)
        fig, axes = plt.subplots(
            row_n,
            col_n,
            sharex=sharex,
            sharey=sharey,
            figsize=(fig_length * col_n, fig_width * row_n),
            squeeze=False
        )
        axes = axes.flatten()
        for i, (k, v) in enumerate(res.items()):
            plot_ax(axes[i], v, title=k, **kwargs)
    if title:  # add title
        fig.suptitle(title, fontsize=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument(
        '--fig-length',
        type=int,
        default=6,
        help='matplotlib figure length (default: 6)'
    )
    parser.add_argument(
        '--fig-width',
        type=int,
        default=6,
        help='matplotlib figure width (default: 6)'
    )
    parser.add_argument(
        '--style',
        default='seaborn',
        help='matplotlib figure style (default: seaborn)'
    )
    parser.add_argument(
        '--title', default=None, help='matplotlib figure title (default: None)'
    )
    parser.add_argument(
        '--xkey',
        default='env_step',
        help='x-axis key in csv file (default: env_step)'
    )
    parser.add_argument(
        '--ykey', default='rew', help='y-axis key in csv file (default: rew)'
    )
    parser.add_argument(
        '--smooth', type=int, default=0, help='smooth radius of y axis (default: 0)'
    )
    parser.add_argument(
        '--xlabel', default='Timesteps', help='matplotlib figure xlabel'
    )
    parser.add_argument(
        '--ylabel', default='Episode Reward', help='matplotlib figure ylabel'
    )
    parser.add_argument(
        '--shaded-std',
        action='store_true',
        help='Multi-seed: use CSV rew:shaded (cross-seed std). Single-seed with --smooth>0: '
        'shade min..max of raw rew in the same sliding window as smooth (data spread).'
    )
    parser.add_argument(
        '--sharex',
        action='store_true',
        help='whether to share x axis within multiple sub-figures'
    )
    parser.add_argument(
        '--sharey',
        action='store_true',
        help='whether to share y axis within multiple sub-figures'
    )
    parser.add_argument(
        '--legend-outside',
        action='store_true',
        help='place the legend outside of the figure'
    )
    parser.add_argument(
        '--xlim', type=int, default=None, help='x-axis limitation (default: None)'
    )
    parser.add_argument('--root-dir', default='./', help='root dir (default: ./)')
    parser.add_argument(
        '--file-pattern',
        type=str,
        default=r".*/test_rew_\d+seeds.csv$",
        help='regular expression to determine whether or not to include target csv '
        'file, default to including all test_rew_{num}seeds.csv file under rootdir'
    )
    parser.add_argument(
        '--group-pattern',
        type=str,
        default=r"(/|^)\w*?\-v(\d|$)",
        help='regular expression to group files in sub-figure, default to grouping '
        'according to env_name dir, "" means no grouping'
    )
    parser.add_argument(
        '--legend-pattern',
        type=str,
        default=r".*",
        help='regular expression to extract legend from csv file path, default to '
        'using file path as legend name.'
    )
    parser.add_argument('--show', action='store_true', help='show figure')
    parser.add_argument(
        '--output-path', type=str, help='figure save path', default="./figure.png"
    )
    parser.add_argument(
        '--dpi', type=int, default=200, help='figure dpi (default: 200)'
    )
    args = parser.parse_args()
    cwd_start = os.getcwd()
    # Resolve before chdir: plot_figure ends with chdir to root_dir, and a fragile "../../"
    # would break paths like ./bidexhands/logs/.../figure.png (duplicate bidexhands).
    output_abs = os.path.abspath(os.path.join(cwd_start, args.output_path))
    os.makedirs(os.path.dirname(output_abs) or ".", exist_ok=True)

    file_lists = find_all_files(args.root_dir, re.compile(args.file_pattern))
    file_lists = [os.path.relpath(f, args.root_dir) for f in file_lists]
    if args.style:
        plt.style.use(args.style)
    os.chdir(args.root_dir)
    plot_figure(
        file_lists,
        group_pattern=args.group_pattern,
        legend_pattern=args.legend_pattern,
        fig_length=args.fig_length,
        fig_width=args.fig_width,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        xkey=args.xkey,
        ykey=args.ykey,
        xlim=args.xlim,
        sharex=args.sharex,
        sharey=args.sharey,
        smooth_radius=args.smooth,
        shaded_std=args.shaded_std,
        legend_outside=args.legend_outside,
    )

    os.chdir(cwd_start)
    if output_abs:
        plt.savefig(output_abs, dpi=args.dpi, bbox_inches='tight')
    if args.show:
        plt.show()
