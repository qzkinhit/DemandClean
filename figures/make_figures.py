#!/usr/bin/env python
"""Figure 4 (fig_baselines.pdf) and Figure 5 (fig_ablation.pdf) of the paper.

    python figures/build_figure_data.py
    python figures/make_figures.py [--out figures]

Figure 4 has four panels over eight datasets: Tol_rel, record-level REDR, GT usage of the methods
that read GT and detection F1 of DemandClean and eighteen baselines, including FM (Opus 4.8). Only
DemandClean is drawn as a line; a baseline carries its marker alone except on the GT panel, whose
four series are curves. DAGPipe outputs transformed features and has no REDR or detection F1. Figure 5 has (a) a radar of
Tol_rel of DemandClean and of its four ablation variants and (b) the distribution of these values
over the eight datasets, where the black line and the label mark the mean. The inputs are
figures/data/fig_baselines.csv and figures/data/fig_ablation.csv. The axis transforms and
display limits match the paper; the input files retain the untransformed observations.
FM uses zero GT, while API dollars are reported separately in results/llm/llm_cost.json.
Requires matplotlib and numpy.
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.transforms as transforms  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    'font.size': 20, 'axes.titlesize': 26, 'axes.titleweight': 'normal', 'axes.labelsize': 22,
    'axes.labelweight': 'normal', 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 20,
    'figure.dpi': 300, 'axes.linewidth': 2.5, 'xtick.major.width': 2.2, 'ytick.major.width': 2.2,
    'xtick.major.size': 7, 'ytick.major.size': 7, 'xtick.minor.width': 1.2, 'ytick.minor.width': 1.2,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix',
})

DATASETS = ['beers', 'breast_cancer', 'smartfactory', 'bike', 'mercedes', 'nasa', 'soilmoisture', 'har']
TICKS = ['beers', 'breast_c.', 'smartf.', 'bike', 'mercedes', 'nasa', 'soil.', 'har']
METHODS = ['holoclean', 'raha_baran', 'uniclean', 'lopster', 'horizon', 'activeclean', 'comet', 'boostclean',
           'saga', 'diffprep', 'ctxpipe', 'simpleimputer', 'mlimputer', 'llm', 'diana', 'learn2clean',
           'retclean', 'dagpipe', 'DemandClean']
COLORS = {'DemandClean': '#D62728', 'holoclean': '#1F77B4', 'raha_baran': '#2CA02C', 'uniclean': '#FF7F0E',
          'lopster': '#9467BD', 'activeclean': '#17BECF', 'boostclean': '#BCBD22', 'ctxpipe': '#8C564B',
          'simpleimputer': '#7F7F7F', 'horizon': '#E377C2', 'mlimputer': '#393B79', 'comet': '#AD494A',
          'saga': '#637939', 'diffprep': '#E7BA52', 'llm': '#00A087', 'diana': '#6B6ECF',
          'learn2clean': '#8CA252', 'retclean': '#CE6DBD', 'dagpipe': '#9C9EDE'}
MARKERS = {'DemandClean': '*', 'holoclean': 'v', 'raha_baran': 's', 'uniclean': 'D', 'lopster': 'p',
           'activeclean': '^', 'boostclean': 'h', 'ctxpipe': '<', 'simpleimputer': 'o', 'horizon': '>',
           'mlimputer': 'X', 'comet': 'P', 'saga': 'd', 'diffprep': '8', 'llm': 'H', 'diana': r'$\clubsuit$',
           'learn2clean': r'$\spadesuit$', 'retclean': r'$\heartsuit$', 'dagpipe': r'$\diamondsuit$'}
LABELS = {'DemandClean': 'DemandClean', 'holoclean': 'HoloClean', 'raha_baran': 'Raha+Baran',
          'uniclean': 'UniClean', 'lopster': 'Lopster', 'activeclean': 'ActiveClean', 'boostclean': 'BoostClean',
          'ctxpipe': 'CtxPipe', 'simpleimputer': 'SimpleImputer', 'horizon': 'Horizon', 'mlimputer': 'MLImputer',
          'comet': 'Comet', 'saga': 'SAGA', 'diffprep': 'DiffPrep', 'llm': 'FM (Opus 4.8)', 'diana': 'DIANA',
          'learn2clean': 'Learn2Clean', 'retclean': 'RetClean', 'dagpipe': 'DAGPipe'}
VARIANT_LABELS = {'DemandClean': 'DC (Full)', 'DemandClean-NGT': 'DC-NGT', 'DemandClean-ND': 'DC-ND',
                  'DemandClean-Arch': 'DC-Arch', 'DemandClean-Stage': 'DC-Stage'}
VARIANT_COLORS = {'DC (Full)': '#D62728', 'DC-NGT': '#FF7F0E', 'DC-ND': '#1F77B4', 'DC-Arch': '#2CA02C',
                  'DC-Stage': '#9467BD'}
VARIANT_MARKERS = {'DC (Full)': '*', 'DC-NGT': 'D', 'DC-ND': 's', 'DC-Arch': '^', 'DC-Stage': 'o'}


def read_csv(name):
    with open(os.path.join(HERE, 'data', name)) as f:
        return list(csv.DictReader(f))


def box_spines(ax):
    for side in ('top', 'right', 'left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2.5)


def gt_axis(v):
    """Paper GT axis: expanded through 10%, compressed through 50%, then the >50 line."""
    if v <= 10:
        return v * 2.0
    if v <= 30:
        return 20 + (v - 10) * 0.2
    if v <= 50:
        return 24 + (v - 30) * 0.1
    return 27.3


def squash_below(scale, at):
    """Reduce one part of an axis while retaining labels in the original units."""
    def forward(y):
        y = np.asarray(y, dtype=float)
        return np.where(y >= at, y, at + (y - at) * scale)

    def inverse(y):
        y = np.asarray(y, dtype=float)
        return np.where(y >= at, y, at + (y - at) / scale)
    return forward, inverse


def detection_axis(at=0.5, low=0.5, high=1.5):
    def forward(y):
        y = np.asarray(y, dtype=float)
        return np.where(y < at, y * low, at * low + (y - at) * high)

    def inverse(y):
        y = np.asarray(y, dtype=float)
        return np.where(y < at * low, y / low, at + (y - at * low) / high)
    return forward, inverse


def radius(values):
    """Paper radar axis: enlarge the tolerance band from 0.8 to 1.3."""
    values = np.asarray(values, dtype=float)
    return np.where(values < 0, 0.25 * values, np.where(values < 0.8, 0.5 * values,
                    np.where(values < 1.3, 0.4 + 2.0 * (values - 0.8),
                             1.4 + (values - 1.3) / 3.0)))


def figure_baselines(out_dir):
    rows = read_csv('fig_baselines.csv')
    data = {(r['dataset'], r['method']): r for r in rows}
    expected = {(ds, method) for ds in DATASETS for method in METHODS}
    if len(rows) != len(expected) or set(data) != expected:
        raise ValueError('Figure 4 needs exactly one observation per dataset and method. Rebuild its CSV input')
    panels = [(r'(a) Relative Tolerance $\mathrm{Tol}_{\mathrm{rel}}$', 'tol_rel', (-1.3, 2.5)),
              ('(b) Record-Level Error Drop (REDR)', 'redr', (-2.4, 1.08)),
              ('(c) GT Usage of GT-Using Methods (%)', 'gt_pct', None),
              ('(d) Cell-Level Detection F1', 'det_f1', (-0.05, 1.08))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()
    x = np.arange(len(DATASETS))
    for pidx, (title, key, ylim) in enumerate(panels):
        ax = axes[pidx]
        box_spines(ax)
        for midx, m in enumerate(METHODS):
            have = [(i, ds) for i, ds in enumerate(DATASETS) if (ds, m) in data and data[(ds, m)][key] != '']
            vx = [i for i, _ in have]
            vy = [float(data[(ds, m)][key]) for _, ds in have]
            if not vy:
                continue
            if key == 'gt_pct':
                if m != 'DemandClean' and not any(v > 0 for v in vy):
                    continue
                vy = [gt_axis(v) for v in vy]
            elif key == 'tol_rel':
                vy = [max(-1.0, min(v, 3.0)) for v in vy]
            elif key == 'redr':
                vy = [max(-2.35, min(v, 1.0)) for v in vy]
            dc = m == 'DemandClean'
            if ylim and dc:
                margin = (ylim[1] - ylim[0] * (0.3 if key == 'redr' else 1.0)) * 0.04
                vy = [max(min(v, ylim[1] - margin), ylim[0] + margin) for v in vy]
            jitter = (midx - len(METHODS) / 2) * 0.04
            ax.plot([xi + jitter for xi in vx], vy, marker=MARKERS[m], markersize=22 if dc else 13,
                    linewidth=4.5 if dc else 2.8, color=COLORS[m], alpha=1.0 if dc else 0.85,
                    linestyle='-' if dc or key == 'gt_pct' else 'none',
                    zorder=10 if dc else 3, markeredgecolor='#222222', markeredgewidth=2.0 if dc else 1.2,
                    solid_capstyle='round')
        if key == 'gt_pct':
            real = [0, 2, 4, 6, 8, 10, 30, 50]
            mapped = [gt_axis(v) for v in real]
            mapped[-1] = 27.3
            ax.set_yticks(mapped)
            ax.set_yticklabels([f'{v:.0f}' if v < 50 else f'>{v:.0f}' for v in real])
            ax.set_ylim(-1, 28.3)
            for b in (gt_axis(10), gt_axis(30)):
                ax.axhline(y=b, color='#999999', linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)
        if key == 'tol_rel':
            ax.axhspan(-10, 0, facecolor='#1F77B4', alpha=0.16, zorder=0)
            ax.axhspan(1, 10, facecolor='#D62728', alpha=0.07, zorder=0)
            ax.axhline(y=0, color='#1F77B4', linewidth=2.6, linestyle=(0, (6, 3)), alpha=0.85, zorder=2)
            ax.axhline(y=1, color='#D62728', linewidth=2.6, linestyle=(0, (6, 3)), alpha=0.85, zorder=2)
            box = dict(boxstyle='round,pad=0.14', facecolor='white', linewidth=1.2, alpha=0.95)
            ax.text(0.985, 1.22, r'RepairAll ($D_{\mathit{clean}}$)', fontsize=13, color='#D62728',
                    transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    va='bottom', ha='right', bbox=dict(box, edgecolor='#D62728', alpha=0.72), zorder=20)
            nofix = ax.text(1.07, 0.035, r'NoFix ($D_{\mathit{dirty}}$)', fontsize=12, color='#1F77B4',
                            transform=ax.transAxes, va='center', ha='right', clip_on=False, zorder=20,
                            bbox=dict(box, edgecolor='#1F77B4', alpha=0.72))
            nofix.set_in_layout(False)
            ax.set_yscale('function', functions=squash_below(0.3, 0.0))
            ax.set_yticks([-1.0, 0.0, 1.0, 2.0])
            ax.set_yticklabels(['<-1', '0', '1', '2'])
        if key == 'redr':
            ax.set_yscale('function', functions=squash_below(0.3, 0.0))
            ax.set_yticks([-2.0, -1.0, 0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['-2', '-1', '0', '0.25', '0.5', '0.75', '1'])
        if key == 'det_f1':
            ax.set_yscale('function', functions=detection_axis())
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
            ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '0.9', '1'])
        for b in x[:-1] + 0.5:
            ax.axvline(x=b, color='#999999', linewidth=1.2, linestyle=(0, (3, 3)), alpha=0.6, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(TICKS, rotation=40, ha='right', fontsize=18)
        ax.set_xlabel(title, fontsize=22, labelpad=-7,
                      fontfamily='Times New Roman', fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linewidth=0.6, linestyle='--', color='#888888')
        ax.set_axisbelow(True)
        if ylim:
            ax.set_ylim(ylim)
        else:
            lo, hi = ax.get_ylim()
            pad = (hi - lo) * 0.08
            ax.set_ylim(lo - pad, hi + pad)

    def handle(m):
        dc = m == 'DemandClean'
        return Line2D([0], [0], marker=MARKERS[m], color=COLORS[m], linewidth=3.5 if dc else 0,
                      linestyle='-' if dc else 'none',
                      markersize=16 if dc else 11, markeredgecolor='#222222', markeredgewidth=1.8 if dc else 0.8,
                      label=LABELS[m] + ('      ' if dc else ''))
    blank = Line2D([0], [0], color='none', marker='', linestyle='', label=' ')
    baselines = [m for m in METHODS if m != 'DemandClean']
    handles = [handle('DemandClean'), blank, blank, blank]
    for i in range(0, len(baselines), 4):
        chunk = baselines[i:i + 4]
        handles += [handle(m) for m in chunk] + [blank] * (4 - len(chunk))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles) // 4, fontsize=19, frameon=True,
               framealpha=0.95, edgecolor='#AAAAAA', fancybox=True, shadow=False, bbox_to_anchor=(0.5, 0.118),
               handlelength=0.9, handleheight=0.9, columnspacing=0.18, handletextpad=0.14, borderpad=0.28,
               labelspacing=0.28)
    plt.tight_layout(rect=[0, 0.235, 1, 1], h_pad=0.6, w_pad=0.5)
    path = os.path.join(out_dir, 'fig_baselines.pdf')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()
    return path


def figure_ablation(out_dir):
    rows = read_csv('fig_ablation.csv')
    expected = {(ds, variant) for ds in DATASETS for variant in VARIANT_LABELS}
    if len(rows) != len(expected) or {(r['dataset'], r['variant']) for r in rows} != expected:
        raise ValueError('Figure 5 needs exactly one observation per dataset and configuration. Rebuild its CSV input')
    names = [VARIANT_LABELS[v] for v in VARIANT_LABELS]
    tol = {n: [0.0] * len(DATASETS) for n in names}
    for r in rows:
        tol[VARIANT_LABELS[r['variant']]][DATASETS.index(r['dataset'])] = float(r['tol_rel'])
    fig = plt.figure(figsize=(17, 8.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.4, 0.75], wspace=-0.05, left=0.01, right=0.98, bottom=0.14,
                           top=0.84)
    ax1 = fig.add_subplot(gs[0], polar=True)
    ax1.set_theta_offset(np.pi / 4)
    angles = np.linspace(0, 2 * np.pi, len(DATASETS), endpoint=False).tolist() + [0]
    for n in names:
        vals = radius(tol[n] + [tol[n][0]])
        full = n == 'DC (Full)'
        ax1.plot(angles, vals, marker=VARIANT_MARKERS[n], markersize=20 if full else 14,
                 linewidth=4.5 if full else 2.8, label=n, color=VARIANT_COLORS[n], markeredgecolor='#222222',
                 markeredgewidth=2.0 if full else 1.2, zorder=10 if full else 3)
        ax1.fill(angles, vals, alpha=0.15 if full else 0.03, color=VARIANT_COLORS[n])
    ax1.set_theta_offset(np.radians(90 - 360.0 / len(DATASETS) * 1.5))
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(TICKS, fontsize=30)
    tick_box = dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=1.0)
    for label, angle in zip(ax1.get_xticklabels(), angles[:-1]):
        label.set_bbox(dict(tick_box))
        label.set_zorder(10)
        if label.get_text() == 'mercedes':
            label.set_transform(transforms.offset_copy(label.get_transform(), fig=fig,
                                                       x=-22, units='points'))
        elif label.get_text() == 'beers':
            label.set_verticalalignment('top')
            label.set_y(label.get_position()[1] + 0.01)
        elif label.get_text() == 'smartf.':
            label.set_visible(False)
            ax1.text(angle, float(radius(2.5)) * 1.015, label.get_text(), fontsize=30,
                     ha='right', va='bottom', bbox=dict(tick_box), zorder=10, clip_on=False)
        elif label.get_text() == 'breast_c.':
            label.set_visible(False)
            ax1.text(angle, float(radius(2.5)) * 1.015, label.get_text(), fontsize=30,
                     ha='left', va='bottom', bbox=dict(tick_box), zorder=10, clip_on=False)
    ax1.set_ylim(float(radius(-1.0)), float(radius(2.5)))
    ax1.tick_params(axis='x', pad=12)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.set_yticks([float(radius(v)) for v in [0.0, 0.8, 1.0, 1.3]])
    ax1.set_yticklabels(['0', '0.8', '1.0', '1.3'])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(theta, [float(radius(1.0))] * len(theta), color='#D62728', linewidth=2.6,
             linestyle=(0, (6, 3)), alpha=0.9, zorder=2)
    for sp in ax1.spines.values():
        sp.set_zorder(1)
    ax1.set_axisbelow(True)
    for label in ax1.get_yticklabels():
        label.set_zorder(10)
    ax1.grid(linewidth=1.2, alpha=0.4)

    ax2 = fig.add_subplot(gs[1])
    box_spines(ax2)
    data = [tol[n] for n in names]
    bp = ax2.boxplot(data, patch_artist=True, widths=0.55, whis=[0, 100], showmeans=True, meanline=True,
                     meanprops=dict(color='black', linewidth=3.5, linestyle='-', zorder=5),
                     medianprops=dict(linewidth=0), whiskerprops=dict(linewidth=2.2, color='#222222'),
                     capprops=dict(linewidth=2.8, color='#222222', zorder=6), showfliers=False)
    for patch, n in zip(bp['boxes'], names):
        patch.set_facecolor(VARIANT_COLORS[n])
        patch.set_edgecolor('#222222')
        patch.set_linewidth(2.2)
        patch.set_alpha(0.7)
    for i, d in enumerate(data):
        avg = float(np.mean(d))
        ax2.text(i + 1, avg - 0.04, f'{avg:.2f}', ha='center', va='top', fontsize=20, zorder=6)
    ax2.set_xticks(range(1, len(names) + 1))
    ax2.set_xticklabels(names, fontsize=22, rotation=25, ha='right')
    ax2.set_ylabel(r'$\mathrm{Tol}_{\mathrm{rel}}$', fontsize=24)
    lo, hi = ax2.get_ylim()
    ax2.axhspan(lo, 0.0, facecolor='#1F77B4', alpha=0.16, zorder=0)
    ax2.axhspan(1.0, hi, facecolor='#D62728', alpha=0.07, zorder=0)
    ax2.axhline(y=1.0, color='#D62728', linewidth=3.0, linestyle=(0, (6, 3)), alpha=0.9, zorder=2)
    ax2.axhline(y=0.0, color='#1F77B4', linewidth=3.0, linestyle=(0, (6, 3)), alpha=0.9, zorder=2)
    ax2.set_ylim(lo, hi)
    note_box = dict(boxstyle='round,pad=0.16', facecolor='white', linewidth=1.6, alpha=0.72)
    ax2.text(2.9, 1.17, r'RepairAll ($D_{\mathit{clean}}$)', fontsize=18,
             color='#D62728', ha='center', va='bottom',
             bbox=dict(note_box, edgecolor='#D62728'), zorder=8)
    ax2.text(4, -0.15, r'NoFix ($D_{\mathit{dirty}}$)', fontsize=18, color='#1F77B4',
             ha='center', va='top', bbox=dict(note_box, edgecolor='#1F77B4'), zorder=8)
    ax2.tick_params(axis='y', labelsize=20)
    for label in ax2.get_yticklabels():
        label.set_bbox(dict(tick_box))
    ax2.set_xlim(0.3, len(names) + 0.5)
    ax2.grid(axis='y', alpha=0.3, linewidth=0.6, linestyle='--')
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=22, loc='upper center', bbox_to_anchor=(0.58, 0.975), ncol=5,
               frameon=True, framealpha=0.95, edgecolor='#AAAAAA', fancybox=True, columnspacing=1.0,
               handletextpad=0.4, handlelength=2.2, handleheight=1.5, borderpad=0.6, markerscale=1.3)
    font = {'family': 'Times New Roman', 'fontsize': 35, 'fontweight': 'bold'}
    fig.text(0.33, -0.01, '(a) Tolerance per Dataset', ha='center', **font)
    fig.text(0.80, -0.01, '(b) Tolerance Distribution', ha='center', **font)
    path = os.path.join(out_dir, 'fig_ablation.pdf')
    fig.savefig(path, bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close()
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=HERE, help='output directory of the PDF files')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for path in (figure_baselines(a.out), figure_ablation(a.out)):
        print(f'{path}: {os.path.getsize(path) / 1024:.1f} KB')


if __name__ == '__main__':
    main()
