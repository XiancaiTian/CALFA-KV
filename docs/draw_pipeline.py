#!/usr/bin/env python3
"""Draw the CALFA-KV paper overview as PNG, vector SVG and vector PDF.

Requires matplotlib and numpy. All tensors, rankings and scores in the figure
are schematic, not experimental observations. PNG is exported at 300 dpi.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)
from matplotlib.path import Path as MplPath
import numpy as np

ROOT = Path(__file__).resolve().parent
W, H = 1600, 985
INK = '#273249'
MUTED = '#606D80'
BORDER = '#D9DEE8'
BLUE = '#4C64A0'
BLUE_PALE = '#EDF0F8'
PURPLE = '#8B7BA6'
TEAL = '#3D877D'
TEAL_PALE = '#EDF5F1'
AMBER = '#B3624D'
AMBER_PALE = '#FAEFEA'
GRAY = '#9AA5B7'
GRAY_PALE = '#F1F3F7'
WHITE = '#FFFFFF'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'path',
    'svg.hashsalt': 'calfa-kv-overview',
    'axes.unicode_minus': False,
    'savefig.facecolor': WHITE,
})
fig = plt.figure(figsize=(14.2, 14.2 * H / W), facecolor=WHITE)
ax = fig.add_axes((0, 0, 1, 1))
ax.set(xlim=(0, W), ylim=(H, 0), aspect='equal')
ax.axis('off')
TEXTS = []


def mix(color, white):
    return colors.to_hex(np.array(colors.to_rgb(color)) * (1 - white) + white)


def text(x, y, label, size=10.5, color=INK, weight='normal', ha='center', **kwargs):
    artist = ax.text(x, y, label, fontsize=size, color=color, fontweight=weight,
                     ha=ha, va='center', linespacing=1.35, zorder=8, **kwargs)
    TEXTS.append(artist)
    return artist


def box(x, y, width, height, fill=WHITE, edge=BORDER, radius=7, lw=.8, z=2):
    patch = FancyBboxPatch((x, y), width, height,
                          boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    return patch


def line(points, color=MUTED, lw=1.0, dashed=False, z=3):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='round',
            linestyle=(0, (4, 3)) if dashed else '-', zorder=z)


def arrow(points, color=INK, lw=1.15, dashed=False, head=9, z=4):
    if len(points) > 2:
        line(points[:-1], color, lw, dashed, z)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                                mutation_scale=head, linewidth=lw,
                                linestyle=(0, (4, 3)) if dashed else '-',
                                color=color, shrinkA=0, shrinkB=0, zorder=z))


def curved(points, color=INK, lw=1.1, dashed=False, head=9):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
    patch = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=head,
                            linewidth=lw, linestyle=(0, (4, 3)) if dashed else '-',
                            color=color, zorder=5)
    ax.add_patch(patch)


def dot(x, y, color=INK, radius=2.6):
    ax.add_patch(Circle((x, y), radius, color=color, lw=0, zorder=6))


def matrix(x, y, cols=8, rows=5, cell=9, color=BLUE, seed=0, pair=False):
    rng = np.random.default_rng(seed)
    palette = [BLUE, PURPLE, TEAL, AMBER]
    for r in range(rows):
        for c in range(cols):
            base = palette[(c // 2) % 4] if pair else color
            fill = mix(base, .22 + .52 * rng.random())
            ax.add_patch(Rectangle((x + c * cell, y + r * cell), cell - .65,
                                   cell - .65, facecolor=fill, edgecolor='none',
                                   zorder=4))
    ax.add_patch(Rectangle((x - .8, y - .8), cols * cell, rows * cell,
                           facecolor='none', edgecolor=mix(color, .35),
                           linewidth=.6, zorder=4))


def token_column(x, y, width, height, color, intensity, outline=False):
    for r in range(4):
        ax.add_patch(Rectangle((x, y + r * height / 4), width, height / 4 - .8,
                               facecolor=mix(color, intensity + r * .06),
                               edgecolor='none', zorder=4))
    if outline:
        ax.add_patch(Rectangle((x - .6, y - .6), width + 1.2, height + 1.2,
                               facecolor='none', edgecolor=AMBER,
                               linewidth=1.1, zorder=5))


def chip(x, y, width, label, color=BLUE, size=11.5):
    box(x, y, width, 31, mix(color, .91), mix(color, .65), radius=5, lw=.7)
    text(x + width / 2, y + 15.5, label, size, color)


def section_header(letter, title, y, color, right_note):
    box(43, y - 13, 27, 27, color, color, radius=5, lw=0)
    text(56.5, y + .5, letter, 12, WHITE, 'bold')
    text(84, y + 1, title, 13.1, color, 'bold', ha='left')
    text(1546, y + 1, right_note, 9.1, MUTED, ha='right')


def draw_calibration():
    text(164, 141, 'Calibration states', 11.3, weight='bold')
    # The three overlapping tensors represent the same fixed model state.
    for x, y, color, label, seed in [(77, 186, PURPLE, r'$Q$', 1),
                                    (111, 216, BLUE, r'$K$', 2),
                                    (145, 246, TEAL, r'$V$', 3)]:
        box(x - 7, y - 20, 91, 81, WHITE, mix(color, .5), radius=4)
        text(x + 10, y - 8, label, 12.5, color)
        matrix(x, y + 5, cols=8, rows=5, cell=9, color=color, seed=seed)
    text(164, 352, 'Frozen model weights', 9.7, MUTED)
    text(164, 376, 'Queries, keys and values', 8.8, MUTED)
    arrow([(265, 274), (307, 274)], BLUE)


def draw_coordinates():
    text(496, 141, 'Coordinates & codebooks', 11.3, weight='bold')
    # D: query energy gives nonuniform positive channel multipliers.
    heights = np.array([.35, .72, .48, .96, .57, .4, .82, .58])
    for i, height in enumerate(heights):
        ax.add_patch(Rectangle((333 + i * 11, 227 - height * 45), 7.7,
                               height * 45, facecolor=mix(BLUE, .15 + .06 * (i % 3)),
                               edgecolor='none', zorder=4))
    line([(331, 229), (423, 229)], BORDER, .7)
    text(377, 250, r'$D$', 14, BLUE)
    # N: token magnitudes are separated from normalized vectors.
    for x, radius in zip([448, 473, 498], [4, 10, 6]):
        ax.add_patch(Circle((x, 195), radius, fc=mix(TEAL, .7), ec=TEAL, lw=.6))
        ax.add_patch(Circle((x, 221), 5, fc=mix(TEAL, .45), ec=TEAL, lw=.6))
    text(474, 250, r'$N_t$', 14, TEAL)
    # P: complete low/high RoPE pairs stay adjacent within groups.
    matrix(537, 188, cols=8, rows=4, cell=10, color=PURPLE, seed=5, pair=True)
    text(577, 250, r'$P$', 14, PURPLE)
    text(496, 277, 'Channel scale / token scale / paired groups', 8.8, MUTED)
    line([(377, 291), (377, 302), (575, 302), (575, 291)], mix(BLUE, .6), .8)
    arrow([(476, 302), (476, 319)], BLUE, .9, head=7)
    matrix(361, 324, cols=8, rows=4, cell=8, color=BLUE, seed=9)
    matrix(524, 324, cols=8, rows=4, cell=8, color=TEAL, seed=10)
    text(438, 342, r'$C_K$', 12.5, BLUE)
    text(605, 342, r'$C_V$', 12.5, TEAL)
    text(496, 386, r'$P\,\leftrightarrow\,C_K$; joint K/V initialization', 9.5, MUTED)
    text(496, 410, 'codewords per group', 8.8, MUTED)
    arrow([(656, 274), (705, 274)], BLUE)


def draw_joint_output():
    text(988, 141, 'Output-aware joint calibration', 11.3, weight='bold')
    box(740, 182, 276, 57, GRAY_PALE, BORDER)
    text(878, 210.5, 'Reference attention\n+ output projection', 10)
    box(740, 278, 276, 57, BLUE_PALE, mix(BLUE, .45))
    text(878, 306.5, 'Reconstructed K/V\nattention + projection', 10, BLUE)
    text(727, 210, r'$Q,K,V$', 10.4, MUTED, ha='right')
    text(728, 306, r'$Q,K,V$', 10.4, BLUE, ha='right')
    arrow([(1016, 210), (1064, 210)], MUTED, 1.0)
    arrow([(1016, 306), (1064, 306)], BLUE, 1.0)
    matrix(1069, 191, cols=3, rows=5, cell=7.5, color=GRAY, seed=11)
    matrix(1069, 287, cols=3, rows=5, cell=7.5, color=BLUE, seed=12)
    text(1110, 209, r'$o^{\star}$', 14, MUTED)
    text(1110, 305, r'$\hat o$', 14, BLUE)
    arrow([(1135, 210), (1180, 210), (1180, 242)], MUTED, .9)
    arrow([(1135, 306), (1180, 306), (1180, 277)], BLUE, .9)
    ax.add_patch(Circle((1180, 260), 19, facecolor=WHITE, edgecolor=BLUE,
                        linewidth=1.0, zorder=5))
    text(1180, 260, r'$\mathcal{L}$', 15, BLUE)
    for x, width, label in [(751, 48, r'$\mu$'), (810, 48, r'$A$'),
                             (869, 60, r'$C_K$'), (940, 60, r'$C_V$')]:
        chip(x, 359, width, label, size=13)
    arrow([(876, 359), (876, 337)], BLUE, .9, head=7)
    curved([(1192, 276), (1280, 379), (1092, 406), (1002, 374)], BLUE,
           lw=.95, head=8)
    text(1140, 347, 'joint update', 9.5, BLUE)
    text(988, 414, r'$\mathcal{L}(\mu,A,C_K,C_V)'
         r'=\frac{1}{2E}\sum_w\|\hat o_w-o_w^\star\|_2^2$', 12.8, INK)
    arrow([(1218, 260), (1282, 260)], BLUE)


def draw_frozen_codec():
    text(1419, 141, 'Frozen codec', 11.3, weight='bold')
    box(1315, 186, 204, 167, mix(BLUE, .965), mix(BLUE, .55), radius=8)
    box(1305, 178, 204, 167, mix(BLUE, .94), mix(BLUE, .5), radius=8)
    box(1295, 170, 204, 167, WHITE, mix(BLUE, .35), radius=8)
    text(1397, 207, r'$D,\ P$', 17, BLUE)
    text(1397, 234, 'fixed during calibration', 9.1, MUTED)
    line([(1322, 253), (1472, 253)], BORDER, .8)
    text(1397, 280, r'$\mu,\ A,\ C_K,\ C_V$', 14.7, BLUE)
    text(1397, 307, 'optimized parameters', 9.1, MUTED)
    text(1397, 381, 'Freeze after calibration', 8.7, MUTED)
    # Dashed blue is a parameter handoff; no gradient crosses into FAKV.
    arrow([(1499, 310), (1533, 310), (1533, 500), (340, 500), (340, 615)], BLUE,
          lw=.95, dashed=True, head=8, z=3)
    text(657, 484, 'frozen encoding / decoding parameters', 8.9, BLUE)
    # Fisher belongs to FAKV but is calibrated offline, after the codec is frozen.
    box(817, 450, 350, 37, AMBER_PALE, mix(AMBER, .55), radius=6)
    text(992, 469, r'Final-coordinate Fisher  $F_K,F_V$', 9.6, AMBER)
    arrow([(1533, 450), (1533, 469), (1167, 469)], AMBER, .95,
          dashed=True, head=8)
    # Fisher uses reference-model gradients, not the joint output-loss gradient.
    arrow([(164, 394), (164, 469), (817, 469)], PURPLE, .95, head=8)
    text(456, 452, 'reference-model gradients', 9.4, PURPLE)
    # A small white bridge makes the crossing with the codec rail unambiguous.
    line([(992, 487), (992, 524), (925, 524), (925, 716), (909, 716)], WHITE, 3.7, z=4)
    arrow([(992, 487), (992, 524), (925, 524), (925, 716), (909, 716)], AMBER,
          lw=.95, dashed=True, head=8, z=5)


def draw_cache():
    box(250, 615, 180, 40, BLUE_PALE, mix(BLUE, .5), radius=6)
    text(340, 635, 'Encode K/V', 10.5, BLUE)
    arrow([(340, 655), (340, 703)], BLUE, 1.0, head=8)
    text(222, 687, 'Quantized history', 10.5, weight='bold')
    text(536, 681, 'Recent K/V\nBF16', 9.6, TEAL, 'bold')
    # 20+5 drawn columns are a schematic timeline, not the cache capacity.
    box(85, 703, 508, 64, WHITE, BORDER, radius=4)
    selected = {4, 10, 16}
    for i in range(20):
        token_column(91 + i * 19.8, 709, 16.8, 51, GRAY, .3 + .06 * (i % 4),
                     outline=i in selected)
    for i in range(5):
        token_column(491 + i * 19.3, 709, 16.3, 51, TEAL, .32 + .04 * i)
    line([(482, 704), (482, 767)], BORDER, .8)
    text(283, 745, 'code indices + token scale', 8.8, INK,
         bbox=dict(fc=WHITE, ec='none', alpha=.9, pad=1.3))
    # Current K/V remains exact in Recent; only demoted rows enter the encoder.
    text(585, 625, 'new K/V', 9.1, TEAL)
    arrow([(585, 641), (585, 703)], TEAL, 1.05, head=8)
    # One mixed-history input to attention (drawn behind candidate traffic).
    arrow([(344, 767), (344, 892), (1245, 892), (1245, 816), (1270, 816)],
          MUTED, lw=1.15, head=9, z=2)
    text(830, 914, 'Mixed history: reconstructed K/V + unchanged Recent K/V', 9.1, MUTED)
    # Page-8 demotion preserves raw K/V until candidate scoring has finished.
    raw_route = [(550, 767), (550, 822), (65, 822), (65, 635), (250, 635)]
    line(raw_route, WHITE, 3.5, z=3)
    arrow(raw_route, TEAL, lw=1.0, head=8, z=4)
    text(155, 615, 'prefill / Recent eviction', 8.4, TEAL)
    text(263, 843, 'newly quantized candidates', 8.8, TEAL)
    # Candidates are newly quantized rows, not a second exact-history cache.
    arrow([(445, 767), (445, 789), (640, 789), (640, 736), (689, 736)],
          GRAY, lw=.95, head=8)
    text(595, 777, 'codes', 8.2, MUTED)
    dot(550, 822, TEAL, 2.4)
    arrow([(550, 822), (655, 822), (655, 786), (689, 786)], TEAL, .95, head=8)
    text(606, 844, 'raw K/V', 8.6, TEAL)
    text(304, 878, 'time', 8.5, MUTED, ha='right')
    arrow([(104, 875), (260, 875)], GRAY, .75, head=6)


def draw_scores():
    text(800, 635, 'Fisher-weighted error', 11.0, weight='bold')
    text(800, 662, 'actual reconstruction error', 8.9, MUTED)
    box(689, 701, 220, 97, mix(AMBER, .975), mix(AMBER, .7), radius=6)
    heights = [.25, .72, .39, .95, .51, .3, .84, .44]
    for i, val in enumerate(heights):
        color = AMBER if i in (1, 3, 6) else mix(GRAY, .22)
        ax.add_patch(Rectangle((713 + i * 23, 778 - val * 61), 13.8, val * 61,
                               facecolor=color, edgecolor='none', zorder=5))
    line([(705, 780), (893, 780)], mix(GRAY, .45), .65)
    text(800, 817, r'$s_t=\langle F_K,(e_t^K)^2\rangle'
         r'+\langle F_V,(e_t^V)^2\rangle$', 10.3, INK)
    text(800, 847, 'final codec coordinates', 9.6, MUTED)
    arrow([(909, 750), (948, 750)], AMBER, 1.15, head=9)


def draw_anchors():
    text(1077, 668, 'Retain TOP-N', 11.0, AMBER, 'bold')
    # The loop denotes merge-and-prune of the old bank and new candidates.
    curved([(1183, 722), (1250, 598), (1000, 593), (1000, 699)], AMBER,
           lw=.9, head=8)
    text(1080, 614, 'merge old + new', 8.5, AMBER)
    box(948, 701, 250, 97, AMBER_PALE, mix(AMBER, .45), radius=6)
    text(969, 729, r'$K$', 12.5, AMBER)
    text(969, 773, r'$V$', 12.5, AMBER)
    for i in range(4):
        x = 990 + i * 48
        matrix(x, 713, cols=4, rows=3, cell=7.1, color=AMBER, seed=20 + i)
        matrix(x, 756, cols=4, rows=3, cell=7.1, color=AMBER, seed=30 + i)
        line([(x + 13, 737), (x + 13, 753)], mix(AMBER, .45), .7)
    text(1184, 749, r'$\cdots$', 12, AMBER)
    text(1074, 817, 'paired FP16 K/V', 9.7, AMBER)
    text(1074, 845, 'per request / layer / KV head', 8.4, MUTED)
    arrow([(1198, 750), (1270, 750)], AMBER, 1.15, head=9)
    text(1235, 732, 'replace', 8.3, AMBER)


def draw_attention():
    text(1270, 667, 'Attention with FAKV', 11.0, weight='bold', ha='left')
    ax.add_patch(Circle((1467, 611), 18, facecolor=BLUE_PALE, edgecolor=BLUE,
                        linewidth=.85, zorder=5))
    text(1467, 611, r'$Q$', 13.3, BLUE)
    arrow([(1467, 629), (1467, 692)], BLUE, 1.0, head=8)
    box(1270, 692, 224, 166, WHITE, BORDER, radius=7)
    text(1382, 716, 'Match token positions', 9.4, MUTED)
    box(1290, 738, 184, 34, AMBER_PALE, mix(AMBER, .5), radius=4)
    text(1382, 755, 'Replace selected K/V', 10.3, AMBER)
    text(1382, 795, r'RoPE$(K)$  $\rightarrow$  attention', 10.6)
    line([(1292, 820), (1472, 820)], BORDER, .65)
    text(1382, 840, 'hits bypass scale and center', 9.0, MUTED)
    arrow([(1494, 795), (1534, 795)], INK, 1.15, head=9)
    text(1551, 795, r'$o$', 17.5, INK)
    text(1382, 889, 'one RoPE at each true position', 9.0, MUTED)


def legend():
    text(W / 2, 27, 'CALFA-KV', 18, INK, 'bold')
    arrow([(1038, 28), (1076, 28)], INK, .95, head=7)
    text(1089, 28, 'data / updates', 9.0, MUTED, ha='left')
    arrow([(1263, 28), (1301, 28)], BLUE, .95, dashed=True, head=7)
    text(1314, 28, 'frozen parameters', 9.0, MUTED, ha='left')
    for x, color, label in [(52, GRAY, 'quantized history'),
                            (305, TEAL, 'BF16 Recent K/V'),
                            (551, AMBER, 'FP16 TOP-N K/V')]:
        ax.add_patch(Rectangle((x, 961), 12, 12, fc=mix(color, .3), ec=color, lw=.6))
        text(x + 22, 967, label, 8.6, MUTED, ha='left')


def main():
    box(24, 66, 1552, 374, mix(BLUE, .982), BORDER, radius=10, lw=.85, z=0)
    box(24, 543, 1552, 399, mix(AMBER, .982), BORDER, radius=10, lw=.85, z=0)
    section_header('a', 'Joint optimization', 95, BLUE,
                   'OFFLINE  /  per-layer calibration  /  FAKV disabled')
    section_header('b', 'FAKV', 572, AMBER,
                   'ONLINE  /  retain high-error history positions')
    for x in [288, 682, 1255]:
        line([(x, 161), (x, 417)], mix(BORDER, .2), .7, dashed=True, z=1)
    draw_calibration()
    draw_coordinates()
    draw_joint_output()
    draw_cache()
    draw_scores()
    draw_anchors()
    draw_attention()
    draw_frozen_codec()
    legend()
    # Check the export boundary, then save the same native drawing in all formats.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for artist in TEXTS:
        bbox = artist.get_window_extent(renderer)
        if bbox.x0 < bounds.x0 or bbox.x1 > bounds.x1 or bbox.y0 < bounds.y0 or bbox.y1 > bounds.y1:
            raise RuntimeError('Text leaves canvas: ' + artist.get_text())
    metadata = dict(Title='CALFA-KV', Author='CALFA-KV',
                    Subject='Joint optimization and FAKV TOP-N retention; schematic method diagram')
    fig.savefig(ROOT / 'calfa_kv_pipeline.png', dpi=300, facecolor=WHITE)
    fig.savefig(ROOT / 'calfa_kv_pipeline.pdf', facecolor=WHITE, metadata=metadata)
    svg_path = ROOT / 'calfa_kv_pipeline.svg'
    fig.savefig(svg_path, facecolor=WHITE, metadata={
        'Title': metadata['Title'],
        'Creator': 'CALFA-KV',
        'Description': metadata['Subject'],
        'Date': None,
    })
    # Normalize generated path whitespace for repository checks.
    svg_path.write_text('\n'.join(row.rstrip() for row in
                                  svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)
    print('Created 300-dpi PNG, vector SVG and vector PDF.')


if __name__ == '__main__':
    main()
