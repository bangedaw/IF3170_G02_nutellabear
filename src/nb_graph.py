import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import ArrowStyle, FancyArrowPatch


def feature_class_mi(model):
    prior = np.exp(model.log_prior_)
    rows = {}
    for column in model.columns_:
        conditional = np.exp(model.log_lik_[column])
        marginal = prior @ conditional
        rows[column] = float((prior[:, None] * conditional * np.log2(conditional / marginal)).sum())
    return pd.Series(rows, name='mutual_information_bits').sort_values(ascending=False)


def plot_nb_graph(model, path, class_name='is_canceled'):
    mi = feature_class_mi(model)
    n = len(mi)
    angles = np.pi / 2 - 2 * np.pi * np.arange(n) / n
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    strength = (mi / mi.max()).to_numpy()
    binned = np.array([model.edges_[column] is not None for column in mi.index])

    fig, ax = plt.subplots(figsize=(11, 11))
    for (x, y), s in zip(points, strength):
        tail = 0.12 + 1.1 * s
        style = ArrowStyle.Simple(head_length=1.2 + tail, head_width=1.2 + 2 * tail, tail_width=tail)
        ax.add_patch(FancyArrowPatch((0, 0), (x, y), arrowstyle=style, mutation_scale=9,
                                     shrinkA=26, shrinkB=22, color='#4c78a8', alpha=0.35 + 0.55 * s,
                                     linewidth=0))
    ax.scatter(*points[binned].T, s=900, marker='s', color='#f58518', zorder=3)
    ax.scatter(*points[~binned].T, s=900, marker='o', color='#54a24b', zorder=3)
    ax.scatter([0], [0], s=3200, marker='o', color='#e45756', zorder=3)
    ax.text(0, 0, class_name, ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)
    for (x, y), name, value in zip(points, mi.index, mi.to_numpy()):
        side = 'left' if x > 0.15 else 'right' if x < -0.15 else 'center'
        ax.text(1.13 * x, 1.13 * y, f'{name}\nMI = {value:.3f}', ha=side,
                va='bottom' if y > 0.9 else 'top' if y < -0.9 else 'center', fontsize=9)
    ax.legend(handles=[
        Line2D([], [], marker='o', linestyle='', color='#e45756', markersize=12, label='class'),
        Line2D([], [], marker='o', linestyle='', color='#54a24b', markersize=10, label='categorical feature'),
        Line2D([], [], marker='s', linestyle='', color='#f58518', markersize=10, label='binned numeric feature'),
    ], loc='lower left', fontsize=9)
    ax.set_title('Naive Bayes structure: arrow width = mutual information between feature and class (bits)\n'
                 'no edges between features (independence given the class)', fontsize=11)
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig, mi
