import os
from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
import io
from PIL import Image
import numpy as np
import pygraphviz as pgv

import catppuccin
FLAVOR = catppuccin.PALETTE.frappe
COLOR_CYCLE = [
    FLAVOR.colors.blue,
    FLAVOR.colors.red,
    FLAVOR.colors.green,
    FLAVOR.colors.pink,
    FLAVOR.colors.sky,
    FLAVOR.colors.text,
    FLAVOR.colors.mauve,
    FLAVOR.colors.flamingo,
    FLAVOR.colors.peach,
]
HATCHES = ['']*len(COLOR_CYCLE) + ['///']*len(COLOR_CYCLE)

from catppuccin.extras.matplotlib import get_colormap_from_list
backend = os.environ.get('MPLBACKEND', 'GTK3Agg')
mpl.use(backend)
mpl.style.use(FLAVOR.identifier)
plt.rcParams['axes.prop_cycle'] = cycler('color', map(lambda x: x.hex, COLOR_CYCLE))

from vtypes import GameData

def rotate_positions_only(G: pgv.AGraph, prog: str = 'dot'):
    # 1) Compute positions in‑process
    # G.layout(prog=prog)

    # 2) Swap X/Y for every node
    for n in G.nodes():
        pos = G.get_node(n).attr.get('pos')
        if not pos: 
            continue
        x_str, y_str = pos.split(',')
        # swap coords
        G.get_node(n).attr['pos'] = f"{y_str},{x_str}"

    # 3) Write out using no‑layout to freeze your swapped positions
    #    For 'dot' you can use prog='dot -n2'; for others similarly.
    return G

def auto_rotate_agraph(G: pgv.AGraph, layout_prog: str = 'dot') -> pgv.AGraph:
    """
    Given a PyGraphviz AGraph, runs layout in-process to compute
    bounding box, checks if height > width, and if so, adjusts
    the graph attributes to rotate (or change rankdir) and returns
    the modified graph.

    Parameters
    ----------
    G : pgv.AGraph
        The input directed or undirected graph.
    layout_prog : str
        Graphviz layout engine to use (e.g., 'dot', 'neato').
    nodesep : float
        Minimum separation between nodes (in inches) when rotating.
    ranksep : float
        Minimum separation between ranks (in inches) when rotating.

    Returns
    -------
    pgv.AGraph
        The potentially modified graph with updated attributes.
    """
    # 1. Compute initial layout in-process
    G.layout(prog=layout_prog)

    # 2. Extract all node positions
    xs, ys = [], []
    for node in G.nodes():
        pos = G.get_node(node).attr.get('pos')
        if pos:
            x_str, y_str = pos.split(',')
            xs.append(float(x_str))
            ys.append(float(y_str))
    # If no positions or too few nodes, return as-is
    if not xs or not ys:
        return G

    # 3. Compute bounding box
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    # 4. Heuristic: rotate if taller than wide
    if height > width:
        rotate_positions_only(G, layout_prog)

    return G

# Plot heatmap
def plot_vote_heatmap(matrix, names):
    # Compute column sums
    column_sums = matrix.sum(axis=0)

    # Create a new matrix for the heatmap without the totals row
    matrix_without_totals = matrix

    # Create the heatmap without the totals, using a custom color scale
    plt.figure(figsize=(10, 8))
    cmap = get_colormap_from_list(
        FLAVOR.identifier,
        ["yellow", "peach", "red"],
    )
    ax = sns.heatmap(matrix_without_totals, annot=True, fmt='d', cmap=cmap,
                     xticklabels=names, yticklabels=names,
                     cbar_kws={'label': 'Votes'},
                     vmin=matrix_without_totals.min(), vmax=matrix_without_totals.max())  # Fix color scale

    # Plot the totals row separately by adding a text label on the heatmap
    # We will add a horizontal line and annotate the total row on top
    for i, col_sum in enumerate(column_sums):
        ax.text(i + 0.5, len(matrix), f'{col_sum}', ha='center', va='center',
                fontsize=12, color='black', fontweight='bold', backgroundcolor='white')

    # Set labels and title
    ax.set_xlabel("Target of Vote")
    ax.set_ylabel("Voter")
    ax.set_title("Voting Heatmap (Cumulative)")

    # Rotate x and y ticks for better readability
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust layout to ensure everything fits
    plt.tight_layout()

    return plt

def draw_vote_flow_graph(game_data: GameData, G:nx.Graph, immune: set[str], winners: set[str]):
    A = to_agraph(G)

    traitors = set(game_data.traitors)

    # Select Catppuccin palette
    color_background = FLAVOR.colors.base.hex     # Base color for backgrounds
    color_node       = FLAVOR.colors.surface0.hex # Mantle for nodes
    color_traitor    = FLAVOR.colors.red.hex      # For traitors or highlights
    color_text       = FLAVOR.colors.text.hex     # Text color for labels

    # Configure node appearance
    for node in A.nodes():
        name = node.get_name()
        label:str = str(name)
        if name in immune:
            label += " [S]"

        node.attr['label'] = label
        node.attr['style'] = 'filled'

        # Color nodes based on their status
        if name in traitors:
            node.attr['fillcolor'] = color_traitor
            node.attr['fontcolor'] = color_background
        else:
            node.attr['fillcolor'] = color_node
            node.attr['fontcolor'] = color_text

        node.attr['shape'] = 'ellipse'
        node.attr['width'] = '1.75'
        node.attr['height'] = '1'
        node.attr['fixedsize'] = 'true'
        node.attr['fontname'] = 'Helvetica'
        node.attr['fontsize'] = '26'

        if name in winners:
            node.attr['color']    = FLAVOR.colors.yellow.hex
            node.attr['penwidth'] = '8'

    A.graph_attr.update({
        'nodesep': '+1',   # horizontal space between nodes
        'ranksep': '-1',   # vertical space between ranks (for dot)
        'margin': '0',     # margin around the entire graph
        'overlap': 'false',
        'sep': '-20',
        'dpi': '150',
        'bgcolor': color_background,  # Background color of the entire graph
    })

    # Configure graph layout
    A.layout(prog='fdp')  # Or try 'neato', 'fdp', 'sfdp'

    # Render to memory and display
    img_buf = io.BytesIO()
    A.draw(img_buf, format='png')
    img_buf.seek(0)

    img = Image.open(img_buf)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    return plt

def draw_vote_flow_cum_rounds_weighted(game_data:GameData, G:nx.Graph, vote_counts):
    A = to_agraph(G)

    traitors = set(game_data.traitors)

    # Select Catppuccin palette
    color_background = FLAVOR.colors.base.hex     # Base color for backgrounds
    color_node       = FLAVOR.colors.surface0.hex # Mantle for nodes
    color_traitor    = FLAVOR.colors.red.hex      # For traitors or highlights
    color_text       = FLAVOR.colors.text.hex     # Text color for labels

    # Style nodes
    for node in A.nodes():
        name = node.get_name()
        label = name

        node.attr['label'] = label
        node.attr['style'] = 'filled'

        # Color nodes based on their status
        if name in traitors:
            node.attr['fillcolor'] = color_traitor
            node.attr['fontcolor'] = color_background
        else:
            node.attr['fillcolor'] = color_node
            node.attr['fontcolor'] = color_text

        node.attr['shape'] = 'ellipse'
        node.attr['width'] = '1.75'
        node.attr['height'] = '1'
        node.attr['fixedsize'] = 'true'
        node.attr['fontname'] = 'Helvetica'
        node.attr['fontsize'] = '26'

    # Normalize vote counts to [0, 1] for color mapping
    max_votes = max(vote_counts.values())
    norm = Normalize(vmin=1, vmax=max_votes)
    cmap = get_colormap_from_list(
        FLAVOR.identifier,
        ["yellow", "peach", "red"],
    )
    for (voter, target), count in vote_counts.items():
        edge = A.get_edge(voter, target)
        rgba = cmap(norm(count))
        # Convert RGBA to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
        edge.attr.update({
            'label': str(count),
            'fontsize': '15',
            'fontcolor': FLAVOR.colors.text.hex,
            'penwidth': str(1.5 + count * 1.5),
            'color': hex_color,
        })

    A.graph_attr.update({
        'nodesep': '+1',   # horizontal space between nodes
        'ranksep': '-1',   # vertical space between ranks (for dot)
        'margin': '0',
        'overlap': 'false',
        'sep': '0.5',
        'dpi': '150',
        'bgcolor': color_background,  # Background color of the entire graph
    })

    A.layout(prog='fdp')

    # Render and show
    img_buf = io.BytesIO()
    A.draw(img_buf, format='png')
    img_buf.seek(0)

    img = Image.open(img_buf)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Vote Flow Diagram – All Rounds (Color & Width = Vote Frequency)", fontsize=14)
    return plt

def plot_all_clusterings(game_data, G_base:nx.Graph, methods: list):
    """
    Plot a 2x2 grid of community‐detection results on the vote‐flow graph.

    Parameters
    ----------
    game_data : GameData
        Your loaded game data structure.
    until_round : None|int
        Data until which round should be considered
    """
    # 1) Build base vote‐flow graph
    traitors = set(game_data.traitors)


    # 2) Prepare methods

    # 3) Subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for ax, (title, func) in zip(axes, methods):
        # 1) Run clustering
        G_cluster, colors = func(G_base)

        # 2) Convert to AGraph
        A = to_agraph(G_cluster)

        # 3) Style nodes using the clustering colors
        for n in A.nodes():
            name = n.get_name()
            n.attr['label'] = name
            n.attr['style'] = 'filled'

            # Use the clustering color for every node; fall back to surface color
            fill = COLOR_CYCLE[colors.get(name, 0) % len(COLOR_CYCLE)]
            n.attr['fillcolor']  = fill.hex
            # Choose a contrasting font color
            n.attr['fontcolor']  = FLAVOR.colors.surface0.hex if fill.hsl.l > .4 else FLAVOR.colors.text.hex

            n.attr['shape']      = 'ellipse'
            n.attr['width']      = '1.75'
            n.attr['height']     = '1'
            n.attr['fixedsize']  = 'true'
            n.attr['fontname']   = 'Helvetica'
            n.attr['fontsize']   = '27'

            if name in traitors:
                n.attr['color']    = FLAVOR.colors.yellow.hex
                n.attr['penwidth'] = '8'

        # 4) Style edges (same as before)
        for u, v, data in G_cluster.edges(data=True):
            edge = A.get_edge(u, v)
            count = data.get('weight', 1)
            edge.attr['penwidth'] = str(1.5 + count * 1.5)
            edge.attr['color']    = 'lightgray'
            edge.attr['arrowhead']= 'normal'

        # 5) Layout & render
        A.graph_attr.update({
            'nodesep': '+1',   # horizontal space between nodes
            'ranksep': '-1',   # vertical space between ranks (for dot)
            'margin':  '0',
            'overlap': 'false',
            'sep': '-20',
            'dpi':     '300',
            'bgcolor': FLAVOR.colors.base.hex,
            'start': '12345',     # make fdp deterministic
        })
        A = auto_rotate_agraph(A, layout_prog='fdp')

        buf = io.BytesIO()
        A.draw(buf, format='png', prog='fdp', args='-n2')
        buf.seek(0)
        img = Image.open(buf)

        # 6) Show on axes
        ax.imshow(img)
        ax.set_title(title, fontsize=14)

        # Restore default axes (light gray border, no ticks)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    return plt

def plot_vote_share(vote_share_df):
    rounds = vote_share_df.index
    bottom = np.zeros(len(rounds))

    plt.figure(figsize=(10, 6))
    for i, player in enumerate(vote_share_df.columns):
        counts = vote_share_df[player].values
        plt.bar(rounds, counts, bottom=bottom, label=player, hatch=HATCHES[i % len(HATCHES)])
        bottom += counts

    plt.xlabel('Round')
    plt.ylabel('Number of Votes')
    plt.title('Vote Distribution per Round')
    plt.xticks(rounds)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()

    return plt

def plot_breakfast_groups(grouping_table):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- Left: Heatmap ----
    cmap = get_colormap_from_list(
        catppuccin.PALETTE.frappe.identifier,
        ["green", "peach", "red"],
    )
    cmap.set_bad(color=FLAVOR.colors.surface0.hex)
    im = ax0.imshow(grouping_table, aspect='auto', cmap=cmap)
    ax0.set_xticks(np.arange(grouping_table.shape[1]))
    ax0.set_xticklabels(grouping_table.columns)
    ax0.set_yticks(np.arange(grouping_table.shape[0]))
    ax0.set_yticklabels(grouping_table.index)
    ax0.set_xlabel("Round")
    ax0.set_ylabel("Group")
    ax0.set_title("Heatmap of Traitor Counts")
    fig.colorbar(im, ax=ax0, label="Traitor Count", fraction=0.046, pad=0.04)

    # ---- Right: Bar chart ----
    total_traitors = np.nansum(grouping_table.values, axis=1)
    ax1.bar(
        grouping_table.index.astype(str),
        total_traitors,
        hatch=HATCHES[:len(total_traitors)]
    )
    ax1.set_xlabel("Group")
    ax1.set_ylabel("Total Traitor Count")
    ax1.set_title("Total Traitors per Group Across All Rounds")

    plt.tight_layout()
    return plt
