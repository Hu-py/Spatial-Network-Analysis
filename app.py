#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 10:20:25 2025

@author: zha
"""


import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, distance_matrix
from scipy.stats import spearmanr
from io import StringIO, BytesIO
import json
import time

st.set_page_config(page_title="Spatial Network Analysis")

# ------------------------------
# Scenario generators (same logic as original)
# ------------------------------
def generate_grid(m=10, n=10, diagonals=False):
    G = nx.Graph()
    pos = {}
    for i in range(m):
        for j in range(n):
            idx = i*n + j
            x = j/(n-1) if n>1 else 0
            y = i/(m-1) if m>1 else 0
            G.add_node(idx)
            pos[idx] = (x, y)
    for i in range(m):
        for j in range(n):
            u = i*n + j
            if i+1 < m:
                v = (i+1)*n + j
                G.add_edge(u, v, length=1.0)
            if j+1 < n:
                v = i*n + (j+1)
                G.add_edge(u, v, length=1.0)
            if diagonals:
                if i+1 < m and j+1 < n:
                    G.add_edge(u, (i+1)*n + (j+1), length=np.sqrt(2))
                if i+1 < m and j-1 >= 0:
                    G.add_edge(u, (i+1)*n + (j-1), length=np.sqrt(2))
    return G, pos


def generate_organic(n=120, radius=0.15, extra_ratio=0.15, seed=1):
    rng = np.random.default_rng(seed)
    pts = rng.random((n,2))
    G = nx.random_geometric_graph(n, radius, pos={i: tuple(pts[i]) for i in range(n)})
    pos = nx.get_node_attributes(G, 'pos')
    D = distance_matrix(pts, pts)
    full = nx.Graph()
    for i in range(n):
        full.add_node(i)
    for i in range(n):
        for j in range(i+1, n):
            full.add_edge(i,j, weight=D[i,j])
    T = nx.minimum_spanning_tree(full, weight='weight')
    G.add_edges_from(T.edges())
    k = max(3, int(np.log2(n)))
    tree = KDTree(pts)
    total_extra = int(extra_ratio * n)
    added = 0
    for u in range(n):
        if added >= total_extra:
            break
        dists, idxs = tree.query(pts[u], k=k+1)
        for v in idxs[1:]:
            if not G.has_edge(u,v):
                G.add_edge(u,v, length=D[u,v])
                added += 1
                if added >= total_extra:
                    break
    return G, pos


def generate_hybrid(m=8, n=8, right_n=80, radius=0.18, bridges=8, seed=2):
    Gg, posg = generate_grid(m, n, diagonals=False)
    for u,(x,y) in posg.items():
        posg[u] = (0.02 + x*0.46, y)
    Go, poso = generate_organic(n=right_n, radius=radius, extra_ratio=0.12, seed=seed)
    for u,(x,y) in poso.items():
        poso[u] = (0.52 + x*0.46, y)
    offset = len(Gg.nodes)
    mapping = {u: u+offset for u in Go.nodes}
    Go = nx.relabel_nodes(Go, mapping)
    poso = {mapping[u]: p for u,p in poso.items()}
    H = nx.Graph(); H.update(Gg); H.update(Go)
    pos = {**posg, **poso}
    left_nodes = [u for u,(x,y) in pos.items() if x < 0.5]
    right_nodes = [u for u,(x,y) in pos.items() if x >= 0.5]
    left_pts = np.array([pos[u] for u in left_nodes])
    right_pts = np.array([pos[u] for u in right_nodes])
    treeR = KDTree(right_pts)
    added = 0; tried = set()
    for i,u in enumerate(left_nodes):
        if added>=bridges: break
        d, j = treeR.query(pos[u])
        v = right_nodes[j]
        if (u,v) in tried: continue
        H.add_edge(u,v, length=np.hypot(pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]))
        added += 1
        tried.add((u,v))
    return H, pos

# ------------------------------
# Centralities
# ------------------------------
CENTRALS = ['degree', 'closeness', 'betweenness', 'eigenvector', 'pagerank']

def compute_centralities(G):
    cent = {}
    cent['degree'] = {u: d for u,d in G.degree()}
    maxdeg = max(cent['degree'].values()) or 1
    cent['degree'] = {u: d/maxdeg for u,d in cent['degree'].items()}
    # closeness with 'length' fallback
    try:
        cent['closeness'] = nx.closeness_centrality(G, distance='length')
    except Exception:
        cent['closeness'] = nx.closeness_centrality(G)
    cent['betweenness'] = nx.betweenness_centrality(G, weight='length', normalized=True)
    try:
        cent['eigenvector'] = nx.eigenvector_centrality_numpy(G, weight='length')
    except Exception:
        cent['eigenvector'] = {u: 0.0 for u in G.nodes}
    try:
        cent['pagerank'] = nx.pagerank(G, weight='length')
    except Exception:
        cent['pagerank'] = nx.pagerank(G)
    return cent

# ------------------------------
# Helpers for visuals & dataframes
# ------------------------------
def draw_network_matplot(G, pos, centrality, title='Network'):
    vals = np.array([centrality[u] for u in G.nodes()])
    vmin, vmax = float(vals.min()), float(vals.max())
    fig, ax = plt.subplots(figsize=(6,5))
    nx.draw_networkx_edges(G, pos, width=0.8, alpha=0.5, ax=ax)
    sizes = 80 + 420 * (vals - vmin + 1e-9)/(vmax - vmin + 1e-9)
    nodes = nx.draw_networkx_nodes(G, pos, node_color=vals, cmap='viridis', node_size=sizes, ax=ax)
    nodes.set_edgecolor('k')
    ax.set_title(title)
    ax.axis('off')
    cbar = plt.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('centrality')
    return fig

def show_adjacency_figure(G):
    idx = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=idx, weight=None, dtype=int)
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(A, cmap='Greys', interpolation='nearest')
    ax.set_title('Adjacency (unweighted)')
    ax.set_xlabel('node'); ax.set_ylabel('node')
    plt.tight_layout()
    return fig

def centrality_dataframe(cent_dict):
    keys = sorted(next(iter(cent_dict.values())).keys())
    rows = []
    for u in keys:
        row = {'node': u}
        for name, m in cent_dict.items():
            row[name] = m[u]
        rows.append(row)
    return pd.DataFrame(rows).set_index('node')

def dashboard_plot(df, dfz, corr, title_prefix='', cent_ref=None):
    labels = list(dfz.columns)
    means = dfz.mean(axis=0).values
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    means_c = np.concatenate([means, [means[0]]])
    angles_c = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(14,8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1.2], width_ratios=[1,1])
    # Radar
    ax0 = fig.add_subplot(gs[0,0], polar=True)
    ax0.plot(angles_c, means_c, linewidth=2)
    ax0.fill(angles_c, means_c, alpha=0.2)
    ax0.set_xticks(angles); ax0.set_xticklabels(labels)
    ax0.set_title(f'{title_prefix} Mean normalized centralities (z)')

    # Correlation heatmap
    ax1 = fig.add_subplot(gs[0,1])
    im = ax1.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_yticks(range(len(labels))); ax1.set_yticklabels(labels)
    ax1.set_title('Spearman correlation among metrics')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Rank scatter
    ax2 = fig.add_subplot(gs[1,0])
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        ax2.scatter(df[a].rank(), df[b].rank(), alpha=0.6)
        r, p = spearmanr(df[a], df[b])
        ax2.set_xlabel(f'rank({a})'); ax2.set_ylabel(f'rank({b})')
        ax2.set_title(f'Rank–rank scatter: ρ={r:.2f}, p={p:.1e}')
    else:
        ax2.text(0.5, 0.5, 'Select ≥2 metrics', ha='center', va='center')
        ax2.axis('off')

    # Delta
    ax3 = fig.add_subplot(gs[1,1])
    if cent_ref is not None:
        delta_mean = (df - cent_ref).mean(axis=0)
        ax3.bar(range(len(labels)), delta_mean.values)
        ax3.set_xticks(range(len(labels))); ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_title('Mean Δ centrality (after − before)')
    else:
        ax3.text(0.5, 0.5, 'No reference (baseline) provided', ha='center', va='center')
        ax3.axis('off')

    fig.suptitle(f'{title_prefix} Multi-metric dashboard', fontsize=14)
    plt.tight_layout()
    return fig

# ------------------------------
# Session state initialization
# ------------------------------
if 'G' not in st.session_state:
    st.session_state.G = None
if 'pos' not in st.session_state:
    st.session_state.pos = None
if 'cent_before' not in st.session_state:
    st.session_state.cent_before = None
if 'cent_after' not in st.session_state:
    st.session_state.cent_after = None
if 'last_build_time' not in st.session_state:
    st.session_state.last_build_time = None

# ------------------------------
# Sidebar: Controls (similar to original)
# ------------------------------
st.sidebar.title("Scenario & Parameters")
scenario = st.sidebar.selectbox("Scenario", ['Grid','Organic','Hybrid'])
central_choice = st.sidebar.selectbox("Color by centrality", CENTRALS, index=CENTRALS.index('betweenness'))

# Grid params
st.sidebar.markdown("**Grid parameters**")
m = st.sidebar.slider("Grid m", 3, 20, 10)
n = st.sidebar.slider("Grid n", 3, 20, 10)
diagonals = st.sidebar.checkbox("8-neighbor (diagonals)", value=False)

# Organic params
st.sidebar.markdown("**Organic parameters**")
n_org = st.sidebar.slider("Organic nodes", 40, 400, 140, step=10)
radius = st.sidebar.slider("Radius", 0.05, 0.3, 0.15, step=0.01)
extra_ratio = st.sidebar.slider("Extra ratio", 0.0, 0.5, 0.15, step=0.05)

# Hybrid params
st.sidebar.markdown("**Hybrid parameters**")
m_h = st.sidebar.slider("Hybrid grid m", 4, 16, 8)
n_h = st.sidebar.slider("Hybrid grid n", 4, 16, 8)
right_n = st.sidebar.slider("Right nodes", 20, 300, 80, step=10)
r_h = st.sidebar.slider("Right radius", 0.06, 0.3, 0.18, step=0.01)
bridges = st.sidebar.slider("Bridges", 1, 30, 8)

st.sidebar.markdown("---")
st.sidebar.markdown("**Adjacency edit**")
# populate node selectors dynamically
def ensure_graph_built():
    if st.session_state.G is None:
        build_graph()
    # else leave as is

# Buttons
recompute = st.sidebar.button("Recompute scenario")
add_edge_btn = st.sidebar.button("Add edge")
remove_edge_btn = st.sidebar.button("Remove edge")
st.sidebar.markdown("---")
show_adj = st.sidebar.checkbox("Show adjacency", value=True)
show_hist = st.sidebar.checkbox("Show histogram", value=True)

st.sidebar.markdown("**Multi-metric dashboard**")
metrics_sel = st.sidebar.multiselect("Metrics to include", CENTRALS, default=CENTRALS)
show_dashboard_btn = st.sidebar.button("Show dashboard")
export_csv_btn = st.sidebar.button("Export CSVs")

# ------------------------------
# Functions to build graph & UI helpers
# ------------------------------
def build_graph():
    if scenario == 'Grid':
        G,pos = generate_grid(m, n, diagonals=diagonals)
    elif scenario == 'Organic':
        G,pos = generate_organic(n_org, radius, extra_ratio)
    else:
        G,pos = generate_hybrid(m_h, n_h, right_n, r_h, bridges)
    st.session_state.G = G
    st.session_state.pos = pos
    st.session_state.cent_before = None
    st.session_state.cent_after = None
    st.session_state.last_build_time = time.time()

def summarize_shift(cent0, cent1):
    idx = sorted(cent0.keys())
    v0 = np.array([cent0[i] for i in idx]); v1 = np.array([cent1[i] for i in idx])
    rho, p = spearmanr(v0, v1)
    delta = v1 - v0
    order = np.argsort(-np.abs(delta))[:10]
    top = [(idx[i], float(delta[i])) for i in order]
    return rho, p, top

# make sure initial build
ensure_graph_built = st.sidebar.checkbox("Ensure graph built (internal)", value=True, key="ensure_build")
if ensure_graph_built:
    if st.session_state.G is None or recompute:
        build_graph()

# Node selectors (after graph available)
G = st.session_state.G
pos = st.session_state.pos
node_list = sorted(G.nodes()) if G is not None else []
u = st.sidebar.selectbox("u", node_list, index=0 if node_list else None)
v = st.sidebar.selectbox("v", node_list, index=1 if len(node_list)>1 else None)

# Action handlers
if recompute:
    build_graph()
    st.experimental_rerun()

if add_edge_btn:
    if G is not None and u is not None and v is not None and u != v:
        if not G.has_edge(u,v):
            x0,y0 = pos[u]; x1,y1 = pos[v]
            G.add_edge(u,v, length=np.hypot(x0-x1, y0-y1))
            # set before/after tracking
            if st.session_state.cent_before is None:
                st.session_state.cent_before = compute_centralities(G)
                st.session_state.cent_after = None
            else:
                st.session_state.cent_after = compute_centralities(G)
    st.experimental_rerun()

if remove_edge_btn:
    if G is not None and u is not None and v is not None and u != v:
        if G.has_edge(u,v):
            G.remove_edge(u,v)
            if st.session_state.cent_before is None:
                st.session_state.cent_before = compute_centralities(G)
                st.session_state.cent_after = None
            else:
                st.session_state.cent_after = compute_centralities(G)
    st.experimental_rerun()

# ------------------------------
# Main layout (two columns)
# ------------------------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.header(f"{scenario} — nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    cent = compute_centralities(G)
    # update session state's centralities if baseline not set
    if st.session_state.cent_before is None:
        st.session_state.cent_before = cent
        st.session_state.cent_after = None
    else:
        # leave as is; current cent represents "current" state
        pass

    fig_net = draw_network_matplot(G, pos, cent[central_choice], title=f"{scenario} (colored by {central_choice})")
    st.pyplot(fig_net)

    if show_adj:
        fig_adj = show_adjacency_figure(G)
        st.pyplot(fig_adj)

    if show_hist:
        vals = np.array(list(cent[central_choice].values()))
        fig_hist, axh = plt.subplots(figsize=(6,2.5))
        axh.hist(vals, bins=20, alpha=0.8)
        axh.set_title(f'{central_choice} distribution')
        axh.set_xlabel('value'); axh.set_ylabel('count')
        st.pyplot(fig_hist)

    # If there is baseline and after, show summary
    if st.session_state.cent_before is not None and st.session_state.cent_after is not None:
        rho, p, top = summarize_shift(st.session_state.cent_before[central_choice], st.session_state.cent_after[central_choice])
        st.markdown(f"**Spearman rank correlation (before vs after) for {central_choice}:** rho={rho:.3f} (p={p:.3g})")
        st.markdown("**Top centrality changes (node, Δvalue):**")
        for nid, dv in top:
            st.write(f"- {nid:>4}: {dv:+.4f}")

with right_col:
    st.header("Multi-metric dashboard & export")

    # prepare dataframe(s)
    cent_current = compute_centralities(G)
    df = centrality_dataframe({k: cent_current[k] for k in CENTRALS})
    dfz = df.apply(lambda c: (c - c.mean())/(c.std() or 1.0))
    corr = df.corr(method='spearman')

    if show_dashboard_btn:
        # if we have a before and after, pass reference
        cent_ref_df = None
        if st.session_state.cent_before is not None and st.session_state.cent_after is not None:
            cent_before_df = centrality_dataframe({k: st.session_state.cent_before[k] for k in CENTRALS})
            cent_after_df = centrality_dataframe({k: st.session_state.cent_after[k] for k in CENTRALS})
            # compute df (after) and cent_ref (before)
            df_after = cent_after_df
            df_before = cent_before_df
            # create dashboard with after vs before delta
            sel_metrics = metrics_sel if metrics_sel else CENTRALS
            df_selected = df_after[sel_metrics]
            dfz_selected = (df_selected - df_selected.mean())/(df_selected.std() or 1.0)
            corr_sel = df_selected.corr(method='spearman')
            fig_dash = dashboard_plot(df_selected, dfz_selected, corr_sel, title_prefix=f'{scenario}', cent_ref=df_before[sel_metrics])
            st.pyplot(fig_dash)
            # allow CSV export
            st.session_state.df_export = df_selected
            st.session_state.dfz_export = dfz_selected
            st.session_state.corr_export = corr_sel
        else:
            sel_metrics = metrics_sel if metrics_sel else CENTRALS
            df_selected = df[sel_metrics]
            dfz_selected = dfz[sel_metrics]
            corr_sel = df_selected.corr(method='spearman')
            fig_dash = dashboard_plot(df_selected, dfz_selected, corr_sel, title_prefix=f'{scenario}', cent_ref=None)
            st.pyplot(fig_dash)
            st.session_state.df_export = df_selected
            st.session_state.dfz_export = dfz_selected
            st.session_state.corr_export = corr_sel

    # Export CSVs or provide downloads
    if export_csv_btn:
        if hasattr(st.session_state, 'df_export'):
            ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            csv1 = st.session_state.df_export.to_csv(index=True)
            csv2 = st.session_state.dfz_export.to_csv(index=True)
            csv3 = st.session_state.corr_export.to_csv(index=True)
            st.download_button("Download centralities CSV", data=csv1, file_name=f'centralities_{ts}.csv', mime='text/csv')
            st.download_button("Download centralities_z CSV", data=csv2, file_name=f'centralities_z_{ts}.csv', mime='text/csv')
            st.download_button("Download centralities_corr CSV", data=csv3, file_name=f'centralities_corr_{ts}.csv', mime='text/csv')
            st.success("Prepared CSVs for download.")
        else:
            st.warning("No data to export. Generate dashboard first.")

# Also add small "state inspector"
with st.expander("Session state (debug)"):
    ss = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'last_build_time': st.session_state.last_build_time
    }
    st.json(ss)

st.sidebar.markdown("---")
st.sidebar.caption("Controls ready. Use sidebar to pick a scenario, add/remove edges, and view dashboard. First 'Recompute' sets baseline automatically.")
