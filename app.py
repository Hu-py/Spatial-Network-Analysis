#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Spatial Network Analysis — 自动响应版
Created on Mon Oct 20 10:20:25 2025
@author: zha (modified)
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
    treeR = KDTree(right_pts) if len(right_pts)>0 else None
    added = 0; tried = set()
    for i,u in enumerate(left_nodes):
        if added>=bridges or treeR is None: break
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

    labels = df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_c = np.concatenate([angles, [angles[0]]])  # 闭合
    means_c = np.concatenate([dfz.mean(axis=0).values, [dfz.mean(axis=0).values[0]]])

    fig, axes = plt.subplots(4, 1, figsize=(8, 20), constrained_layout=True)

    # ------------------ 1. Radar ------------------
    ax0 = plt.subplot(4, 1, 1, polar=True)
    ax0.plot(angles_c, means_c, linewidth=2)
    ax0.fill(angles_c, means_c, alpha=0.2)
    ax0.set_xticks(angles)
    ax0.set_xticklabels(labels)
    #ax0.set_yticklabels([])        # 去掉极坐标的圆环刻度
    #ax0.spines['polar'].set_visible(False)  # 去掉雷达外框
    ax0.set_title(f'{title_prefix} Mean normalized centralities (z)', fontsize=12)

    # ------------------ 2. Correlation heatmap ------------------
    ax1 = axes[1]
    im = ax1.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_title('Spearman correlation among metrics', fontsize=12)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # ------------------ 3. Rank scatter ------------------
    ax2 = axes[2]
    if len(labels) >= 2:
        a, b = labels[0], labels[1]
        ax2.scatter(df[a].rank(), df[b].rank(), alpha=0.6)
        r, p = spearmanr(df[a], df[b])
        ax2.set_xlabel(f'rank({a})')
        ax2.set_ylabel(f'rank({b})')
        ax2.set_title(f'Rank–rank scatter: ρ={r:.2f}, p={p:.1e}')
    else:
        ax2.text(0.5, 0.5, 'Select ≥2 metrics', ha='center', va='center')
        ax2.axis('off')

    # ------------------ 4. Delta centrality ------------------
    ax3 = axes[3]
    if cent_ref is not None:
        delta_mean = (df - cent_ref).mean(axis=0)
        ax3.bar(range(len(labels)), delta_mean.values)
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_title('Mean Δ centrality (after − before)')
    else:
        ax3.text(0.5, 0.5, 'No baseline provided', ha='center', va='center')
        ax3.axis('off')

    fig.suptitle(f'{title_prefix} Multi-metric dashboard', fontsize=14)

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
if 'last_params' not in st.session_state:
    st.session_state.last_params = {}

# ------------------------------
# Sidebar: Controls (reactive)
# ------------------------------
st.set_page_config(page_title="Spatial Network Analysis")
st.title("Spatial Network Analysis")

st.sidebar.title("Scenario & Parameters")

scenario = st.sidebar.selectbox("Scenario", ['Grid','Organic','Hybrid'])
central_choice = st.sidebar.selectbox("Color by centrality", ['degree','closeness','betweenness','eigenvector','pagerank'], index=2)

# Grid parameters
if scenario == 'Grid':
    st.sidebar.markdown("**Grid parameters**")
    m = st.sidebar.slider("Grid m", 3, 20, m)
    n = st.sidebar.slider("Grid n", 3, 20, n)
    diagonals = st.sidebar.checkbox("8-neighbor (diagonals)", value=diagonals)
elif scenario == 'Organic':
    st.sidebar.markdown("**Organic parameters**")
    n_org = st.sidebar.slider("Organic nodes", 40, 400, n_org, step=10)
    radius = st.sidebar.slider("Radius", 0.05, 0.3, radius, step=0.01)
    extra_ratio = st.sidebar.slider("Extra ratio", 0.0, 0.5, extra_ratio, step=0.05)
else:  # Hybrid
    st.sidebar.markdown("**Hybrid parameters**")
    m_h = st.sidebar.slider("Hybrid grid m", 4, 16, m_h)
    n_h = st.sidebar.slider("Hybrid grid n", 4, 16, n_h)
    right_n = st.sidebar.slider("Right nodes", 20, 300, right_n, step=10)
    r_h = st.sidebar.slider("Right radius", 0.06, 0.3, r_h, step=0.01)
    bridges = st.sidebar.slider("Bridges", 1, 30, bridges)


st.sidebar.markdown("---")
st.sidebar.markdown("**Adjacency editing (select nodes, click button to apply)**")
add_edge_click = st.sidebar.button("Add edge")
remove_edge_click = st.sidebar.button("Remove edge")
st.sidebar.markdown("---")

st.sidebar.markdown("**Multi-metric dashboard**")
metrics_sel = st.sidebar.multiselect("Select metrics", ['degree','closeness','betweenness','eigenvector','pagerank'], default=['degree','closeness','betweenness','eigenvector','pagerank'])
st.sidebar.markdown("---")
st.sidebar.caption("All changes automatically update the right panel; edge edits require button click.")

# ------------------------------
# Rebuild graph if parameters change
# ------------------------------
current_params = {
    'scenario': scenario, 'm': m, 'n': n, 'diagonals': diagonals,
    'n_org': n_org, 'radius': float(radius), 'extra_ratio': float(extra_ratio),
    'm_h': m_h, 'n_h': n_h, 'right_n': right_n, 'r_h': float(r_h), 'bridges': bridges
}

def need_rebuild(last_params, current_params):
    return last_params != current_params

if 'last_params' not in st.session_state:
    st.session_state.last_params = {}

if st.session_state.get('G') is None or need_rebuild(st.session_state.last_params, current_params):
    st.session_state.last_params = current_params.copy()
    if scenario == 'Grid':
        G, pos = generate_grid(m, n, diagonals)
    elif scenario == 'Organic':
        G, pos = generate_organic(n_org, radius, extra_ratio)
    else:
        G, pos = generate_hybrid(m_h, n_h, right_n, r_h, bridges)
    st.session_state.G = G
    st.session_state.pos = pos
    st.session_state.cent_before = None
    st.session_state.cent_after = None
    st.session_state.last_build_time = time.time()

G = st.session_state.G
pos = st.session_state.pos

# Update node selection options
node_list = sorted(G.nodes())
u = st.sidebar.selectbox("Node u", node_list, index=0 if node_list else None)
v = st.sidebar.selectbox("Node v", node_list, index=1 if len(node_list)>1 else None)

# ------------------------------
# Edge edit handlers
# ------------------------------
if add_edge_click and G is not None and u is not None and v is not None and u != v:
    prev_cent = compute_centralities(G)
    if not G.has_edge(u,v):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        G.add_edge(u,v, length=np.hypot(x0-x1, y0-y1))
    st.session_state.cent_before = prev_cent
    st.session_state.cent_after = compute_centralities(G)

if remove_edge_click and G is not None and u is not None and v is not None and u != v:
    prev_cent = compute_centralities(G)
    if G.has_edge(u,v):
        G.remove_edge(u,v)
    st.session_state.cent_before = prev_cent
    st.session_state.cent_after = compute_centralities(G)

# ------------------------------
# Main layout
# ------------------------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader(f"{scenario} — nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    cent = compute_centralities(G)
    if st.session_state.cent_before is None:
        st.session_state.cent_before = cent
        st.session_state.cent_after = None

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

    if st.session_state.cent_before is not None and st.session_state.cent_after is not None:
        rho, p, top = (lambda c0, c1: (spearmanr(
            np.array([c0[i] for i in sorted(c0.keys())]),
            np.array([c1[i] for i in sorted(c1.keys())]))[0],
            spearmanr(
            np.array([c0[i] for i in sorted(c0.keys())]),
            np.array([c1[i] for i in sorted(c1.keys())]))[1],
            sorted([(n, float(c1[n]-c0[n])) for n in sorted(c0.keys())], key=lambda x: -abs(x[1]))[:10]
        ))(st.session_state.cent_before[central_choice], st.session_state.cent_after[central_choice])
        st.markdown(f"**Spearman rank correlation (before vs after) for {central_choice}:** rho={rho:.3f} (p={p:.3g})")
        st.markdown("**Top nodes by change (node, Δvalue):**")
        for nid, dv in top:
            st.write(f"- {nid:>4}: {dv:+.4f}")

with right_col:
    st.subheader("Multi-metric dashboard & export")
    cent_current = compute_centralities(G)
    df_all = centrality_dataframe({k: cent_current[k] for k in ['degree','closeness','betweenness','eigenvector','pagerank']})
    dfz_all = df_all.apply(lambda c: (c - c.mean())/(c.std() or 1.0))

    sel_metrics = metrics_sel if metrics_sel else ['degree','closeness','betweenness','eigenvector','pagerank']
    df_selected = df_all[sel_metrics]
    dfz_selected = dfz_all[sel_metrics]

    cent_ref_df = None
    if st.session_state.cent_before is not None and st.session_state.cent_after is not None:
        cent_ref_df = centrality_dataframe({k: st.session_state.cent_before[k] for k in ['degree','closeness','betweenness','eigenvector','pagerank']})[sel_metrics]
        df_after = centrality_dataframe({k: st.session_state.cent_after[k] for k in ['degree','closeness','betweenness','eigenvector','pagerank']})[sel_metrics]
        df_for_dash = df_after
        dfz_for_dash = (df_for_dash - df_for_dash.mean())/(df_for_dash.std() or 1.0)
        corr_for_dash = df_for_dash.corr(method='spearman')
        fig_dash = dashboard_plot(df_for_dash, dfz_for_dash, corr_for_dash, title_prefix=f'{scenario}', cent_ref=cent_ref_df)
    else:
        corr_sel = df_selected.corr(method='spearman')
        fig_dash = dashboard_plot(df_selected, dfz_selected, corr_sel, title_prefix=f'{scenario}', cent_ref=None)

    st.pyplot(fig_dash)

    ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    csv1 = df_selected.to_csv(index=True)
    csv2 = dfz_selected.to_csv(index=True)
    csv3 = df_selected.corr(method='spearman').to_csv(index=True)

    st.download_button("Download centralities CSV", data=csv1, file_name=f'centralities_{ts}.csv', mime='text/csv')
    st.download_button("Download centralities_z CSV", data=csv2, file_name=f'centralities_z_{ts}.csv', mime='text/csv')
    st.download_button("Download centralities_corr CSV", data=csv3, file_name=f'centralities_corr_{ts}.csv', mime='text/csv')

# ------------------------------
# Debug / session state
# ------------------------------
with st.expander("Session state (debug)"):
    ss = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'last_build_time': st.session_state.get('last_build_time'),
        'last_params_snapshot': st.session_state.get('last_params')
    }
    st.json(ss)
