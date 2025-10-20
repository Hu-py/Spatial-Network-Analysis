#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Spatial Network Analysis — 简化版（不支持手动增删边）
"""
import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import KDTree, distance_matrix
from scipy.stats import spearmanr
import random

# ------------------------------
# Scenario generators
# ------------------------------
def generate_grid(m=10, n=10, diagonals=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
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
            if i+1 < m: G.add_edge(u, (i+1)*n+j, length=1.0)
            if j+1 < n: G.add_edge(u, i*n+j+1, length=1.0)
            if diagonals:
                if i+1 < m and j+1 < n: G.add_edge(u, (i+1)*n+j+1, length=np.sqrt(2))
                if i+1 < m and j-1 >=0: G.add_edge(u, (i+1)*n+j-1, length=np.sqrt(2))
    return G, pos

def generate_organic(n=120, radius=0.15, extra_ratio=0.15, seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.random((n,2))
    G = nx.random_geometric_graph(n, radius, pos={i: tuple(pts[i]) for i in range(n)})
    pos = nx.get_node_attributes(G, 'pos')
    D = distance_matrix(pts, pts)
    full = nx.Graph()
    for i in range(n): full.add_node(i)
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
        if added >= total_extra: break
        dists, idxs = tree.query(pts[u], k=k+1)
        for v in idxs[1:]:
            if not G.has_edge(u,v):
                G.add_edge(u,v, length=D[u,v])
                added += 1
                if added >= total_extra: break
    return G, pos

def generate_hybrid(m=8, n=8, right_n=80, radius=0.18, bridges=8, seed=None):
    Gg, posg = generate_grid(m, n, diagonals=False, seed=seed)
    for u,(x,y) in posg.items(): posg[u] = (0.02 + x*0.46, y)
    Go, poso = generate_organic(n=right_n, radius=radius, extra_ratio=0.12, seed=seed)
    for u,(x,y) in poso.items(): poso[u] = (0.52 + x*0.46, y)
    offset = len(Gg.nodes)
    mapping = {u:u+offset for u in Go.nodes}
    Go = nx.relabel_nodes(Go, mapping)
    poso = {mapping[u]:p for u,p in poso.items()}
    H = nx.Graph(); H.update(Gg); H.update(Go)
    pos = {**posg, **poso}
    return H, pos

# ------------------------------
# Centralities
# ------------------------------
CENTRALS = ['degree','closeness','betweenness','eigenvector','pagerank']

def compute_centralities(G):
    cent = {}
    cent['degree'] = {u: d for u,d in G.degree()}
    maxdeg = max(cent['degree'].values()) or 1
    cent['degree'] = {u: d/maxdeg for u,d in cent['degree'].items()}
    try: cent['closeness'] = nx.closeness_centrality(G, distance='length')
    except: cent['closeness'] = nx.closeness_centrality(G)
    cent['betweenness'] = nx.betweenness_centrality(G, weight='length', normalized=True)
    try: cent['eigenvector'] = nx.eigenvector_centrality_numpy(G, weight='length')
    except: cent['eigenvector'] = {u:0.0 for u in G.nodes}
    try: cent['pagerank'] = nx.pagerank(G, weight='length')
    except: cent['pagerank'] = nx.pagerank(G)
    return cent

# ------------------------------
# Visual helpers
# ------------------------------
def draw_network_matplot(G,pos,centrality,title='Network'):
    vals = np.array([centrality[u] for u in G.nodes()])
    vmin,vmax = float(vals.min()), float(vals.max())
    fig,ax = plt.subplots(figsize=(6,5))
    nx.draw_networkx_edges(G,pos,width=0.8,alpha=0.5,ax=ax)
    sizes = 80 + 420*(vals-vmin+1e-9)/(vmax-vmin+1e-9)
    nodes = nx.draw_networkx_nodes(G,pos,node_color=vals,cmap='viridis',node_size=sizes,ax=ax)
    nodes.set_edgecolor('k')
    ax.set_title(title)
    ax.axis('off')
    cbar=plt.colorbar(nodes,ax=ax,fraction=0.02,pad=0.03,shrink=0.4)
    cbar.set_label('centrality')
    return fig

def centrality_dataframe(cent_dict):
    keys = sorted(next(iter(cent_dict.values())).keys())
    rows = []
    for u in keys:
        row = {'node':u}
        for name,m in cent_dict.items(): row[name]=m[u]
        rows.append(row)
    return pd.DataFrame(rows).set_index('node')

def dashboard_plot(df,dfz,corr,title_prefix='',cent_ref=None,main_seed=None,base_seed=None):
    labels = df.columns.tolist()
    angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
    angles_c = np.concatenate([angles,[angles[0]]])
    means_c = np.concatenate([dfz.mean(axis=0).values,[dfz.mean(axis=0).values[0]]])
    fig,axes = plt.subplots(4,1,figsize=(8,20),constrained_layout=True)
    # Radar
    ax0 = plt.subplot(3,1,1,polar=True)
    ax0.plot(angles_c,means_c,linewidth=2)
    ax0.fill(angles_c,means_c,alpha=0.2)
    ax0.set_xticks(angles)
    ax0.set_xticklabels(labels)
    ax0.set_title(f'{title_prefix} Mean normalized centralities (z)')
    # Heatmap
    ax1 = axes[1]
    im=ax1.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels,rotation=45,ha='right')
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels)
    ax1.set_title('Spearman correlation among metrics')
    fig.colorbar(im,ax=ax1,fraction=0.046,pad=0.04)
    # Rank scatter
    ax2 = axes[2]
    if len(labels)>=2:
        a,b=labels[0],labels[1]
        ax2.scatter(df[a].rank(),df[b].rank(),alpha=0.6)
        r,p=spearmanr(df[a],df[b])
        ax2.set_xlabel(f'rank({a})'); ax2.set_ylabel(f'rank({b})')
        ax2.set_title(f'Rank–rank scatter: ρ={r:.2f}, p={p:.1e}')
    else:
        ax2.text(0.5,0.5,'Select ≥2 metrics',ha='center',va='center'); ax2.axis('off')

    fig.suptitle(f'{title_prefix} Multi-metric dashboard',fontsize=14)
    return fig

def dashboard_compare(df_main, df_compare, title_prefix='', main_seed=None, compare_seed=None):
    labels = df_main.columns.tolist()
    x = np.arange(len(labels))
    width = 0.35
    main_vals = df_main.mean(axis=0).values
    compare_vals = df_compare.mean(axis=0).values

    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(x - width/2, main_vals, width, label='Main', color='tab:blue', alpha=0.7)
    ax.bar(x + width/2, compare_vals, width, label='Comparison', color='tab:orange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(f'Mean centrality comparison\n(main seed={main_seed}, comparison seed={compare_seed})')
    ax.legend()
    fig.suptitle(f'{title_prefix} Comparison', fontsize=14)
    return fig
    
# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Spatial Network Analysis")
st.title("Spatial Network Analysis")

# Sidebar: seeds
rand_seed_main = random.randint(0,1_000_000)
rand_seed_base = random.randint(0,1_000_000)
st.sidebar.markdown(f"**Main seed:** `{rand_seed_main}`")
st.sidebar.markdown(f"**Baseline seed:** `{rand_seed_base}`")

# Sidebar: scenario & centrality
scenario = st.sidebar.selectbox("Scenario", ['Grid','Organic','Hybrid'])
central_choice = st.sidebar.selectbox("Color by centrality", CENTRALS, index=2)

# Default params
m,n,diagonals = 10,10,False
n_org,radius,extra_ratio = 140,0.15,0.15
m_h,n_h,right_n,r_h,bridges = 8,8,80,0.18,8

# Sidebar sliders
if scenario=='Grid':
    m = st.sidebar.slider("Grid m",3,20,m)
    n = st.sidebar.slider("Grid n",3,20,n)
    diagonals = st.sidebar.checkbox("8-neighbor (diagonals)",value=diagonals)
elif scenario=='Organic':
    n_org = st.sidebar.slider("Organic nodes",40,400,n_org,step=10)
    radius = st.sidebar.slider("Radius",0.05,0.3,radius,step=0.01)
    extra_ratio = st.sidebar.slider("Extra ratio",0.0,0.5,extra_ratio,step=0.05)
else:
    m_h = st.sidebar.slider("Hybrid grid m",4,16,m_h)
    n_h = st.sidebar.slider("Hybrid grid n",4,16,n_h)
    right_n = st.sidebar.slider("Right nodes",20,300,right_n,step=10)
    r_h = st.sidebar.slider("Right radius",0.06,0.3,r_h,step=0.01)
    bridges = st.sidebar.slider("Bridges",1,30,bridges)

# ------------------------------
# Generate main & baseline graphs
# ------------------------------
def build_graph(scenario,seed):
    if scenario=='Grid': return generate_grid(m,n,diagonals,seed)
    elif scenario=='Organic': return generate_organic(n_org,radius,extra_ratio,seed)
    else: return generate_hybrid(m_h,n_h,right_n,r_h,bridges,seed)

G_main,pos_main = build_graph(scenario,rand_seed_main)
G_base,pos_base = build_graph(scenario,rand_seed_base)

cent_main = compute_centralities(G_main)
cent_base = compute_centralities(G_base)

df_main = centrality_dataframe(cent_main)
df_base = centrality_dataframe(cent_base)

dfz_main = df_main.apply(lambda c: (c-c.mean())/(c.std() or 1.0))
df_selected = df_main[CENTRALS]
dfz_selected = dfz_main[CENTRALS]

# ------------------------------
# Display network & dashboard
# ------------------------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader(f"{scenario} network (main seed)")
    fig_net = draw_network_matplot(
        G_main, pos_main, cent_main[central_choice],
        title=f"{scenario} nodes={G_main.number_of_nodes()}, edges={G_main.number_of_edges()} (colored by {central_choice})"
    )
    st.pyplot(fig_net)

    st.subheader("Adjacency matrix (unweighted)")
    idx = list(G_main.nodes())
    A = nx.to_numpy_array(G_main, nodelist=idx, weight=None, dtype=int)
    fig_adj, ax_adj = plt.subplots(figsize=(5,5))
    ax_adj.imshow(A, cmap='Greys', interpolation='nearest')
    ax_adj.set_title('Adjacency (unweighted)')
    ax_adj.set_xlabel('node'); ax_adj.set_ylabel('node')
    st.pyplot(fig_adj)

    st.subheader(f"{central_choice} distribution (main seed)")
    vals = np.array(list(cent_main[central_choice].values()))
    fig_hist, axh = plt.subplots(figsize=(6,2.5))
    axh.hist(vals, bins=20, alpha=0.8)
    axh.set_title(f'{central_choice} distribution')
    axh.set_xlabel('value'); axh.set_ylabel('count')
    st.pyplot(fig_hist)


with right_col:
    st.subheader("Multi-metric dashboard & Δ centrality vs baseline")
    corr_main = df_selected.corr(method='spearman')
    fig_dash = dashboard_plot(
        df_selected, dfz_selected, corr_main,
        title_prefix=scenario,
        cent_ref=df_base[CENTRALS],
        main_seed=rand_seed_main,
        base_seed=rand_seed_base
    )
    st.pyplot(fig_dash)


left_col, mid_col, right_col = st.columns([1,1,1])

# 左：主网络
with left_col:
    st.markdown("**Main network**")
    fig_net = draw_network_matplot(G_main, pos_main, cent_main[central_choice],
                                   title=f"{scenario} nodes={G_main.number_of_nodes()}, edges={G_main.number_of_edges()} (colored by {central_choice})")
    st.pyplot(fig_net)

# 中：对比图
with mid_col:
    st.markdown(f"**Comparison network ({scenario}, seed={rand_seed_base})**")
    fig_net_cmp = draw_network_matplot(G_base, pos_base, cent_base[central_choice],
                                       title=f"{scenario} nodes={G_base.number_of_nodes()}, edges={G_base.number_of_edges()} (colored by {central_choice})")
    st.pyplot(fig_net_cmp)
    
# 右： 对比网络
with right_col:
    st.markdown(f"**Comparison centralities**")
    fig_cmp = dashboard_compare(df_selected, df_base[CENTRALS],
                                title_prefix=scenario,
                                main_seed=rand_seed_main,
                                compare_seed=rand_seed_base)
    st.pyplot(fig_cmp)



# ------------------------------
# Export CSVs
# ------------------------------
ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

st.download_button("Download main centralities CSV",
               data=df_main.to_csv(index=True),
               file_name=f'centralities_main_{ts}.csv',
               mime='text/csv')

st.download_button("Download baseline centralities CSV",
               data=df_base.to_csv(index=True),
               file_name=f'centralities_baseline_{ts}.csv',
               mime='text/csv')

st.download_button("Download Δ centralities CSV",
               data=(df_selected - df_base[CENTRALS]).to_csv(index=True),
               file_name=f'centralities_delta_{ts}.csv',
               mime='text/csv')

