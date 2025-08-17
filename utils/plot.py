import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import logomaker
import networkx as nx
from sklearn.manifold import TSNE
from process_data import load_embeddings


# Fig.3d: Scatter of Automated Experimental Replicates
def plot_scatter(data_file):
    df = pd.read_excel(data_file, sheet_name=0, header=1)
    fig = plt.figure(figsize=(6, 3))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    ax = fig.add_subplot(111, projection='3d')
    size, alpha = 5, 0.7

    x = df['Replicate1']
    y = df['Replicate2']
    z = df['Replicate3']

    df['zero'] = 0
    a = df['zero']
    ax.scatter(x, y, z, s=size, alpha=alpha)
    ax.scatter(x, y, a, s=size, alpha=alpha)
    ax.scatter(x, a, z, s=size, alpha=alpha)
    ax.scatter(a, y, z, s=size, alpha=alpha)

    ax.set_xlabel('Replicate 1')
    ax.set_ylabel('Replicate 2')
    ax.set_zlabel('Replicate 3')
    ax.set_title('Consistency of Parallel Experiments')
    plt.tight_layout()
    plt.show()


# Heatmap of Site Saturation Mutagenesis
def plot_heatmap(data_file):
    df = pd.read_excel(data_file, sheet_name=0)
    plt.figure(figsize=(8, 2.8))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tick_params(labelsize=12)

    adjusted_cmap = mcolors.LinearSegmentedColormap.from_list('adjusted', plt.cm.Oranges(np.linspace(0.1, 0.8, 256)))
    sns.heatmap(df.T, cmap=adjusted_cmap, annot=False, cbar_kws={'label': 'Yield'})

    plt.title('Heatmap of Site Saturation Mutagenesis', fontsize=14)
    plt.ylabel('Mutated Amino Acid', fontsize=12)
    plt.xlabel('Round', fontsize=12)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig('heatmap_SSM.svg', dpi=300, transparent=True)
    plt.show()


# Fig.4b: Sequence Logo
def row_minmax_sum1(df):
    row_min = df.min(axis=1)
    row_max = df.max(axis=1)
    denom   = row_max - row_min 
    norm = df.subtract(row_min, axis=0).div(denom.replace(0, np.nan), axis=0)
    norm = norm.fillna(0)
    row_sum = norm.sum(axis=1)
    scaled = norm.div(row_sum.replace(0, np.nan), axis=0)
    const_rows = scaled.isna().all(axis=1)
    scaled.loc[const_rows] = 1.0 / df.shape[1]
    scaled = scaled.fillna(0)
    return scaled 

def plot_logo(data_file):
    df = pd.read_excel(data_file, sheet_name=0, header=1)
    scaled = row_minmax_sum1(df)
    ww_logo = logomaker.Logo(scaled,
                            font_name='Arial',
                            color_scheme='weblogo_protein',
                            stack_order='big_on_top',
                            figsize=(6, 3),
                            vpad=.1,
                            width=.8)

    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tick_params(labelsize=12)

    ww_logo.style_xticks(anchor=0, spacing=1)
    ww_logo.ax.set_ylabel('Normalized Yield')

    plt.xticks(list(range(1,11)), 
        ['L75','V82','T180','E207','A330','S332','M354','L437','S466','R498'])
    plt.tight_layout()
    plt.savefig('logo_ssm_scaled.svg', dpi=300, transparent=True)


# Fig.4e: Boxplot of Site Saturation Mutagenesis
def plot_boxplot(data_file):
    df = pd.read_excel(data_file, sheet_name=0)
    plt.figure(figsize=(13, 3))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tick_params(labelsize=12)

    sns.boxplot(x='position', y='yield', data=filtered_data, color='black', fill=False, widths=0.4)
    
    plt.title('Yield Change by Position', fontsize=14)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Yield', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'boxplot_position_SSM.svg', dpi=300, transparent=True)
    plt.show()


# Fig.5b: Network analysis of double mutants
def norm_G(node_scaler, edge_scaler, final_single_mutants, single_mutants, final_double_mutants):
    G = nx.Graph()
    for mut in final_single_mutants:
        G.add_node(mut, activity=single_mutants[mut])
    for pair, value in final_double_mutants.items():
        mut1, mut2 = pair.split('-')
        G.add_edge(mut1, mut2, activity=value)
    node_sizes = node_scaler.fit_transform([[d['activity']] for n, d in G.nodes(data=True) if len(d) != 0]).flatten()
    edge_widths = edge_scaler.fit_transform([[d['activity']] for u, v, d in G.edges(data=True) if len(d) != 0]).flatten()
    return G, node_sizes, edge_widths

def plot_network(data_file, selected_num=20):
    df = pd.read_excel(data_file, sheet_name=0)
    df['name'] = df['name'].astype(str)
    df['yield'] = df['yield'].astype(float)

    def is_double_mutation(name):
        return name.count('-') == 1
    single_df = df[~df['name'].str.contains('-')].copy()
    single_mutants = single_df.set_index('name')['yield'].to_dict()
    double_df = df[df['name'].apply(is_double_mutation)]
    double_mutants = double_df.set_index('name')['yield'].to_dict()

    top_double = (double_df.sort_values(by='yield', ascending=False).head(selected_num))
    final_double_mutants = dict(zip(top_double['name'], top_double['yield']))
    single_yield_map = dict(zip(single_df['name'], single_df['yield']))

    final_single_mutants = {}
    for pair in final_double_mutants:
        mut1, mut2 = pair.split('-')
        # Add final_single_mutants
        if mut1 in single_yield_map:
            final_single_mutants[mut1] = single_yield_map[mut1]
        if mut2 in single_yield_map:
            final_single_mutants[mut2] = single_yield_map[mut2]

    node_scaler = MinMaxScaler(feature_range=(2000, 4000)) 
    edge_scaler = MinMaxScaler(feature_range=(1, 4)) 
    G, node_sizes, edge_widths = norm_G(node_scaler, edge_scaler, final_single_mutants, single_mutants, final_double_mutants)
    pos = nx.spring_layout(G, k=16, iterations=300, seed=4, scale=1) 

    plt.figure(figsize=(10, 5))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='white', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='orange', alpha=0.4)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='arial')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('mutation_network.svg', dpi=300, transparent=True)
    plt.show()


# Fig.5f: t-SNE projection of ESM-2 embeddings
def plot_tsne(embedding_folder):
    X, yield_data = load_embeddings(embeddings_folder)
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(4, 3))
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.tick_params(labelsize=12)

    viridis_cmap = mcolors.LinearSegmentedColormap.from_list('viridis_adjusted', plt.cm.viridis_r(np.linspace(0.1, 1, 256)))

    sc1 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.log(yield_data), cmap=viridis_cmap, alpha=0.7, s=5)
    cbar = plt.colorbar(sc1, shrink=0.6)
    cbar.set_label('Yield')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Protein Embeddings')
    plt.savefig('esm2-tsne-viridis.svg', dpi=300, transparent=True)
    plt.show()