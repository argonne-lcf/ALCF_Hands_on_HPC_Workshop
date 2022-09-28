import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd

import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import RandomForestClassifier


def plot_pca(X, components=[1, 2], figsize=(8, 6),
             color_vector=None, scale=False, title=None):
    """
    Apply PCA to input X.
    Args:
        color_vector : each element corresponds to a row in X. Unique elements are colored with a different color.

    Returns:
        pca : object of sklearn.decomposition.PCA()
        x_pca : pca matrix
        fig : PCA plot figure handle
    """
    if color_vector is not None:
        assert len(X) == len(color_vector), 'len(df) and len(color_vector) must be the same size.'
        n_colors = len(np.unique(color_vector))
        colors = iter(cm.rainbow(np.linspace(0, 1, n_colors)))

    X = pd.DataFrame(X)

    # PCA
    if scale:
        xx = StandardScaler().fit_transform(X.values)
    else:
        xx = X.values

    n_components = max(components)
    pca = PCA(n_components=n_components)
    x_pca = pca.fit_transform(xx)
    pc0 = components[0] - 1
    pc1 = components[1] - 1

    # Start plotting
    fig, ax = plt.subplots(figsize=figsize)
    alpha = 0.7

    if color_vector is not None:
        for color in np.unique(color_vector):
            idx = color_vector == color
            c = next(colors)
            ax.scatter(x_pca[idx, pc0], x_pca[idx, pc1], alpha=alpha,
                       marker='o', edgecolor=c, color=c,
                       label=f'{color}')
    else:
        ax.scatter(x_pca[:, pc0], x_pca[:, pc1], alpha=alpha,
                   marker='s', edgecolors=None, color='b')

    ax.set_xlabel('PC' + str(components[0]))
    ax.set_ylabel('PC' + str(components[1]))
    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0.0), ncol=1, borderaxespad=0, frameon=True)
    plt.grid(True)
    if title:
        ax.set_title(title)

    print('Explained variance by PCA components [{}, {}]: [{:.5f}, {:.5f}]'.format(
        components[0], components[1],
        pca.explained_variance_ratio_[pc0],
        pca.explained_variance_ratio_[pc1]))

    return pca, x_pca


def load_cancer_data():
    """ Return cancer dataset (unscaled).
    Returns:
        X, Y
    """
    # Load data
    from sklearn import datasets
    data = datasets.load_breast_cancer()

    # Get features and target
    X = pd.DataFrame(data['data'], columns=data['feature_names'])
    X = X[sorted(X.columns)]
    Y = data['target']
    return X, Y


def plot_kmeans_obj(X_sc, tot_clusters=10):
    opt_obj_vec = []
    for k in range(1, tot_clusters):
        model = KMeans(n_clusters=k)  
        model.fit(X_sc)
        opt_obj_vec.append(model.inertia_/X_sc.shape[0])
        
    # Plot
    k = np.arange(len(opt_obj_vec)) + 1
    
    plt.figure(figsize=(8, 6))
    plt.plot(k, opt_obj_vec, '--o')
    plt.xlabel('Number of clusters (k)', fontsize=14)
    plt.ylabel('Inertia', fontsize=14)
    plt.grid(True)
        
    return opt_obj_vec


def split_tr_te(X, Y, te_size=0.2):
    from sklearn.model_selection import train_test_split
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    
    xtr, xte, ytr, yte = train_test_split(X, Y, test_size=te_size)
    xtr.reset_index(drop=True, inplace=True)
    xte.reset_index(drop=True, inplace=True)
    
    return xtr, xte, ytr, yte


def chk_tissues(x, tissues):
    return any([True if t in x else False for t in tissues])


def create_rna_data():
    # Load data
    rna_org = pd.read_csv('/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1/combined_rnaseq_data_lincs1000', sep='\t')
    meta_org = pd.read_csv('/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1/combined_metadata_2018May.txt', sep='\t')
    ctypes_org = pd.read_csv('/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1/combined_cancer_types', sep='\t')

    print(rna_org.shape)
    print(meta_org.shape)
    print(ctypes_org.shape)

    rna = rna_org.copy()
    meta = meta_org.copy()
    
    # Update rna df
    rna.rename(columns={'Sample': 'sample'}, inplace=True)
    rna.set_index('sample', inplace=True)         # set dataframe index to sample (we will later merge dataframes on index)
    rna.columns = ['GE_'+c for c in rna.columns]  # add prefix 'GE_' to all gene expression columns
    
    # Scale the features
    rna_index, rna_cols = rna.index, rna.columns
    rna = StandardScaler().fit_transform(rna)
    rna = pd.DataFrame(rna, index=rna_index, columns=rna_cols)
    
    # Update meta df
    meta = meta.rename(
        columns={'sample_name': 'sample', 'dataset': 'src',
                 'sample_category': 'category', 'sample_descr': 'descr',
                 'tumor_site_from_data_src': 'csite', 'tumor_type_from_data_src': 'ctype',
                 'simplified_tumor_site': 'simp_csite', 'simplified_tumor_type': 'simp_ctype'})

    # Extract a subset of cols
    meta = meta[['sample', 'src', 'csite', 'ctype', 'simp_csite', 'simp_ctype', 'category', 'descr']]

    meta['src'] = meta['src'].map(lambda x: x.lower())
    meta['csite'] = meta['csite'].map(lambda x: x.strip())
    meta['ctype'] = meta['ctype'].map(lambda x: x.strip())
    meta['src'] = meta['src'].map(lambda x: 'gdc' if x=='tcga' else x)

    meta.set_index('sample', inplace=True)  # add prefix 'GE_' to all gene expression columns
    
    # Filter on source
    sources = ['gdc']
    rna = rna.loc[rna.index.map(lambda s: s.split('.')[0].lower() in sources), :]
    print(rna.shape)
    
    # Update rna and meta
    on = 'sample'
    df = pd.merge(meta, rna, how='inner', on=on)
    
    col = 'csite'
    df[col] = df[col].map(lambda x: x.split('/')[0])
    # df[col].value_counts()
    
    # GDC
    tissues = ['breast', 'skin', 'lung', 'prostate']
    df = df.loc[ df[col].map(lambda x: chk_tissues(x, tissues)), : ]
    print(df['csite'].value_counts())
        
    # Randomly sample a subset of samples to create a balanced dataset
    sample_sz = 7
    df_list = []
    for i, t in enumerate(tissues):
        print(t)
        tmp = df[ df[col].isin([t]) ].sample(n=sample_sz, random_state=seed)
        df_list.append(tmp)

    df = pd.concat(df_list, axis=0)
    df = df.sample(frac=1.0)    

    # Dump df
    df = df.drop(columns=['simp_csite', 'simp_ctype', 'category', 'descr'])
    df.to_csv('rna.csv')
    
    
def plot_hists(k_means_bins, y_bins, x_labels = ['Malignant', 'Benign']):
    """ Specific function to plot histograms from bins.
    matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    x = np.arange(len(x_labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, y_bins, width, label='True label')
    rects2 = ax.bar(x + width/2, k_means_bins, width, label='K-means')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Total count')
    ax.set_title('Histogram')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 450)
    ax.legend(loc='best')

    def autolabel(rects):
        """ Attach a text label above each bar in *rects*, displaying its height. """
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.show()
    