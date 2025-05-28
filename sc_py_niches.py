import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, vstack



#%%

# read in data
data_prefix = '/users/dgibbs/Work/CRUK/py-sc-niches/data/'
q = sc.read_h5ad(data_prefix+'gastric_fibs_and_epi0_4.h5ad')


#%%

def subsample_balanced(adata, group_key, n_per_group=None, random_state=426736):
    """Subsample an AnnData object so that each group in `group_key` has equal cell count."""
    groups = adata.obs[group_key]
    rng = np.random.default_rng(seed=random_state)

    # Determine number to sample per group
    group_sizes = groups.value_counts()
    min_group_size = group_sizes.min()
    n = min_group_size if n_per_group is None else min(n_per_group, min_group_size)

    # Sample indices per group
    sampled_indices = []
    for g in group_sizes.index:
        idx = adata.obs[adata.obs[group_key] == g].index
        sampled = rng.choice(idx, size=n, replace=False)
        sampled_indices.extend(sampled)

    # Return new AnnData object
    return adata[sampled_indices].copy()


### get 100 of each cell type

qq = subsample_balanced(q, 'major_cell_types', n_per_group=100)


#%%

# this is from Omnipath, and annotated with gene symbols.
# we're only looking at single gene-ligand / receptor interactions
merged_df = pd.read_csv(data_prefix+'omnipath_ligrecextra_annot.csv')

# then we have our list of genes
source_genes = filtered_df.source_symbol.values
target_genes = filtered_df.target_symbol.values
# and the .var index here
source_target = [x+'_'+y for x,y in zip(source_genes, target_genes)]

seen_pairs = set()
unique_source_target = []
unique_source_genes = []
unique_target_genes = []

for pair, s_genes, t_genes in zip(source_target, source_genes, target_genes):
    if pair not in seen_pairs:
        seen_pairs.add(pair)
        unique_source_target.append(pair)
        unique_source_genes.append(s_genes)
        unique_target_genes.append(t_genes)

# these are the new column names for the obs table
cols_ci = ['ci_' + colx for colx in q.obs.columns]
cols_cj = ['cj_' + colx for colx in q.obs.columns]

# Chunking utility
def chunk_indices(n, chunk_size):
    return [list(range(i, min(i + chunk_size, n))) for i in range(0, n, chunk_size)]

# Core logic to process a chunk of ci indices
def process_ci_chunk(ci_chunk, qq, source_genes, target_genes, cols_ci, cols_cj):
    obs_list = []
    X_list = []
    error_list = []

    print(ci_chunk)

    for ci in ci_chunk:
        for cj in range(ci + 1, qq.shape[0]):
            try:
                cellid = f"{ci}_{cj}"
                df_ci = qq.obs.iloc[[ci]].copy()
                df_cj = qq.obs.iloc[[cj]].copy()
                df_ci.columns = cols_ci
                df_cj.columns = cols_cj

                df_obs_merge = pd.concat([df_ci.reset_index(drop=True),
                                          df_cj.reset_index(drop=True)], axis=1)
                df_obs_merge["cellcellid"] = cellid

                x1 = qq[ci, source_genes].X.todense()
                x2 = qq[cj, target_genes].X.todense()
                x1_x2 = np.multiply(x1, x2)

                obs_list.append(df_obs_merge)
                X_list.append(x1_x2)
            except:
                error_list.append(f"{ci}_{cj}")

    return obs_list, X_list, error_list



n_cells = qq.shape[0]
chunk_size = 100  # adjust based on memory and CPU cores
ci_chunks = chunk_indices(n_cells, chunk_size)

cols_ci = ['ci_'+colx for colx in q.obs.columns]
cols_cj = ['cj_'+colx for colx in q.obs.columns]

results = Parallel(n_jobs=8, backend="loky")(
    delayed(process_ci_chunk)(ci_chunk, qq, unique_source_genes, unique_target_genes, cols_ci, cols_cj)
    for ci_chunk in ci_chunks
)

# Merge results
obs_list   = [item for chunk in results for item in chunk[0]]
X_list     = [item for chunk in results for item in chunk[1]]
error_list = [item for chunk in results for item in chunk[2]]

obs_df = pd.concat(obs_list, axis=0)

# Convert each dense matrix (1 x N) to a sparse row
X_sparse_rows = [csr_matrix(x) for x in X_list]
# Stack into a single sparse matrix
X_sparse = vstack(X_sparse_rows, format='csr')

var_df = pd.DataFrame({'source_symbol': unique_source_genes,
                       'target_symbol': unique_target_genes}, index=unique_source_target)

qfinal = AnnData(X=X_sparse, obs=obs_df, var=var_df)

sc.write('cell_cell_pairs.h5ad', qfinal)
