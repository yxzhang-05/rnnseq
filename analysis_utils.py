import numpy as np
import matplotlib.pyplot as plt
import par_transfer as par
import scipy.cluster.hierarchy as sch
from itertools import combinations
from collections import defaultdict
from sklearn.decomposition import PCA

######################################################
# Radial Pattern Score
######################################################

def radial_pattern_score(latent_points, labels, types, normalize=True, return_components=False):
   
    if len(latent_points) < 3:
        return (0.0, 0.0, 0.0, 0.0) if return_components else 0.0
    
    # Identify center line type (abbb pattern) and plane types
    center_type_idx = None
    for idx, t in enumerate(types):
        # Check if pattern is like 'abbb' (one letter appears 3 times, another appears 1 time)
        letter_counts = {}
        for c in t:
            letter_counts[c] = letter_counts.get(c, 0) + 1
        counts = sorted(letter_counts.values())
        if counts == [1, 3]:  # abbb pattern
            center_type_idx = idx
            break
    
    if center_type_idx is None:
        # Fallback: use type with fewest samples as center
        unique_labels = np.unique(labels)
        label_counts = [(l, np.sum(labels == l)) for l in unique_labels]
        center_type_idx = min(label_counts, key=lambda x: x[1])[0]
    
    # Separate center and plane indices
    center_seq_idx = np.where(labels == center_type_idx)[0].tolist()
    unique_labels = np.unique(labels)
    plane_type_indices = {l: np.where(labels == l)[0].tolist() 
                         for l in unique_labels if l != center_type_idx}
    
    # 1. Planarity - compute for each plane type separately
    # Use PCA explained variance ratio (more stable)
    planarity_scores = []
    for type_idx, indices in plane_type_indices.items():
        if len(indices) < 3:  # Need at least 3 points for meaningful 2D plane
            continue
        X_plane = latent_points[indices, :]
        n_comp = min(2, len(indices), X_plane.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(X_plane)
        # Sum of explained variance of top 2 components = planarity
        plane_score = np.sum(pca.explained_variance_ratio_[:2])
        planarity_scores.append(plane_score)
    P = np.mean(planarity_scores) 

    # 2. Centerline Linear Score
    # Use PCA explained variance ratio for linearity
    X_center = latent_points[center_seq_idx, :]
    if len(center_seq_idx) >= 3:  # Need at least 3 points for meaningful line
        n_comp = min(len(center_seq_idx), X_center.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(X_center)
        # First component explained variance = linearity
        L = pca.explained_variance_ratio_[0]
    else:
        L = 0.0

    # 3. Plane-Centerline Distance (Alignment)
    # Normalize distance by characteristic length scale of data
    if len(center_seq_idx) >= 3 and plane_type_indices:
        center_mean = np.mean(X_center, axis=0)
        pca_center = PCA(n_components=1)
        pca_center.fit(X_center)
        line_vec = pca_center.components_[0]
        
        # Compute characteristic scale: std of all points
        data_scale = np.std(latent_points) + 1e-8
        
        align_scores = []
        for type_idx, indices in plane_type_indices.items():
            if len(indices) < 1:
                continue
            X_plane = latent_points[indices, :]
            plane_center = np.mean(X_plane, axis=0)
            
            # Distance from plane center to line
            vec_to_plane = plane_center - center_mean
            proj_len = np.dot(vec_to_plane, line_vec)
            closest_point = center_mean + proj_len * line_vec
            dist = np.linalg.norm(plane_center - closest_point)
            
            # Normalize by data scale and invert (smaller distance = better)
            normalized_dist = dist / data_scale
            align_score = np.exp(-normalized_dist)  # Exponential decay: 0 dist -> 1, large dist -> 0
            align_scores.append(align_score)
        A = np.mean(align_scores) if align_scores else 0.0
    else:
        A = 0.0
    A = np.clip(A, 0, 1)

    # Overall score
    radial_score = (P + L + A) / 3.0
    
    return radial_score, P, L, A


######################################################
# Dot product of the representations in hidden layer #
######################################################


"""Return the Hamming distance between equal-length sequences."""
def compute_hamming_distance(tokens_set):
    # print(tokens_set)
    D=np.ndarray((par.L, len(tokens_set), len(tokens_set)))

    for k, l in enumerate(range(1,par.L+1)):
        _set = tokens_set[:,:l]
        for i, s1 in enumerate(_set):
            for j, s2 in enumerate(_set):
                if len(s1) != len(s2):
                    raise ValueError("Undefined for sequences of unequal length.")
                else:
                    D[k,i,j] = sum(char1 != char2 for char1, char2 in zip(s1, s2))
    return D

def compute_correlation_matrix(y_hidden, task):
    y_hidden=np.array(y_hidden)
    n_set=len(y_hidden)
    overlap_tokens=np.ndarray((par.cue_size+par.L+par.delay, n_set, n_set))
    
    for idx_token1 in range(n_set):
        # print(idx_token1)
        for idx_token2 in range(n_set):
            # print(idx_token2)
            for idx_t in range(par.cue_size+par.L+par.delay):
                v1 = y_hidden[idx_token1, idx_t, :]
                norm1 = np.sqrt(np.sum(v1*v1))
                v2 = y_hidden[idx_token2, idx_t, :]
                norm2 = np.sqrt(np.sum(v2*v2))
                overlap_tokens[idx_t, idx_token1, idx_token2] = np.dot(v1, v2)/(norm1 * norm2)
    return overlap_tokens

def dimension (VE, threshold=.8):
    '''
    Input
    -----
    VE, np.array (N,) or (M,N)
        fraction of explained variance. If (N,), for a single N-dimensional dataset
        if (M,N,) for M different datasets.
    Output
    ------
    dimensions: np.array (1,) or (M,)
    '''
    _shape = VE.shape
    _VE = VE
    if len(_shape) == 1:
        _VE = np.reshape(VE, (1,*VE.shape))
    above_threshold = _VE > threshold
    dimensions = np.argmax(above_threshold, axis=-1) + 1
    return dimensions

def compute_cosine_similarity (X):
    '''
    X: ndarray (..., p, D)
        p = number of data points
        D = dimension

    output: ndarray (..., p, p)
        Cosine similarity between data points
    '''
    *_shape, p, D = X.shape
    _X = X.reshape(-1, p, D)
    _dot_prod = np.sum(_X[:,None,:,:] * _X[:,:,None,:], axis=-1)
    _norm = np.sqrt(np.sum(_X**2, axis=-1))
    _cs = _dot_prod / (_norm[:,:,None]*_norm[:,None,:])

    _cs = _cs.reshape(*_shape, p, p)

    return _cs

def compute_distance (X):
    '''
    X: ndarray (..., p, D)
        p = number of data points
        D = dimension

    output: ndarray (..., p, p)
        Euclidean distance between data points
    '''
    *_shape, p, D = X.shape
    _X = X.reshape(-1, p, D)
    
    _dot_prod = np.sum(_X[:,None,:,:] * _X[:,:,None,:], axis=-1)
    _norm_sq = np.sum(_X**2, axis=-1)
    _dist = np.sqrt(_norm_sq[:,:,None] + _norm_sq[:,None,:] - 2.*_dot_prod)
    _dist = _dist.reshape(*_shape, p, p)

    return _dist

def compute_umcontent(M, return_triplets=True):
    '''
    M: ndarray (time, seq, seq)
        Matrix of cosine similarities between sequences
        (i.e. between their corresponding hidden activities)
        at every time in the sequence.

    returns tuple of ndarray (time,)
    - d_min / d_max
    - d_med / d_max
    - uc
    '''

    D_combos = np.stack([[sorted([D[i,j], D[j,k], D[i,k]]) for i, j, k in combinations(range(D.shape[0]), 3)] for D in M])
    # D_combos = np.stack([np.sort(np.array([[D[i,j], D[j,k], D[i,k]] for i, j, k in combinations(range(D.shape[0]), 3)]), axis=1) for D in M])

    if return_triplets:
        return D_combos

    # D_combos (time, triplets, 3)
    _log_min_max = np.log( D_combos[:,:,0] / D_combos[:,:,2] )
    _log_med_max = np.log( D_combos[:,:,1] / D_combos[:,:,2] )
    lamda_combos = (_log_min_max - _log_med_max) / (_log_min_max + _log_med_max)
    return np.nanmean(lamda_combos, axis=1)


def compute_overlap_AE(M, tokens, types):
    n_set=len(tokens)
    tokens = np.array([list(token) for token in tokens])

    M_t = np.ndarray((par.cue_size+par.L+par.delay, n_set, n_set))
    sorted_idx_t =  np.ndarray((par.cue_size+par.L+par.delay, n_set))
    sorted_ticklabels_t =  np.empty((par.cue_size+par.L+par.delay, n_set), dtype=object)

    for idx_t in range(par.L):
        
        if par.clustering == 'class':
            idx = np.arange(len(tokens))
            ticklabels = np.array(types)

        elif par.clustering == 'hierarchical':
            idx = np.lexsort([tokens[:, i] for i in range(tokens.shape[1]-1, -1, -1)])
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted.astype(str)]  

        elif par.clustering == 'distance':
            pairwise_distances = sch.distance.pdist(M)
            linkage = sch.linkage(pairwise_distances, method='complete')
            cluster_distance_threshold = pairwise_distances.max()/2
            idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
            idx = np.argsort(idx_to_cluster_array)
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted.astype(str)]  

        elif par.clustering == 'timestep':
            idx = np.argsort(tokens[:,idx_t])
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted]

        elif par.clustering == 'transitions':
            if idx_t == 0:
                tokens = tokens[:,idx_t].reshape(-1,1)
            else:
                tokens = tokens[:,idx_t-1:idx_t+1]
            idx = np.lexsort(tokens.T[::-1])
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted[:,:tokens.shape[1]].astype(str)]  

        M_sorted = np.take(M, idx, axis=0)
        M_sorted = np.take(M_sorted, idx, axis=1)
        M_t[idx_t] = M_sorted
        sorted_idx_t[idx_t] = idx
        sorted_ticklabels_t[idx_t] = ticklabels

    return M_t, sorted_ticklabels_t

def plot_overlap_AE(M_t, ticklabels_t, filename, task, whichmatrix='overlap'):
    fig, ax = plt.subplots(1, par.L, figsize=[10, 2.5])
    ax = ax.ravel()

    for idx_t in range(par.L):
        im=ax[idx_t].imshow(M_t[idx_t])
        
        # this only works with whichclustering=class
        ax[idx_t].set_xticks(np.arange(len(ticklabels_t[idx_t])))
        ax[idx_t].set_xticklabels(ticklabels_t[idx_t], rotation=45, size=4)            
        ax[idx_t].set_yticks(np.arange(len(ticklabels_t[idx_t])))
        ax[idx_t].set_yticklabels(ticklabels_t[idx_t], size=4)            

        ax[idx_t].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    fig.tight_layout()
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(f"{par.folder}/figs/{whichmatrix}_{filename}_whichclustering{par.clustering}_{par.whichset}_task{task}.svg", bbox_inches="tight", dpi=600)


def to_dict(d):
    '''
    Converts a defaultdict with any depth into a standard dictionary
    '''
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    return d

if __name__ == "__main__":

    exit()