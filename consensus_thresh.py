import numpy as np

def fcn_group_bins(adj,dist,hemiid,nbins):
    '''
    fcn_distance_dependent_threshold(A,dist,hemiid,frac) generates a
    group-representative structural connectivity matrix by preserving
    within-/between-hemisphere connection length distributions.
    All rights reserved to Richard Betzel, Indiana University, 2018
    Matlab to Python: Gidon Levakov

    If you use this code, please cite:
    Betzel, R. F., Griffa, A., Hagmann, P., & Miic, B. (2018).
    Distance-dependent consensus thresholds for generating
    group-representative structural brain networks.
    Network Neuroscience, 1-22.

    :param adj: [node x node x subject] structural connectivity matrices, ndarray
    :param dist: [node x node] distance matrix, ndarray
    :param hemiid: indicator vector for left (False) and right (True) hemispheres, ndarray
    :param nbins: number of distance bins, int
    :return:
        G: group matrix (binary) with distance-based consensus, ndarray
        Gc: group matrix (binary) with traditional consistency-based thresholding, ndarray
    '''
    assert adj.shape[0] == adj.shape[1], 'Input matrix must be square, and input shape: [node x node x subject]'
    if hemiid.ndim == 1:
        hemiid = hemiid[:, np.newaxis]
    distbins = np.linspace(np.min(dist[np.nonzero(dist)]), np.max(dist[np.nonzero(dist)]), nbins + 1)
    distbins[-1] += 1
    n, nsub = adj.shape[0], adj.shape[-1] # number nodes(n) and subjects(nsub)
    C = np.sum(adj > 0, axis=2) # consistency
    W = np.sum(adj, axis=2) / C #  average weight
    W[np.isnan(W)] = 0 # remove nans
    Grp = np.zeros((n, n, 2)) # for storing inter / intra hemispheric connections (we do these separately)
    Gc = Grp.copy()
    inter_hemispheric_mask = np.dot(hemiid, ~hemiid.T)
    inter_hemispheric_mask = np.logical_or(inter_hemispheric_mask, inter_hemispheric_mask.T)
    for j in range(2):
        if j: # inter-hemispheric edge mask
            inter_hemi = ~inter_hemispheric_mask
        else:
            inter_hemi = inter_hemispheric_mask
        m = dist * inter_hemi
        D = (adj > 0) * (dist * np.triu(inter_hemi))[..., np.newaxis]
        D = D[np.nonzero(D)]
        tgt = len(D) / nsub # mean number of edges per subject
        G = np.zeros((n,n))
        for i_bin in range(nbins):
            mask = np.where(np.triu((m >= distbins[i_bin]) & (m < distbins[i_bin + 1]), 1))
            frac = int(np.round(tgt * np.sum((D >= distbins[i_bin]) & (D < distbins[i_bin + 1])) / len(D)))
            c = C[mask]
            idx = np.argsort(c)[::-1] # descend
            G[mask[0][idx[:frac]], mask[1][idx[:frac]]] = 1
        Grp[:,:, j] = G
        I = np.where(np.triu(inter_hemi, 1))
        w = W[I]
        idx = np.argsort(w)[::-1]
        w = np.zeros((n,n))
        nnz = int(G.sum())
        w[I[0][idx[:nnz]], I[1][idx[:nnz]]] = 1
        Gc[:,:, j] = w
    G = np.sum(Grp, 2)
    G = G + G.T
    Gc = np.sum(Gc, 2)
    Gc = Gc + Gc.T
    return G, Gc
