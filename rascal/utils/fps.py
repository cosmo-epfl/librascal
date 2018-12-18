from ..lib import sparsification

def fps(feature_matrix, n_select, starting_index=None,
        method='simple', restart = None):
    """
    Farthest Point Sampling [1] routine using librascal.

    Parameters
    ----------
    feature_matrix : numpy.ndarray[float64[n, m], flags.c_contiguous]
        Feature matrix with n samples and m features.
    n_select : int
        Number of sample to select
    starting_index : int, optional
        Index of the first sample to select (the default is None,
                           which corresponds to starting_index == 0)
    method : str, optional
        Select which kind of FPS selection to perform:
        + 'simple' is the basic fps selection [1].
        + 'voronoi' uses voronoi polyhedra to avoid some
                    distance computations.

    Returns
    -------
    sparse_indices : numpy.ndarray[int[n_select]]
        Selected indices refering to the feature_matrix order.
    sparse_minmax_d2 : numpy.ndarray[float64[n_select]]
        MIN-MAX distance at each step in the selection
    lmin_d2: numpy.ndarray[float64[n]]
        array of Hausdorff distances between the n points and the
        n_select FPS points



    .. [1] Ceriotti, M., Tribello, G. A., & Parrinello, M. (2013).
        Demonstrating the Transferability and the Descriptive Power of Sketch-Map.
        Journal of Chemical Theory and Computation, 9(3), 1521–1532.
        https://doi.org/10.1021/ct3010563
    """

    if starting_index is None:
        starting_index = 0


    if method == 'simple':
        if restart is None:
            sparse_indices,sparse_minmax_d2,lmin_d2 = \
             sparsification.fps(feature_matrix,n_select,starting_index)
        else:
            sparse_indices,sparse_minmax_d2,lmin_d2 = \
             sparsification.fps(feature_matrix,n_select,
              starting_index, restart)
        print("restart indices", lmin_d2[0])

    elif method == 'voronoi':
        sparse_indices, sparse_minmax_d2, \
            voronoi_indices, voronoi_r2 = \
            sparsification.fps_voronoi(feature_matrix,
                n_select,starting_index)

    else:
        raise Exception('Unknown FPS algorithm {}'.format(method))

    return sparse_indices,sparse_minmax_d2,lmin_d2

