function   params=umapComputeParams()
% function params=cebraComputeParams()



%% model_type: hypothesis (che include anche hybrid 
%% quando setti hybrid = True) 
% discovery,
% shuffle





params = struct('n_neighbors',                                 15, ...  
                'n_components',                                 2, ...  
                'metric',                             'euclidean', ...  
                'metric_kwds',                             'None', ...  
                'output_metric',                      'euclidean', ...  
                'output_metric_kwds',                      'None', ...  
                'n_epochs',                                'None', ...  
                'learning_rate',                              1.0, ...  
                'init',                                'spectral', ...  
                'min_dist',                                   0.1, ...  
                'spread',                                     1.0, ...  
                'low_memory',                              'True', ...  
                'n_jobs',                                      -1, ...  
                'set_op_mix_ratio',                           1.0, ...  
                'local_connectivity',                         1.0, ...  
                'repulsion_strength',                         1.0, ...  
                'negative_sample_rate',                         5, ...  
                'transform_queue_size',                       4.0, ...  
                'a',                                       'None', ...  
                'b',                                       'None', ...  
                'random_state',                            'None', ...  
                'angular_rp_forest',                      'False', ...  
                'target_n_neighbors',                          -1, ...  
                'target_metric',                    'categorical', ...  
                'target_metric_kwds',                      'None', ...  
                'target_weight',                              0.5, ...  
                'transform_seed',                              42, ...  
                'transform_mode',                     'embedding', ...  
                'force_approximation_algorithm',          'False', ...  
                'verbose',                                'False', ...  
                'tqdm_kwds',                               'None', ...  
                'unique',                                 'False', ...  
                'densmap',                                'False', ...  
                'dens_lambda',                                2.0, ...  
                'dens_frac',                                  0.3, ...  
                'dens_var_shift',                             0.1, ...  
                'output_dens',                             'False', ...  
                'disconnection_distance',                  'None', ...  
                'precomputed_knn',             '(None, None, None)')


params.exec                 = true;
params.script_fit           = 'wrap_umap_fit.py';  % script to be executed in python (full path)
params.script_input_dir     = './';                 % directory where script expects inputs
params.script_output_dir    = './';                 % directory where script save outputs
params.InField              ='spikes';
    
% 
% Umap tries to find  a low dimensional embedding of the data that approximates
%     an underlying manifold.
% 
%     Parameters (4 main parameters are on top of list)
%     ----------
%     n_neighbors: float (optional, default 15)
%         The size of local neighborhood (in terms of number of neighboring
%         sample points) used for manifold approximation. Larger values
%         result in more global views of the manifold. Substantially 
%         This parameter controls how UMAP balances local versus global structure in the data. 
%         In general values should be in the range 2 to 100.
% 
%     n_components: int (optional, default 2)
%         The dimension of the space to embed into. This defaults to 2 to
%         provide easy visualization (can
%     metric: string or function (optional, default 'euclidean')
%         The metric to use to compute distances in high dimensional space.
%         If a string is passed it must match a valid predefined metric. If
%         a general metric is required a function that takes two 1d arrays and
%         returns a float can be provided. For performance purposes it is
%         required that this be a numba jit'd function. Valid string metrics
%         include:
%        Minkowski style metrics
%         * euclidean
%         * manhattan
%         * chebyshev
%         * minkowski
%       Miscellaneous spatial metrics
%         * canberra
%         * braycurtis
%         * haversine
%       Normalized spatial metrics
%         * mahalanobis
%         * wminkowski
%         * seuclidean
%       Angular and correlation metrics
%         * cosine
%         * correlation
%       Metrics for binary data
%         * hamming
%         * jaccard
%         * dice
%         * russelrao
%         * kulsinski
%         * ll_dirichlet
%         * hellinger
%         * rogerstanimoto
%         * sokalmichener
%         * sokalsneath
%         * yule
% 
%         Metrics that take arguments (such as minkowski, mahalanobis etc.)
%         can have arguments passed via the metric_kwds dictionary. At this
%         time care must be taken and dictionary elements must be ordered
%         appropriately; this will hopefully be fixed in the future.

%     min_dist: float (optional, default 0.1)
%         The effective minimum distance between embedded points. Smaller values
%         will result in a more clustered/clumped embedding where nearby points
%         on the manifold are drawn closer together, while larger values will
%         result on a more even dispersal of points. The value should be set
%         relative to the ``spread`` value, which determines the scale at which
%         embedded points will be spread out.
% 
%     n_epochs: int (optional, default None)
%         The number of training epochs to be used in optimizing the
%         low dimensional embedding. Larger values result in more accurate
%         embeddings. If None is specified a value will be selected based on
%         the size of the input dataset (200 for large datasets, 500 for small).
% 
%     learning_rate: float (optional, default 1.0)
%         The initial learning rate for the embedding optimization.
% 
%     init: string (optional, default 'spectral')
%         How to initialize the low dimensional embedding. Options are:
% 
%             * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
%             * 'random': assign initial embedding positions at random.
%             * 'pca': use the first n_components from PCA applied to the
%                 input data.
%             * 'tswspectral': use a spectral embedding of the fuzzy
%                 1-skeleton, using a truncated singular value decomposition to
%                 "warm" up the eigensolver. This is intended as an alternative
%                 to the 'spectral' method, if that takes an  excessively long
%                 time to complete initialization (or fails to complete).
%             * A numpy array of initial embedding positions.
% 
%     min_dist: float (optional, default 0.1)
%         The effective minimum distance between embedded points. Smaller values
%         will result in a more clustered/clumped embedding where nearby points
%         on the manifold are drawn closer together, while larger values will
%         result on a more even dispersal of points. The value should be set
%         relative to the ``spread`` value, which determines the scale at which
%         embedded points will be spread out.
% 
%     spread: float (optional, default 1.0)
%         The effective scale of embedded points. In combination with ``min_dist``
%         this determines how clustered/clumped the embedded points are.
% 
%     low_memory: bool (optional, default True)
%         For some datasets the nearest neighbor computation can consume a lot of
%         memory. If you find that UMAP is failing due to memory constraints
%         consider setting this option to True. This approach is more
%         computationally expensive, but avoids excessive memory use.
% 
%     set_op_mix_ratio: float (optional, default 1.0)
%         Interpolate between (fuzzy) union and intersection as the set operation
%         used to combine local fuzzy simplicial sets to obtain a global fuzzy
%         simplicial sets. Both fuzzy set operations use the product t-norm.
%         The value of this parameter should be between 0.0 and 1.0; a value of
%         1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
%         intersection.
% 
%     local_connectivity: int (optional, default 1)
%         The local connectivity required -- i.e. the number of nearest
%         neighbors that should be assumed to be connected at a local level.
%         The higher this value the more connected the manifold becomes
%         locally. In practice this should be not more than the local intrinsic
%         dimension of the manifold.
% 
%     repulsion_strength: float (optional, default 1.0)
%         Weighting applied to negative samples in low dimensional embedding
%         optimization. Values higher than one will result in greater weight
%         being given to negative samples.
% 
%     negative_sample_rate: int (optional, default 5)
%         The number of negative samples to select per positive sample
%         in the optimization process. Increasing this value will result
%         in greater repulsive force being applied, greater optimization
%         cost, but slightly more accuracy.
% 
%     transform_queue_size: float (optional, default 4.0)
%         For transform operations (embedding new points using a trained model
%         this will control how aggressively to search for nearest neighbors.
%         Larger values will result in slower performance but more accurate
%         nearest neighbor evaluation.
% 
%     a: float (optional, default None)
%         More specific parameters controlling the embedding. If None these
%         values are set automatically as determined by ``min_dist`` and
%         ``spread``.
%     b: float (optional, default None)
%         More specific parameters controlling the embedding. If None these
%         values are set automatically as determined by ``min_dist`` and
%         ``spread``.
% 
%     random_state: int, RandomState instance or None, optional (default: None)
%         If int, random_state is the seed used by the random number generator;
%         If RandomState instance, random_state is the random number generator;
%         If None, the random number generator is the RandomState instance used
%         by `np.random`.
% 
%     metric_kwds: dict (optional, default None)
%         Arguments to pass on to the metric, such as the ``p`` value for
%         Minkowski distance. If None then no arguments are passed on.
% 
%     angular_rp_forest: bool (optional, default False)
%         Whether to use an angular random projection forest to initialise
%         the approximate nearest neighbor search. This can be faster, but is
%         mostly only useful for a metric that uses an angular style distance such
%         as cosine, correlation etc. In the case of those metrics angular forests
%         will be chosen automatically.
% 
%     target_n_neighbors: int (optional, default -1)
%         The number of nearest neighbors to use to construct the target simplicial
%         set. If set to -1 use the ``n_neighbors`` value.
% 
%     target_metric: string or callable (optional, default 'categorical')
%         The metric used to measure distance for a target array is using supervised
%         dimension reduction. By default this is 'categorical' which will measure
%         distance in terms of whether categories match or are different. Furthermore,
%         if semi-supervised is required target values of -1 will be trated as
%         unlabelled under the 'categorical' metric. If the target array takes
%         continuous values (e.g. for a regression problem) then metric of 'l1'
%         or 'l2' is probably more appropriate.
% 
%     target_metric_kwds: dict (optional, default None)
%         Keyword argument to pass to the target metric when performing
%         supervised dimension reduction. If None then no arguments are passed on.
% 
%     target_weight: float (optional, default 0.5)
%         weighting factor between data topology and target topology. A value of
%         0.0 weights predominantly on data, a value of 1.0 places a strong emphasis on
%         target. The default of 0.5 balances the weighting equally between data and
%         target.
% 
%     transform_seed: int (optional, default 42)
%         Random seed used for the stochastic aspects of the transform operation.
%         This ensures consistency in transform operations.
% 
%     verbose: bool (optional, default False)
%         Controls verbosity of logging.
% 
%     tqdm_kwds: dict (optional, defaul None)
%         Key word arguments to be used by the tqdm progress bar.
% 
%     unique: bool (optional, default False)
%         Controls if the rows of your data should be uniqued before being
%         embedded.  If you have more duplicates than you have ``n_neighbors``
%         you can have the identical data points lying in different regions of
%         your space.  It also violates the definition of a metric.
%         For to map from internal structures back to your data use the variable
%         _unique_inverse_.
% 
%     densmap: bool (optional, default False)
%         Specifies whether the density-augmented objective of densMAP
%         should be used for optimization. Turning on this option generates
%         an embedding where the local densities are encouraged to be correlated
%         with those in the original space. Parameters below with the prefix 'dens'
%         further control the behavior of this extension.
% 
%     dens_lambda: float (optional, default 2.0)
%         Controls the regularization weight of the density correlation term
%         in densMAP. Higher values prioritize density preservation over the
%         UMAP objective, and vice versa for values closer to zero. Setting this
%         parameter to zero is equivalent to running the original UMAP algorithm.
% 
%     dens_frac: float (optional, default 0.3)
%         Controls the fraction of epochs (between 0 and 1) where the
%         density-augmented objective is used in densMAP. The first
%         (1 - dens_frac) fraction of epochs optimize the original UMAP objective
%         before introducing the density correlation term.
% 
%     dens_var_shift: float (optional, default 0.1)
%         A small constant added to the variance of local radii in the
%         embedding when calculating the density correlation objective to
%         prevent numerical instability from dividing by a small number
% 
%     output_dens: float (optional, default False)
%         Determines whether the local radii of the final embedding (an inverse
%         measure of local density) are computed and returned in addition to
%         the embedding. If set to True, local radii of the original data
%         are also included in the output for comparison; the output is a tuple
%         (embedding, original local radii, embedding local radii). This option
%         can also be used when densmap=False to calculate the densities for
%         UMAP embeddings.
% 
%     disconnection_distance: float (optional, default np.inf or maximal value for bounded distances)
%         Disconnect any vertices of distance greater than or equal to disconnection_distance when approximating the
%         manifold via our k-nn graph. This is particularly useful in the case that you have a bounded metric.  The
%         UMAP assumption that we have a connected manifold can be problematic when you have points that are maximally
%         different from all the rest of your data.  The connected manifold assumption will make such points have perfect
%         similarity to a random set of other points.  Too many such points will artificially connect your space.
% 
%     precomputed_knn: tuple (optional, default (None,None,None))
%         If the k-nearest neighbors of each point has already been calculated you
%         can pass them in here to save computation time. The number of nearest
%         neighbors in the precomputed_knn must be greater or equal to the
%         n_neighbors parameter. This should be a tuple containing the output
%         of the nearest_neighbors() function or attributes from a previously fit
%         UMAP object; (knn_indices, knn_dists, knn_search_index). If you wish to use
%         k-nearest neighbors data calculated by another package then provide a tuple of
%         the form (knn_indices, knn_dists). The contents of the tuple should be two numpy
%         arrays of shape (N, n_neighbors) where N is the number of items in the
%         input data. The first array should be the integer indices of the nearest
%         neighbors, and the second array should be the corresponding distances. The
%         nearest neighbor of each item should be itself, e.g. the nearest neighbor of
%         item 0 should be 0, the nearest neighbor of item 1 is 1 and so on. Please note
%         that you will *not* be able to transform new data in this case.
%     """

% UMAP, short for Uniform Manifold Approximation and Projection, is a method 
% designed to simplify the complexity of high-dimensional data, making it
% understandable in lower-dimensional spaces, typically two or three dimensions.
% Technically:

% Understanding Data Structure: UMAP starts by looking at each data point in
% the high-dimensional space and understanding how it relates to its neighbors.
% It assumes that all the data points are spread out on a manifold,a matehmetical 
% concept which refers to a space that might be curved or twisted in complex
% ways but looks flat if you zoom in closely enough.

%Building a Network: For each point, UMAP builds a network (or graph) 
% of connections to its nearest neighbors. This network is designed to 
% capture the local and global structure of the data. By focusing on how 
% each point is related to its closest neighbors, UMAP preserves the local
% structure while also considering how these local structures fit together 
% on a larger scale.

% Projecting into Lower Dimensions: Once UMAP has a good understanding of
% the overall structure of the data based on these networks, it tries to 
% lay out a similar network in a lower-dimensional space (like a flat map 
% of the curved surface of the Earth). The goal is to place each point and 
% its neighbors as close as possible to their original configuration in 
% the high-dimensional space.

%Fine-tuning the Layout: The initial layout is then fine-tuned through 
% an optimization process. UMAP iteratively adjusts the positions of the
% points in the lower-dimensional space to make the network of connections 
% as similar as possible to the original high-dimensional network.
% This involves making sure that points that were close together in the 
% high-dimensional space remain close in the lower-dimensional projection,
% while points that were far apart do not end up misleadingly close.

%The beauty of UMAP lies in its ability to  trade off the preservation of 
% local data structures (how each point relates to its immediate neighbors)
% with the overall global data structure (how these neighborhoods connect 
% to form the big picture). This makes UMAP particularly useful for 
% visualizing clusters or groups within the data, uncovering patterns, 
% and revealing relationships that might not be apparent in the 
% high-dimensional space.
