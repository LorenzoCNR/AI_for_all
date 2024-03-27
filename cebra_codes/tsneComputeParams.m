function   params=tsneComputeParams()

     
params = struct('n_components',                 2,             ...
                'perplexity',                   30,            ...        
                'learning_rate',                'auto',        ...       
                'early_exaggeration_iter',      250,           ... 
                'early_exaggeration',           'auto',        ... 
                'n_iter',                       500,           ...   
                'dof',                          1,             ...   
                'theta',                        0.5,           ...
                'n_interpolation_points',       3,             ...
                'min_num_intervals',            50,            ... 
                'ints_in_interval',             1,             ...
                'initialization',               'pca',         ...     
                'metric',                       'euclidean',   ...   
                'initial_momentum',             0.8,           ...
                'final_momentum',               0.8,           ... 
                'max_step_norm',                5,             ... 
                'n_jobs',                       1,             ... 
                'neighbors',                    'auto',        ...       
                'negative_gradient_method',     'auto',        ...
                'callbacks_every_iters',         50,           ...        
                'verbose',                      'True'         ...
                );

                %'exaggeration',                None,          ...  
                %'metric_params',               None,          ...
                %'max_grad_norm',               None,          ...
                %'callbacks',                   None,          ...  
                %'random_state',                None,          ...
 

params.exec                 = true;
params.script_fit           = 'wrap_tsne_fit.py';  % script to be executed in python (full path)
params.script_input_dir     = './';                 % directory where script expects inputs
params.script_output_dir    = './';                 % directory where script save outputs
params.InField              ='spikes';

%  Parameters: (default values)

%- Perplexity: (30)
% determining the scale of neighborhood considered for each point. . 
% Technically this parameter determines the  effective width (i.e. variance)
% of the gaussian   in computing the affinity probability in the 
% HIGH DIMENSIONAL SPACE.
% Considered loosely, it can be thought of as the balance between 
% preserving the global and the local structure of the data; the higher
% perplexity the more is preserved the global structure
% Perplexity can be thought of as the continuous `k` number of
% nearest neighbors, for which t-SNE will attempt to preserve distances.
% 
%   n_components: int (1)
%         The dimension of the embedding space. This deafults to 2 for easy
%         visualization, but sometimes 1 is used for t-SNE heatmaps. t-SNE is
%         not designed to embed into higher dimension and please note that
%         acceleration schemes break down and are not fully implemented.     
% 
%     learning_rate: Union[str, float] ("auto")
%         The learning rate for t-SNE optimization. When ``learning_rate="auto"``
%         the appropriate learning rate is selected according to N / exaggeration
%         Note that
%         this will result in a different learning rate during the early
%         exaggeration phase and afterwards. This should *not* be used when 
%         adding samples into existing embeddings, where the learning rate often
%         needs to be much lower to obtain convergence.
% 
%     early_exaggeration_iter: int (250)
%         The number of iterations to run in the *early exaggeration* phase.
% 
%     early_exaggeration: Union[str, float] ("auto")
%         The exaggeration factor to use during the *early exaggeration* phase.
%         Typical values range from 4 to 32. When ``early_exaggeration="auto"``
%         early exaggeration factor defaults to 12, unless desired subsequent
%         exaggeration is higher, i.e.: ``early_exaggeration = max(12,
%         exaggeration)``.
% 
%     n_iter: int (500)
%         The number of iterations to run in the normal optimization regime.
% 
%     exaggeration: float (None)
%         The exaggeration factor to use during the normal optimization phase.
%         This can be used to form more densely packed clusters and is useful
%         for large data sets.
% 
%     dof: float (1)
%         Degrees of freedom as described in Kobak et al. "Heavy-tailed kernels
%         reveal a finer cluster structure in t-SNE visualisations", 2019.
%  
%     theta: float (0.5)
%         Only used when ``negative_gradient_method="bh"`` or its other aliases.
%         This is the trade-off parameter between speed and accuracy of the tree
%         approximation method. Typical values range from 0.2 to 0.8. The value 0
%         indicates that no approximation is to be made and produces exact results
%         also producing longer runtime. Alternatively, you can use ``auto`` to
%         approximately select the faster method.
% 
%     n_interpolation_points: int (3)
%         Only used when ``negative_gradient_method="fft"`` or its other aliases.
%         The number of interpolation points to use within each grid cell for
%         interpolation based t-SNE. It is highly recommended leaving this value
%         at the default 3.
% 
%     min_num_intervals: int (50)
%         Only used when ``negative_gradient_method="fft"`` or its other aliases.
%         The minimum number of grid cells to use, regardless of the
%         ``ints_in_interval`` parameter. Higher values provide more accurate
%         gradient estimations.
% 
%     ints_in_interval: float (1)
%         Only used when ``negative_gradient_method="fft"`` or its other aliases.
%         Indicates how large a grid cell should be e.g. a value of 3 indicates a
%         grid side length of 3. Lower values provide more accurate gradient
%         estimations.
% 
%     initialization: Union[np.ndarray, str] ("pca")
%         The initial point positions to be used in the embedding space. Can be a
%         precomputed numpy array, ``pca``, ``spectral`` or ``random``. Please
%         note that when passing in a precomputed positions, it is highly
%         recommended that the point positions have small variance
%         (std(Y) < 0.0001), otherwise you may get poor embeddings.
% 
%     metric: Union[str, Callable] ("euclidean")
%         The metric to be used to compute affinities between points in the
%         original space.
% 
%     metric_params: dict (None)
%         Additional keyword arguments for the metric function.
% 
%     initial_momentum: float (0.8)
%         The momentum to use during the *early exaggeration* phase.
% 
%     final_momentum: float (0.8)
%         The momentum to use during the normal optimization phase.
% 
%     max_grad_norm: float (None)
%         Maximum gradient norm. If the norm exceeds this value, it will be
%         clipped. This is most beneficial when adding points into an existing
%         embedding and the new points overlap with the reference points,
%         leading to large gradients. This can make points "shoot off" from
%         the embedding, causing the interpolation method to compute a very
%         large grid, and leads to worse results.
% 
%     max_step_norm: float (5)
%         Maximum update norm. If the norm exceeds this value, it will be
%         clipped. This prevents points from "shooting off" from
%         the embedding.
% 
%     n_jobs: int (1)
%         The number of threads to use while running t-SNE. This follows the
%         scikit-learn convention, ``-1`` meaning all processors, ``-2`` meaning
%         all but one, etc.
% 
%     neighbors: str ("auto")
%         Specifies the nearest neighbor method to use. Can be ``exact``, ``annoy``,
%         ``pynndescent``, ``hnsw``, ``approx``, or ``auto`` (default). ``approx`` uses Annoy
%         if the input data matrix is not a sparse object and if Annoy supports
%         the given metric. Otherwise it uses Pynndescent. ``auto`` uses exact
%         nearest neighbors for N<1000 and the same heuristic as ``approx`` for N>=1000.
% 
%     negative_gradient_method: str ("auto")
%         Specifies the negative gradient approximation method to use. For smaller
%         data sets, the Barnes-Hut approximation is appropriate and can be set
%         using one of the following aliases: ``bh``, ``BH`` or ``barnes-hut``.
%         For larger data sets, the FFT accelerated interpolation method is more
%         appropriate and can be set using one of the following aliases: ``fft``,
%         ``FFT`` or ``ìnterpolation``. Alternatively, you can use ``auto`` to
%         approximately select the faster method.
% 
%     callbacks: Union[Callable, List[Callable]] (None)
%         Callbacks, which will be run every ``callbacks_every_iters`` iterations.
% 
%     callbacks_every_iters: int (50)
%         How many iterations should pass between each time the callbacks are
%         invoked.
% 
%     random_state: Union[int, RandomState] (None)
%         If the value is an int, random_state is the seed used by the random
%         number generator. If the value is a RandomState instance, then it will
%         be used as the random number generator. If the value is None, the random
%         number generator is the RandomState instance used by `np.random`.
% 
%     verbose: bool (False)


%%% Note on t-SNE. 

% Ref:  Visualizing Data using t-SNE (Van der Maaten, Hinton 2008) 
%       Stochastic Neighbor Embedding (Hinton, Howeis, 2002)

% it-SNE (t-Distributed Stochastic Neighbor Embedding) is a powerful
% technique designed to visualize high-dimensional data by mapping it 
% onto a two- or three-dimensional space. This method, building upon
% Stochastic Neighbor Embedding (SNE), focuses on maintaining the local
% structure of data while also revealing global patterns like clusters
% at multiple scales. In particular t-sne is more efficienti in the
% optimization procedure for the gradient are simpler to compute; a t-
% distribution is used rather than a Gaussina in the low dimensional space
% to alleviate both the crowding and the optimizaztion problem of SNE
% , thus producing clearer visualizations of high-dimensional data in
% lower-dimensional space
% Joint Probabilities: Instead of conditional probabilities, t-SNE computes 
% joint probabilities p_ij for pairs of points in the high-dimensional
% space and q_ij in the low-dimensional space. 
% This symmetrical approach simplifies the gradients and the optimization process.
% Heavy-Tailed Distribution in Low-Dimensional Space: t-SNE employs a
% Student t-distribution (specifically, a Cauchy distribution with one
% degree of freedom) for the low-dimensional similarities. 
% This choice allows moderately distant points in the high-dimensional
% space to be mapped to larger distances in the low-dimensional map, 
% effectively addressing the crowding problem.
% Optimization Goal: Similar to SNE, t-SNE minimizes the mismatch between
% high-dimensional and low-dimensional joint probabilities, but using the
% symmetrized KL divergence. The optimization is less susceptible to local 
% minima and does not require simulated annealing.



