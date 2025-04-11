import numpy as np
import scipy.sparse as sparse
from scipy.fftpack import dct, idct
from numpy.random import RandomState
from numbers import Integral
from supports import pairwise_l2_distances_with_full, pairwise_l2_distances_with_self
from supports import compute_weighted_first_moment_array, compute_weighted_first_and_second_moment_array
from supports import pairwise_mahalanobis_distances_spherical, pairwise_mahalanobis_distances_diagonal

class SparsifierPy:
    """Pure Python implementation of a high-dimensional data sparsification tool"""

    def __init__(self, num_feat_full: int, num_feat_comp: int, num_samp: int,
                 mask: np.ndarray = None, transform: str = 'dct',
                 D_indices: np.ndarray = None, num_feat_shared: int = 0,
                 random_state: int = None):
        """
        Parameter initialization
        :param num_feat_full: Original feature dimension
        :param num_feat_comp: Compressed feature dimension
        :param num_samp: Number of samples
        :param mask: Custom mask (optional)
        :param transform: Preprocessing transform type ('dct' or None)
        :param D_indices: Sign-flipping indices (optional)
        :param num_feat_shared: Number of shared features
        :param random_state: Random seed
        """
        # Parameter validation
        self._validate_positive_int(num_feat_full, 'num_feat_full')
        self._validate_positive_int(num_samp, 'num_samp')
        self._validate_feature_size(
            num_feat_comp, num_feat_full, 'num_feat_comp')
        self._validate_feature_size(
            num_feat_shared, num_feat_comp, 'num_feat_shared')

        # Core parameters
        self.num_feat_full = num_feat_full
        self.num_feat_comp = num_feat_comp
        self.num_samp = num_samp
        self.num_feat_shared = num_feat_shared

        # Preprocessing configuration
        self.transform = transform.lower() if transform else None
        if self.transform not in [None, 'dct']:
            raise ValueError("transform must be 'dct' or None")

        # Random state
        self.random_state = self._init_random_state(random_state)

        # Generate key components
        self.D_indices = self._generate_D_indices() if D_indices is None else D_indices
        self.mask = self._generate_mask() if mask is None else mask
        self.RHDX = None  # Storage for compressed data

    # --------------------------
    # Initialization helper methods
    # --------------------------
    def _validate_positive_int(self, value, name):
        """Validate positive integer"""
        if not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, actual value: {value}")

    def _validate_feature_size(self, value, upper_bound, name):
        """Validate feature size range"""
        if not (0 <= value <= upper_bound):
            raise ValueError(
                f"{name} must satisfy 0 <= {name} <= {upper_bound}, actual value: {value}")

    def _init_random_state(self, seed):
        """Initialize random state"""
        if isinstance(seed, RandomState):
            return seed
        return RandomState(seed) if seed is not None else np.random

    # --------------------------
    # Core component generation
    # --------------------------
    def _generate_D_indices(self):
        """Generate sign-flipping indices (D matrix)"""
        if self.transform == 'dct':
            return np.array([i for i in range(self.num_feat_full) 
                            if self.random_state.choice([0, 1])])
        return None

    def _generate_mask(self):
        """Generate random mask  """
        rng = self.random_state
        # Select shared indices
        all_indices = list(range(self.num_feat_full))
        rng.shuffle(all_indices)
        shared_mask = all_indices[:self.num_feat_shared]
        # Randomly select remaining indices
        remaining_indices = all_indices[self.num_feat_shared:]
        num_left_to_draw = self.num_feat_comp - self.num_feat_shared
        if num_left_to_draw > 0:
            random_masks = [rng.choice(remaining_indices,
                num_left_to_draw, replace=False) for n in range(self.num_samp)]
            mask = np.concatenate((random_masks,
                np.tile(shared_mask, (self.num_samp, 1)).astype(int)), axis=1)
        else:
            # All dimensions are shared
            mask = np.tile(shared_mask, (self.num_samp, 1)).astype(int)

        mask.sort(axis=1)
        return mask

    # --------------------------
    # Data transformation and mask operations
    # --------------------------
    def apply_HD(self, X: np.ndarray) -> np.ndarray:
        """Apply preprocessing transform HD (DCT + sign flipping)"""
        # Copy data
        Y = np.copy(X)
        # Apply D matrix
        if self.D_indices is not None:
            Y[:, self.D_indices] *= -1
        # Apply H matrix
        if self.transform == 'dct':
            Y = dct(Y, norm='ortho', axis=1, overwrite_x=False)
        return Y

    def invert_HD(self, HDX: np.ndarray) -> np.ndarray:
        """Inverse transform HD^{-1}"""
        X = np.copy(HDX)
        X = idct(X, norm='ortho', axis=1, overwrite_x=False) if self.transform == 'dct' else X
        if self.D_indices is not None:
            X[:, self.D_indices] *= -1
        return X

    def apply_mask(self, X, mask=None):
        """Apply mask  
        
        Parameters
        ----------
        X : np.ndarray, shape(n, P)
        mask : np.ndarray, shape(n, Q), optional
            If None, uses self.mask
            
        Returns
        -------
        RX : np.ndarray, shape(n, Q)
            Masked X. The nth row of RX is X[n][mask[n]].
        """
        if mask is None:
            mask = self.mask
        X_masked = np.array([X[n][mask[n]] for n in range(mask.shape[0])])
        return X_masked

    def invert_mask_bool(self):
        """Generate inverse boolean mask  """
        col_inds = [n for n in range(self.num_samp) for m in range(self.num_feat_comp)]
        row_inds = list(self.mask.flatten())
        data = np.ones_like(row_inds)
        mask_binary = sparse.csr_matrix((data, (row_inds, col_inds)),
                      shape=(self.num_feat_full, self.num_samp), dtype=bool)
        return mask_binary

    # --------------------------
    # Core algorithm operations
    # --------------------------
    def _check_X(self, X):
        """Validate X"""
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("X must be a 2D array.")
        elif X.shape[0] != self.num_samp:
            raise Exception(f"X must have num_samp = {self.num_samp} rows, but has {X.shape[0]} rows.")
        elif X.shape[1] != self.num_feat_full:
            raise Exception(f"X must have num_feat_full = {self.num_feat_full} columns, but has {X.shape[1]} columns.")

    def _check_HDX(self, HDX):
        """Validate HDX"""
        if not isinstance(HDX, np.ndarray) or HDX.ndim != 2:
            raise TypeError("HDX must be a 2D array.")
        elif HDX.shape[0] != self.num_samp:
            raise Exception(f"HDX must have num_samp = {self.num_samp} rows, but has {HDX.shape[0]} rows.")
        elif HDX.shape[1] != self.num_feat_full:
            raise Exception(f"HDX must have num_feat_full = {self.num_feat_full} columns, but has {HDX.shape[1]} columns.")

    def _check_RHDX(self, RHDX):
        """Validate RHDX"""
        if not isinstance(RHDX, np.ndarray) or RHDX.ndim != 2:
            raise TypeError("RHDX must be a 2D array.")
        elif RHDX.shape[0] != self.num_samp:
            raise Exception(f"RHDX must have num_samp = {self.num_samp} rows, but has {RHDX.shape[0]} rows.")
        elif RHDX.shape[1] != self.num_feat_comp:
            raise Exception(f"RHDX must have num_feat_comp = {self.num_feat_comp} columns, but has {RHDX.shape[1]} columns.")

    def _set_RHDX(self, X, HDX, RHDX):
        """Set RHDX  """
        if RHDX is not None:
            return RHDX.astype(float)
        elif HDX is not None:
            return self.apply_mask(HDX.astype(float), self.mask)
        else:
            return self.apply_mask(X.astype(float), self.mask)

    def fit_sparsifier(self, X=None, HDX=None, RHDX=None):
        """Fit compressed data  """
        if HDX is None and RHDX is None:
            self._check_X(X)
            RHDX = self.apply_mask(self.apply_HD(X), self.mask)
        elif HDX is not None and RHDX is None:
            self._check_HDX(HDX)
            RHDX = self.apply_mask(HDX, self.mask)
        elif RHDX is not None:
            self._check_RHDX(RHDX)
        else:
            raise Exception("Must provide at least one data source: X, HDX, or RHDX.")

        self.RHDX = RHDX

    def pairwise_distances(self, Y=None):
        """Compute pairwise distances  """
        if Y is None:
            result = np.zeros((self.num_samp, self.num_samp), dtype=np.float64)
            pairwise_l2_distances_with_self(result, self.RHDX, self.mask,
                self.num_samp, self.num_feat_comp, self.num_feat_full)
        else:
            K = Y.shape[0]
            result = np.zeros((self.num_samp, K), dtype=np.float64)
            pairwise_l2_distances_with_full(result, self.RHDX, Y, self.mask,
                self.num_samp, K, self.num_feat_comp, self.num_feat_full)

        return result

    def weighted_means(self, W):
        """Compute weighted means  """
        K = np.shape(W)[1]
        means = np.zeros((K, self.num_feat_full), dtype=np.float64)
        compute_weighted_first_moment_array(
                               means,
                               self.RHDX,
                               self.mask,
                               W,
                               self.num_samp,
                               K,
                               self.num_feat_comp,
                               self.num_feat_full)
        return means
    
    def weighted_means_and_variances(self, W):
        """Compute weighted means and variances  """
        K = np.shape(W)[1]
        means = np.zeros((K, self.num_feat_full), dtype=np.float64)
        second_moments = np.zeros((K, self.num_feat_full), dtype=np.float64)
        compute_weighted_first_and_second_moment_array(
                               means,
                               second_moments,
                               self.RHDX,
                               self.mask,
                               W,
                               self.num_samp,
                               K,
                               self.num_feat_comp,
                               self.num_feat_full)
        variances = second_moments - means**2
        return [means, variances]
    
    def pairwise_mahalanobis_distances(self, means, covariances, covariance_type):
        """Compute Mahalanobis distances  """
        K = means.shape[0]
        distances = np.zeros((self.num_samp, K), dtype=np.float64)
        if covariance_type == 'spherical':
            pairwise_mahalanobis_distances_spherical(distances,
                                                     self.RHDX,
                                                     means,
                                                     self.mask,
                                                     covariances,
                                                     self.num_samp,
                                                     K,
                                                     self.num_feat_comp,
                                                     self.num_feat_full)
        elif covariance_type == 'diag':
            pairwise_mahalanobis_distances_diagonal(distances,
                                                    self.RHDX,
                                                    means,
                                                    self.mask,
                                                    covariances,
                                                    self.num_samp,
                                                    K,
                                                    self.num_feat_comp,
                                                    self.num_feat_full)
        else:
            raise Exception("covariance_type must be 'spherical' or 'diag'")
        return distances

    def _pick_K_dense_datapoints_kmpp(self, K):
        """Select K data points using kmpp method  """
        rng = self.random_state
        datapoint_indices = np.zeros(K, dtype=int)
        datapoints = np.zeros((K, self.num_feat_full))

        # Randomly select first point
        datapoint_indices[0] = rng.choice(self.num_samp)
        datapoints[0][self.mask[datapoint_indices[0]]] = \
            self.RHDX[datapoint_indices[0]]

        # Initialize distance counter to max float value
        d_prev = np.ones(self.num_samp) * np.finfo(float).max

        # Select remaining k-1 cluster centers
        for k in range(1, K):
            # Calculate squared distance of all data points to the last added cluster
            latest_cluster = datapoints[k-1, np.newaxis]
            d_curr = self.pairwise_distances(Y=latest_cluster)[:, 0]**2
            # ||x - U|| is this distance or the current minimum
            where_we_have_not_improved = np.where(d_curr > d_prev)[0]
            d_curr[where_we_have_not_improved] = d_prev[where_we_have_not_improved]
            d_prev = np.copy(d_curr)

            d_curr_sum = d_curr.sum()

            # If the mask hasn't eliminated all distance information, randomly select a data point with probability proportional to squared distance to current cluster set
            if d_curr_sum > 0:
                datapoint_indices[k] = rng.choice(self.num_samp, p=d_curr/d_curr_sum)
            else:
                # Mask has eliminated all distance information, so just randomly select a point not yet chosen
                available_indices = set(range(self.num_samp)).difference(set(datapoint_indices))
                datapoint_indices[k] = np.random.choice(list(available_indices))
            # Finally, assign cluster
            datapoints[k][self.mask[datapoint_indices[k]]] = \
                self.RHDX[datapoint_indices[k]]

        return [datapoints, datapoint_indices]

    def _pick_K_dense_datapoints_random(self, K):
        """Randomly select K data points """
        # Randomly and uniformly select K data points
        rng = self.random_state
        datapoint_indices = rng.choice(self.num_samp, K, replace=False)
        datapoint_indices.sort()
        datapoints = np.zeros((K, self.num_feat_full))
        for k in range(K):
            datapoints[k][self.mask[datapoint_indices[k]]] = \
                    self.RHDX[datapoint_indices[k]]
        return [datapoints, datapoint_indices]