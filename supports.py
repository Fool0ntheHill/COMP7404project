import numpy as np


def dist_both_comp(comp_sample_1, comp_sample_2, mask_1, mask_2, num_feat_comp, num_feat_full):
    """
    Calculate the L2 distance between two compressed samples.
    
    Parameters:
        comp_sample_1: First compressed sample, shape (num_feat_comp,)
        comp_sample_2: Second compressed sample, shape (num_feat_comp,)
        mask_1: Mask for the first sample, shape (num_feat_comp,)
        mask_2: Mask for the second sample, shape (num_feat_comp,)
        num_feat_comp: Number of compressed features
        num_feat_full: Number of full features
    
    Returns:
        L2 distance between the two samples
    """
    # Find the intersection of the two masks
    mask_intersection = np.intersect1d(mask_1, mask_2)
    mask_intersection_size = len(mask_intersection)
    
    # Calculate the distance on the intersection
    distance = 0.0
    for i in range(num_feat_comp):
        for j in range(num_feat_comp):
            if mask_1[i] == mask_2[j]:
                diff = comp_sample_1[i] - comp_sample_2[j]
                distance += diff * diff
    
    distance = np.sqrt(distance)
    
    # Apply the correct scaling factor
    if mask_intersection_size > 0:
        distance *= np.sqrt(num_feat_full / mask_intersection_size)
    
    return distance


def dist_one_comp_one_full(comp_sample, full_sample, mask, num_feat_comp, num_feat_full):
    """
    Calculate the L2 distance between a compressed sample and a full sample.
    
    Parameters:
        comp_sample: Compressed sample, shape (num_feat_comp,)
        full_sample: Full sample, shape (num_feat_full,)
        mask: Mask, shape (num_feat_comp,)
        num_feat_comp: Number of compressed features
        num_feat_full: Number of full features
    
    Returns:
        L2 distance between the two samples
    """
    # Directly compute the sum of squared differences
    distance = 0.0
    for i in range(num_feat_comp):
        diff = comp_sample[i] - full_sample[mask[i]]
        distance += diff * diff
    
    # Apply the scaling factor and take the square root
    return np.sqrt(distance) * np.sqrt(num_feat_full / num_feat_comp)


def pairwise_l2_distances_with_self(result, comp_array, mask_array,
                                    num_samples, num_feat_comp, num_feat_full):
    """
    Calculate L2 distances between samples within sparse data (comp_array).
    """
    # Upper triangular matrix
    for i in range(num_samples - 1):
        for j in range(i + 1, num_samples):
            result[i, j] = dist_both_comp(
                comp_array[i], comp_array[j],
                mask_array[i], mask_array[j],
                num_feat_comp, num_feat_full
            )
    
    # Diagonal and lower triangular matrix
    for i in range(num_samples):
        # Diagonal is 0
        result[i, i] = 0
        # Lower triangular mirrors the upper triangular
        for j in range(i):
            result[i, j] = result[j, i]


def pairwise_l2_distances_with_full(result, comp_array, full_array, mask_array,
                                    num_samples_comp, num_samples_full,
                                    num_feat_comp, num_feat_full):
    """
    Calculate L2 distances between sparse data (comp_array) and full-dimensional data (full_array).
    """
    for i in range(num_samples_comp):
        for j in range(num_samples_full):
            result[i, j] = dist_one_comp_one_full(
                comp_array[i], full_array[j],
                mask_array[i], num_feat_comp, num_feat_full
            )


def mahalanobis_distance_spherical(comp_sample, full_mean, mask, spherical_covariance, num_feat_comp, num_feat_full):
    """
    Calculate the Mahalanobis distance (spherical covariance matrix).
    
    Parameters:
    ----------
    comp_sample : np.ndarray, shape (num_feat_comp,)
        Compressed sample vector.
    full_mean : np.ndarray, shape (num_feat_full,)
        Mean vector in the full space.
    mask : np.ndarray, shape (num_feat_comp,)
        Mask for selecting features (boolean or integer array).
    spherical_covariance : float
        Value of the spherical covariance matrix (shared variance across all features).
    num_feat_comp : int
        Number of compressed features.
    num_feat_full : int
        Number of features in the full space.

    Returns:
    -------
    distance : float
        Mahalanobis distance.
    """
    # Check input dimensions
    assert comp_sample.shape == (
        num_feat_comp,), f"comp_sample should have shape ({num_feat_comp},)"
    assert full_mean.shape == (
        num_feat_full,), f"full_mean should have shape ({num_feat_full},)"
    assert mask.shape == (num_feat_comp,), f"mask should have shape ({num_feat_comp},)"

    # Compute the difference
    diff = comp_sample - full_mean[mask]

    # Calculate the Mahalanobis distance
    # 1. Compute the sum of squared differences
    # 2. Divide by the spherical covariance
    # 3. Multiply by the scaling factor num_feat_full/num_feat_comp
    # 4. Take the square root
    distance = np.sqrt((num_feat_full/num_feat_comp) * np.sum(diff ** 2) / spherical_covariance)

    return distance


def mahalanobis_distance_diagonal(comp_sample, full_mean, mask, diagonal_covariance, num_feat_comp, num_feat_full):
    """
    Calculate the Mahalanobis distance (diagonal covariance matrix).

    Parameters:
    ----------
    comp_sample : np.ndarray, shape (num_feat_comp,)
        Compressed sample vector.
    full_mean : np.ndarray, shape (num_feat_full,)
        Mean vector in the full space.
    mask : np.ndarray, shape (num_feat_comp,)
        Mask for selecting features (boolean or integer array).
    diagonal_covariance : np.ndarray, shape (num_feat_full,)
        Diagonal elements of the diagonal covariance matrix.
    num_feat_comp : int
        Number of compressed features.
    num_feat_full : int
        Number of features in the full space.

    Returns:
    -------
    distance : float
        Mahalanobis distance.
    """
    # Check input dimensions
    assert comp_sample.shape == (
        num_feat_comp,), f"comp_sample should have shape ({num_feat_comp},)"
    assert full_mean.shape == (
        num_feat_full,), f"full_mean should have shape ({num_feat_full},)"
    assert mask.shape == (num_feat_comp,), f"mask should have shape ({num_feat_comp},)"
    assert diagonal_covariance.shape == (
        num_feat_full,), f"diagonal_covariance should have shape ({num_feat_full},)"

    # Compute the difference
    diff = comp_sample - full_mean[mask]

    # Extract the covariance elements corresponding to the mask
    masked_covariance = diagonal_covariance[mask]

    # Calculate the Mahalanobis distance
    # 1. Compute the weighted sum of squared differences
    # 2. Multiply by the scaling factor num_feat_full/num_feat_comp
    # 3. Take the square root
    distance = np.sqrt((num_feat_full/num_feat_comp) * np.sum(diff ** 2 / masked_covariance))

    return distance


def pairwise_mahalanobis_distances_spherical(result, comp_array, full_means, mask_array, spherical_covariance_array, num_samples_comp, num_samples_full, num_feat_comp, num_feat_full):
    """Calculate Mahalanobis distances with spherical covariance
    
    Parameters:
    result: Result array, shape (num_samples_comp, num_samples_full)
    comp_array: Compressed data, shape (num_samples_comp, num_feat_comp)
    full_means: Full means, shape (num_samples_full, num_feat_full)
    mask_array: Mask array, shape (num_samples_comp, num_feat_comp)
    spherical_covariance_array: Spherical covariance, shape (num_samples_full,)
    """
    for i in range(num_samples_comp):
        for j in range(num_samples_full):
            # When calculating the distance, only use the features corresponding to the mask
            distance = 0.0
            sigma_squared = spherical_covariance_array[j]
            
            for k in range(num_feat_comp):
                # Get the original feature index
                feat_idx = mask_array[i, k]
                # Compute the difference for this feature
                diff = comp_array[i, k] - full_means[j, feat_idx]
                # Accumulate the squared difference
                distance += diff * diff
            
            # Divide by the variance to get the Mahalanobis distance
            # Modification: Apply the scaling factor num_feat_full/num_feat_comp
            result[i, j] = np.sqrt((num_feat_full/num_feat_comp) * distance / sigma_squared)


def pairwise_mahalanobis_distances_diagonal(result, comp_array, full_means, mask_array, diagonal_covariance_array, num_samples_comp, num_samples_full, num_feat_comp, num_feat_full):
    """Calculate Mahalanobis distances with diagonal covariance
    
    Parameters:
    result: Result array, shape (num_samples_comp, num_samples_full)
    comp_array: Compressed data, shape (num_samples_comp, num_feat_comp)
    full_means: Full means, shape (num_samples_full, num_feat_full)
    mask_array: Mask array, shape (num_samples_comp, num_feat_comp)
    diagonal_covariance_array: Diagonal covariance, shape (num_samples_full, num_feat_full)
    """
    for i in range(num_samples_comp):
        for j in range(num_samples_full):
            # When calculating the distance, only use the features corresponding to the mask
            distance = 0.0
            
            for k in range(num_feat_comp):
                # Get the original feature index
                feat_idx = mask_array[i, k]
                # Compute the difference for this feature
                diff = comp_array[i, k] - full_means[j, feat_idx]
                # Use the covariance corresponding to this feature
                cov = diagonal_covariance_array[j, feat_idx]
                # Accumulate the weighted squared difference
                distance += (diff * diff) / cov
            
            # Mahalanobis distance
            result[i, j] = np.sqrt(distance)


def apply_mask_to_full_sample(result, full_sample, mask, num_feat_comp):
    """
    Apply the mask to a full sample to extract the corresponding features
    
    Parameters:
        result: Output array, shape (num_feat_comp,)
        full_sample: Full sample, shape (num_feat_full,)
        mask: Mask, shape (num_feat_comp,)
        num_feat_comp: Number of compressed features
    """
    for i in range(num_feat_comp):
        result[i] = full_sample[mask[i]]


def logdet_cov_diag(result, cov_array, mask, num_samp_comp, num_cov, num_feat_comp, num_feat_full):
    """
    Calculate the log determinant of a diagonal covariance matrix
    
    Parameters:
        result: Output array, shape (num_samp_comp, num_cov)
        cov_array: Covariance array, shape (num_cov, num_feat_full)
        mask: Mask array, shape (num_samp_comp, num_feat_comp)
        num_samp_comp: Number of samples
        num_cov: Number of covariance matrices
        num_feat_comp: Number of compressed features
        num_feat_full: Number of full features
    """
    for i in range(num_samp_comp):
        for j in range(num_cov):
            # Only use the covariance elements specified by the mask
            logdet = 0.0
            for k in range(num_feat_comp):
                feat_idx = mask[i, k]
                logdet += np.log(cov_array[j, feat_idx])
            result[i, j] = logdet


def update_weighted_first_moment(first_moment_to_update, normalizer_to_update, comp_sample, mask, weight, num_feat_comp, num_feat_full):
    """
    Update the weighted first moment (mean)
    
    Parameters:
        first_moment_to_update: First moment to update, shape (num_feat_full,)
        normalizer_to_update: Normalizer to update, shape (num_feat_full,)
        comp_sample: Compressed sample, shape (num_feat_comp,)
        mask: Mask, shape (num_feat_comp,)
        weight: Weight
        num_feat_comp: Number of compressed features
        num_feat_full: Number of full features
    """
    for i in range(num_feat_comp):
        feat_idx = mask[i]
        first_moment_to_update[feat_idx] += weight * comp_sample[i]
        normalizer_to_update[feat_idx] += weight


def update_weighted_first_moment_array(first_moment_array, normalizer_array, comp_sample, mask, weights, num_samp_full, num_feat_comp, num_feat_full):
    """
    Update a set of weighted first moments (means)
    
    Parameters:
        first_moment_array: First moment array to update, shape (num_samp_full, num_feat_full)
        normalizer_array: Normalizer array to update, shape (num_samp_full, num_feat_full)
        comp_sample: Compressed sample, shape (num_feat_comp,)
        mask: Mask, shape (num_feat_comp,)
        weights: Weight array, shape (num_samp_full,)
        num_samp_full: Number of full samples
        num_feat_comp: Number of compressed features
        num_feat_full: Number of full features
    """
    for i in range(num_samp_full):
        if weights[i] > 0:
            for j in range(num_feat_comp):
                feat_idx = mask[j]
                first_moment_array[i, feat_idx] += weights[i] * comp_sample[j]
                normalizer_array[i, feat_idx] += weights[i]


def compute_weighted_first_moment_array(means, comp_array, mask_array, weights, 
                                       num_samples, num_clusters, num_feat_comp, num_feat_full):
    """
    Calculate the weighted first moment (mean)
    
    Parameters:
        means: Output array, shape (num_clusters, num_feat_full)
        comp_array: Sparse data, shape (num_samples, num_feat_comp)
        mask_array: Mask array, shape (num_samples, num_feat_comp)
        weights: Weight matrix, shape (num_samples, num_clusters)
        num_samples: Number of samples
        num_clusters: Number of clusters
        num_feat_comp: Feature dimension of sparse data
        num_feat_full: Feature dimension of full-dimensional data
    """
    # Initialize the result array
    means.fill(0.0)
    
    # Initialize the normalizer array to track the sum of weights for each feature dimension
    normalizer = np.zeros((num_clusters, num_feat_full))
    
    # For each sample and each cluster
    for i in range(num_samples):
        for k in range(num_clusters):
            if weights[i, k] > 0:
                # For each feature
                for j in range(num_feat_comp):
                    feat_idx = mask_array[i, j]
                    means[k, feat_idx] += weights[i, k] * comp_array[i, j]
                    normalizer[k, feat_idx] += weights[i, k]
    
    # Normalize by feature dimension
    GREATER_THAN_ZERO_TOL = 1e-16
    for k in range(num_clusters):
        for j in range(num_feat_full):
            if normalizer[k, j] > GREATER_THAN_ZERO_TOL:
                means[k, j] /= normalizer[k, j]
            else:
                means[k, j] = 0.0


def compute_weighted_first_and_second_moment_array(means, second_moments, comp_array, mask_array, weights,
                                                 num_samples, num_clusters, num_feat_comp, num_feat_full):
    """
    Calculate the weighted first moment (mean) and second moment
    
    Parameters:
        means: Output mean array, shape (num_clusters, num_feat_full)
        second_moments: Output second moment array, shape (num_clusters, num_feat_full)
        comp_array: Sparse data, shape (num_samples, num_feat_comp)
        mask_array: Mask array, shape (num_samples, num_feat_comp)
        weights: Weight matrix, shape (num_samples, num_clusters)
        num_samples: Number of samples
        num_clusters: Number of clusters
        num_feat_comp: Feature dimension of sparse data
        num_feat_full: Feature dimension of full-dimensional data
    """
    # Initialize the result arrays
    means.fill(0.0)
    second_moments.fill(0.0)
    
    # Initialize the normalizer array to track the sum of weights for each feature dimension
    normalizer = np.zeros((num_clusters, num_feat_full))
    
    # For each sample and each cluster
    for i in range(num_samples):
        for k in range(num_clusters):
            if weights[i, k] > 0:
                # For each feature
                for j in range(num_feat_comp):
                    feat_idx = mask_array[i, j]
                    value = comp_array[i, j]
                    means[k, feat_idx] += weights[i, k] * value
                    second_moments[k, feat_idx] += weights[i, k] * (value ** 2)
                    normalizer[k, feat_idx] += weights[i, k]
    
    # Normalize by feature dimension
    GREATER_THAN_ZERO_TOL = 1e-16
    for k in range(num_clusters):
        for j in range(num_feat_full):
            if normalizer[k, j] > GREATER_THAN_ZERO_TOL:
                means[k, j] /= normalizer[k, j]
                second_moments[k, j] /= normalizer[k, j]
            else:
                means[k, j] = 0.0
                second_moments[k, j] = 0.0
