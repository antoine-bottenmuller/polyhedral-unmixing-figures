import random
import itertools
import numpy as np
from scipy import ndimage, io
from scipy.optimize import linear_sum_assignment
from typing import Tuple, List, Literal, Optional, Callable
from sklearn.metrics import accuracy_score, precision_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import svm

from unmixing_min_norm_point_PYTHON import to_half_space_couples, keep_only_necessary_couples
from unmixing_min_norm_point_PYTHON import norm, normed, scalar, data_in_polyhedron

from unmixing_min_norm_point_PYTHON import minimum_norm_points_to_polyhedra_PYTHON # PYTHON VERSION
#from src.min_norm_point import minimum_norm_points_to_polyhedra # C VERSION

#%%
# Array normalization and standardization functions
###

def normalized(
        array:np.ndarray, 
        axis:Optional[tuple]=None, 
        output_range:Optional[tuple]=None, 
        output_dtype:Optional[np.dtype]=None
) -> np.ndarray:
    if output_dtype is None:
        if np.issubdtype(array.dtype, np.integer):
            output_dtype = np.float64
        else:
            output_dtype = array.dtype
    if output_range is None:
        if np.issubdtype(output_dtype, np.integer):
            output_range = (np.iinfo(output_dtype).min, np.iinfo(output_dtype).max)
        else:
            output_range = (0., 1.)
    if axis is None:
        axis = tuple(np.arange(array.ndim, dtype=int))
    a_min = array.min(axis, keepdims=True)
    delta = array.max(axis, keepdims=True) - a_min
    diff_zero = delta < np.finfo(array.dtype).resolution
    float_out = (output_range[1] - output_range[0]) * ~diff_zero / (delta * ~diff_zero + diff_zero)
    float_out = (array - a_min) * float_out + output_range[0]
    if np.issubdtype(output_dtype, np.integer):
        return np.round(float_out).astype(output_dtype)
    elif float_out.dtype != output_dtype:
        return float_out.astype(output_dtype)
    return float_out

def standardized(
        array:np.ndarray, 
        axis:Optional[tuple]=None, 
        std_prop:Optional[float]=None, 
        output_dtype:Optional[np.dtype]=None
) -> np.ndarray:
    if output_dtype is None:
        if np.issubdtype(array.dtype, np.integer):
            output_dtype = np.float64
        else:
            output_dtype = array.dtype
    std = np.std(array, axis=axis, keepdims=True)
    if std_prop is not None and std_prop != 1:
        std = std * std_prop + (1 - std_prop)
    mean = np.mean(array, axis=axis, keepdims=True)
    std_zero = std < np.finfo(array.dtype).resolution
    float_out = (array - mean) * ~std_zero / (std * ~std_zero + std_zero)
    if np.issubdtype(output_dtype, np.integer):
        return np.round(float_out).astype(output_dtype)
    elif float_out.dtype != output_dtype:
        return float_out.astype(output_dtype)
    return float_out

#%%
# Import MATLAB image and/or mask with wavelength sensitivity calibration
###

def import_mlab_image(
        image_path:str, 
        image_key:str='ref', 
        calib_path:Optional[str]=None, 
        calib_split:Optional[str]=None, 
        dtype:Optional[np.dtype]=None, 
        mask_key:Optional[str]=None
) -> Tuple[np.ndarray,np.ndarray]|np.ndarray|None:
    """
    Imports and calibrates image from a MATLAB .mat file located at 'image_path' 
    under the key 'image_key' regarding calibration file located at 'calib_path'
    (Optional) with a mask under the key 'mask_key' (Optional).
    """
    data = io.loadmat(image_path)
    if mask_key is not None:
        if mask_key not in data.keys():
            print(f"Warning: mask_key '{mask_key}' not in image file's keys. Null mask returned.")
            print(f"Keys:\n{data.keys()}")
            mask = None
        else:
            mask = data[mask_key]
            if type(mask) is not np.ndarray:
                try:
                    mask = np.asarray(mask).astype(bool)
                except:
                    print(f"Warning: mask object of key '{mask_key}' in image file is not a Numpy array, but a {type(mask)}. Null mask returned.")
                    mask = None
            else:
                mask = mask.astype(bool)
    if image_key not in data.keys():
        print(f"Warning: image_key '{image_key}' not in image file's keys. Null image returned.")
        print(f"Keys:\n{data.keys()}")
        if mask_key is None:
            return None
        return None, mask
    img = data[image_key]
    if type(img) is not np.ndarray:
        try:
            img = np.asarray(img)
        except:
            print(f"Warning: image object of key '{image_key}' in image file is not a Numpy array, but a {type(img)}. Null image returned.")
            if mask_key is None:
                return None
            return None, mask
    if calib_path is None:
        if mask_key is None:
            if dtype is None or img.dtype == dtype:
                return img
            return img.astype(dtype)
        if dtype is None or img.dtype == dtype:
            return img, mask
        return img.astype(dtype), mask
    if calib_split is None:
        calib_split = ' '
    calib = open(calib_path, 'r').read().split(calib_split)
    sensi = []
    for c in calib:
        if len(c) > 0:
            try:
                s = float(c)
                sensi.append(s)
            except:
                continue
    if len(sensi) != img.shape[-1]:
        print("Warning: calibration file must contain as many values as the number of channels in image. Original image returned.")
        if mask_key is None:
            if dtype is None or img.dtype == dtype:
                return img
            return img.astype(dtype)
        if dtype is None or img.dtype == dtype:
            return img, mask
        return img.astype(dtype), mask
    sensi = np.asarray(sensi, dtype=img.dtype)
    if dtype is None:
        dtype = img.dtype
    img = img / sensi
    if np.issubdtype(dtype, np.floating):
        if img.dtype != dtype:
            img = img.astype(dtype)
    else:
        img = np.round(img).astype(dtype)
    if mask_key is None:
        return img
    return img, mask

#%%
# Pre-processing image functions
###

def to_diameter(radius:int|float) -> int|float:
    return 2*radius+1

def to_radius(diameter:int|float) -> int|float:
    res = (diameter-1)/2
    if int(res) == res:
        return int(res)
    return res

def binary_ball(diameter:int|float, ndim:int=2, dtype:type=bool) -> np.ndarray:
    radius = to_radius(diameter)
    x = np.arange(diameter)-radius
    xs = [x]*ndim
    grids = np.meshgrid(*xs)
    distance = np.sum(np.asarray(grids)**2, axis=0)
    d = distance <= radius**2 + (radius-int(radius))**2
    if np.sum(d)==0 and np.prod(d.shape)>0:
        return np.ones(d.shape, dtype=dtype)
    return d.astype(dtype)

def channel_preserved_alternating_sequential_filter(
        input:np.ndarray, 
        n:int, 
        increment:int=1, 
        M_or_N:str='M', 
        se_axis:Optional[int]=None, 
        out:Optional[np.ndarray]=None
) -> np.ndarray:
    """
    Special Alternating Sequential Filter algorithm which preserves channels.\n
    Parameters:
    * input: n-dimensional array to process ;
    * n: maximum radius of the structuring element, reached at the last loop ;
    * increment: value of loop incrementation on SE's radius (multiple of 0.5) ;
    * M_or_N: if 'M', the ASF starts with closing, otherwise ('N') by opening ;
    * se_axis (Optional): if none, the SE is a ball, otherwise a line on given axis ;
    * out (Optional): output array in which the computed data is stored.
    """
    if out is None:
        out = np.empty_like(input)
    out[:] = input[:]
    nbi = int(np.ceil(n/increment))
    if input.dtype == bool:
        if M_or_N == 'M':
            for i in range(1,nbi+1):
                size = to_diameter(min(int(i*increment), n))
                if se_axis is None:
                    selem = binary_ball(size, ndim=input.ndim-1, dtype=bool)[..., np.newaxis]
                else:
                    shape = [1 if i!=se_axis else size for i in range(input.ndim)]
                    selem = np.ones(shape, dtype=bool)
                ndimage.binary_closing(out, structure=selem, output=out, border_value=0)
                ndimage.binary_opening(out, structure=selem, output=out, border_value=0)
        else:
            for i in range(1,nbi+1):
                size = to_diameter(min(int(i*increment), n))
                if se_axis is None:
                    selem = binary_ball(size, ndim=input.ndim-1, dtype=bool)[..., np.newaxis]
                else:
                    shape = [1 if i!=se_axis else size for i in range(input.ndim)]
                    selem = np.ones(shape, dtype=bool)
                ndimage.binary_opening(out, structure=selem, output=out, border_value=0)
                ndimage.binary_closing(out, structure=selem, output=out, border_value=0)
    else:
        if M_or_N == 'M':
            for i in range(1,nbi+1):
                size = to_diameter(min(int(i*increment), n))
                if se_axis is None:
                    selem = binary_ball(size, ndim=input.ndim-1, dtype=bool)[..., np.newaxis]
                else:
                    shape = [1 if i!=se_axis else size for i in range(input.ndim)]
                    selem = np.ones(shape, dtype=bool)
                ndimage.grey_closing(out, footprint=selem, output=out, mode="reflect")
                ndimage.grey_opening(out, footprint=selem, output=out, mode="reflect")
        else:
            for i in range(1,nbi+1):
                size = to_diameter(min(int(i*increment), n))
                if se_axis is None:
                    selem = binary_ball(size, ndim=input.ndim-1, dtype=bool)[..., np.newaxis]
                else:
                    shape = [1 if i!=se_axis else size for i in range(input.ndim)]
                    selem = np.ones(shape, dtype=bool)
                ndimage.grey_opening(out, footprint=selem, output=out, mode="reflect")
                ndimage.grey_closing(out, footprint=selem, output=out, mode="reflect")
    return out

def preprocess_image(
        image:np.ndarray, 
        mask:Optional[np.ndarray]=None, 
        crop:Optional[tuple]=None, 
        denoize_image_radius:bool|int=False, 
        homogenize_luminance:bool|float=False, 
        standardize_channels_beforePCA:bool|float=False, 
        ndim_PCA_reduction:Optional[int]=None, 
        standardize_channels_afterPCA:bool|float=False, 
        standardize_globally:bool=False
) -> Tuple[np.ndarray,np.ndarray]|np.ndarray:
    """
    Function to pre-process a spectral image, regarding a mask or not.
    * image: spectral image (..., n_channels) ;
    * mask: mask of spectral image (...,) ;
    * crop: tuple of crops along image dimensions of size len(...) or len(...)+1 ;
    * denoize_image_radius: boolean or integer for radius of the strcutruring element 
     for ASF (denoizing) applied on image ;
    * homogenize_luminance: boolean or floating for proportion of the luminance 
     removal from image (luminance computed pixel-by-pixel along the channel axis), 
     a negative value allows blurring luminance factor on image support ;
    * ndim_PCA_reduction: integer for the number of dimensions kept after PCA applied 
     on image for channel-dimension reduction ;
    * standardize_channels: boolean or floating for proportion of the channel 
     standardization (recommended if no PCA applied).\n
    Returns pre-processed image, with corresponding mask or not.
    """
    
    if type(image) is not np.ndarray:
        raise ValueError("Parameter 'mask' must be a ndarray")
    if image.ndim < 2:
        raise ValueError("Parameter 'image' must be of ndim >= 2, the last dim being for channels")
    
    if mask is not None:
        if type(mask) is not np.ndarray:
            raise ValueError("Parameter 'mask' must be a ndarray")
        if mask.ndim != image.ndim-1:
            raise ValueError("Parameter 'mask' must be of ndim image.ndim-1")

    # crop image (+ mask)
    if crop is not None:
        if len(crop) not in {image.ndim-1, image.ndim}:
            raise ValueError("Parameter 'crop' must be of size image.ndim-1 or image.ndim")
        slices = ()
        for c in crop:
            if c is None:
                slices += (slice(None),)
            elif type(c) is slice:
                slices += (c,)
            else:
                slices += (slice(*c),)
        image = image[slices]
        if mask is not None:
            if len(slices) == image.ndim:
                slices = slices[:-1]
            mask = mask[slices]

    # denoize image with Alternate Sequential Filter
    if denoize_image_radius is not None and denoize_image_radius != False:
        radius = int(np.ceil(float(denoize_image_radius)))
        image = channel_preserved_alternating_sequential_filter(image, n=radius, out=image)

    # homogenize luminance
    if homogenize_luminance is not None and homogenize_luminance != False:
        prop = min(1, max(0, float(homogenize_luminance)) + int(float(homogenize_luminance)<0))
        fluz = max(0, -float(homogenize_luminance))
        axes = tuple(np.arange(image.ndim-1, dtype=int))
        luminance = norm(ndimage.gaussian_filter(image, sigma=fluz, axes=axes), keepdims=True)
        luminance_zero = luminance < np.finfo(luminance.dtype).resolution
        noLuminance_image = image * ~luminance_zero / (luminance * ~luminance_zero + luminance_zero)
        image = image * (1 - prop) + noLuminance_image * prop

    # standardize channels
    if standardize_channels_beforePCA is not None and standardize_channels_beforePCA != False:
        prop = float(standardize_channels_beforePCA)
        axes = tuple(np.arange(image.ndim-1, dtype=int))
        image = standardized(image, axis=axes, std_prop=prop)

    # reduce channel dimensionality with PCA
    if ndim_PCA_reduction is not None:
        if ndim_PCA_reduction > image.shape[-1] or ndim_PCA_reduction < 0:
            ndim_PCA_reduction = image.shape[-1]
        if mask is None:
            temp_mask = np.ones(shape=image.shape[:-1], dtype=bool)
        else:
            temp_mask = mask
        n_components = int(ndim_PCA_reduction)
        reducted_data = PCA(n_components=n_components).fit_transform(image[temp_mask])
        image = np.zeros(shape=image.shape[:-1]+reducted_data.shape[-1:], dtype=reducted_data.dtype)
        image[temp_mask] = reducted_data
    
    # standardize channels
    if standardize_channels_afterPCA is not None and standardize_channels_afterPCA != False:
        prop = float(standardize_channels_afterPCA)
        axes = tuple(np.arange(image.ndim-1, dtype=int))
        image = standardized(image, axis=axes, std_prop=prop)
    
    # globally standardize (preserves directional std)
    if standardize_globally is not None and standardize_globally:
        image = image - np.mean(image, axis=tuple(np.arange(image.ndim-1, dtype=int)), keepdims=True)
        image = standardized(image)

    if mask is not None:
        return image, mask
    return image

#%%
# Computation of Gaussian parameters from hand-given class data interpreted as a rectangle window from p1 to p2 on image
###

def get_data_from_box_coord(img:np.ndarray, p1:tuple, p2:tuple) -> float|np.ndarray:
    """
    Array 'img' must be of dim len(p) (if grayscale image, only 1 feature), 
    or len(p)+1 with the last axis for features (if multiple channels).\n
    p1 is a corner point of the box on 'img' support, p2 is the opposite one.
    """
    coords = tuple([slice(int(min(p1[i],p2[i])),int(max(p1[i],p2[i]))) for i in range(len(p1))])
    window = img[coords]
    dvalue = window.reshape(np.prod(window.shape[:len(p1)]),np.prod(window.shape[len(p1):]))
    return dvalue

def get_mean_from_box_coord(img:np.ndarray, p1:tuple, p2:tuple) -> float|np.ndarray:
    """
    Array 'img' must be of dim len(p) (if grayscale image, only 1 feature), 
    or len(p)+1 with the last axis for features (if multiple channels).\n
    p1 is a corner point of the box on 'img' support, p2 is the opposite one.
    """
    coords = tuple([slice(int(min(p1[i],p2[i])),int(max(p1[i],p2[i]))) for i in range(len(p1))])
    window = img[coords]
    mvalue = np.mean(window.reshape(np.prod(window.shape[:len(p1)]),np.prod(window.shape[len(p1):])), axis=0)
    return mvalue

def get_covar_from_box_coord(img:np.ndarray, p1:tuple, p2:tuple) -> float|np.ndarray:
    """
    Array 'img' must be of dim len(p) (if grayscale image, only 1 feature), 
    or len(p)+1 with the last axis for features (if multiple channels).\n
    p1 is a corner point of the box on 'img' support, p2 is the opposite one.
    """
    coords = tuple([slice(int(min(p1[i],p2[i])),int(max(p1[i],p2[i]))) for i in range(len(p1))])
    window = img[coords]
    cvalue = np.cov(window.reshape(np.prod(window.shape[:len(p1)]),np.prod(window.shape[len(p1):])).T)
    return cvalue

#%%
# Distance uniformization function
###

def to_min_2D_vec(v:np.ndarray) -> np.ndarray:
    if v.ndim == 0:
        return v[np.newaxis, np.newaxis]
    elif v.ndim == 1:
        return v[np.newaxis]
    return v

def orthonormalize(v:np.ndarray, copy:bool=False, eps:float=1e-15) -> np.ndarray:
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    REFERENCE: Anmol Kabra
    Github: gist.github.com/anmolkabra/
    File: gram_schmidt.py
    """
    new_basis = v.copy() if copy else v
    for i in range(v.shape[0]):
        prev_basis = new_basis[0:i]
        coeff_vec = np.dot(prev_basis, new_basis[i].T)
        new_basis[i] -= np.dot(coeff_vec, prev_basis).T
        if norm(new_basis[i]) < eps:
            new_basis[i][new_basis[i] < eps] = 0.
        else:
            new_basis[i] /= norm(new_basis[i])
    return new_basis

def uniformize_data(data:np.ndarray, references:np.ndarray, orthonormalize_matrix:bool=False, project_in_reference_space:bool=True, infos:bool=True) -> np.ndarray:
    """
    Uniformizes data on references.
    * data: list of k points of size n, with shape (k, n) ;
    * references: list of m points of size n, with shape (m, n) ;
    * orthonormalize_matrix: bool ;
    * project_in_reference_space: bool.\n
    The min(m,n) 'raw' (if m>=n) XOR 'column' (if m<n) elements in references must be linearly independant!\n
    Booleans 'orthonormalize_matrix' and 'project_in_reference_space' cannot be True at the same time (the first one is ignored if m>=n).
    """
    # convert arrays into usual ones
    c = to_min_2D_vec(references)
    p = to_min_2D_vec(data)

    # get float resolution of base Matrix
    if np.issubdtype(c.dtype, np.floating):
        res = np.finfo(c.dtype).resolution
    else:
        res = np.finfo(np.float_).resolution
    
    # if asked, project data on references
    if project_in_reference_space:
        if infos:
            print("Data projected in orthonormalized references' space")
        ref_base = orthonormalize(c.copy(), eps=res)
        c = scalar(c[:, np.newaxis], ref_base)
        p = scalar(p[:, np.newaxis], ref_base)
    
    # sizes of references
    m, n = c.shape
    dim = max(m, n)
    
    # build base Matrix from references
    Mat = np.eye(dim, dtype=c.dtype)
    Mat[:n, :m] = c.T #= (c/norm(c, keepdims=True)).T
    
    # if asked, Gram-Schmidt process on Matrix
    if orthonormalize_matrix and dim > m:
        if infos:
            print("Base Matrix orthonormalization applied along added unit vectors")
        orth_Mat = orthonormalize(Mat.copy().T, eps=res).T
        Mat[:, m:] = orth_Mat[:, m:]

    # if Matrix is NOT invertible, non-linear transform
    Mat_det = np.linalg.det(Mat)
    if np.abs(Mat_det) < res:
        if infos:
            print("Base Matrix built from 'references' is not invertible: non-linear transform applied")
        proba = norm(c - p[:, np.newaxis])
        proba = classical_distance_to_probability(proba)
        new_p = np.sum(c * proba[..., np.newaxis], axis=1)
        return new_p

    # if Matrix is invertible, compute its inverse
    Mat_inv = np.linalg.inv(Mat)

    # two cases: m <= n (linear transform possible) and m > n (not possible)
    if m <= n:
        # Linear transform
        if infos:
            print("=> Linear transform")
        new_p = scalar(p[:, np.newaxis], Mat_inv) #* np.abs(p) / np.max(np.abs(p), axis=0)
    else:
        # Non-linear transform
        if infos:
            print("=> Non-linear transform")
        proba = norm(p[:, np.newaxis] - c)
        proba = classical_distance_to_probability(proba)[:, n:]
        extended_p = np.append(p, proba, axis=1)
        new_p = scalar(extended_p[:, np.newaxis], Mat_inv)
    
    # Affine transform
    #u = 1 / m * np.append([1]*m, [0]*(dim-m), axis=0)
    #new_p = new_p - u
    
    return new_p

#%%
# Functions to get endmembers (class extrema) from their distance to polyhedral classes 
# (are used as references to uniformize data in 'uniformize_data')
###

def get_extrema_arg(distances:np.ndarray, n_elements_per_class:int=1) -> np.ndarray:
    """
    * distances: ndarray (n_data, n_classes) ;
    * n_elements_per_class: int = 1.
    """
    if n_elements_per_class == 1:
        return np.argmin(distances, axis=0)[np.newaxis]
    return np.argsort(distances, axis=0)[:n_elements_per_class]

def get_extrema_val(distances:np.ndarray, data:Optional[np.ndarray]=None, n_elements_per_class:int=1, element_mixing_func:Callable=np.mean) -> np.ndarray:
    """
    * distances: ndarray (n_data, n_classes) ;
    * data (Optional): ndarray (n_data, ndim) ;
    * n_elements_per_class: int = 1 ;
    * element_mixing_func: Callable with arguments (array, axis).\n
    If data is None, returns distance values, otherwise returns data values.
    """
    arg_extrema = get_extrema_arg(distances, n_elements_per_class)
    if data is None:
        return element_mixing_func(distances[arg_extrema], 0)
    return element_mixing_func(data[arg_extrema], 0)

#%%
# Functions to switch from classes to class' halfspaces or hyperplanes
###

def from_n_classes_to_n_hyperplanes(n_classes:int) -> int:
    return int(np.round(n_classes*(n_classes-1)/2))

def from_n_hyperplanes_to_n_classes(n_hyperplanes:int) -> int:
    return int(np.round((1+np.sqrt(8*n_hyperplanes))/2))

def from_n_classes_to_n_half_spaces(n_classes:int) -> int:
    return int(np.round(n_classes*(n_classes-1)))

def distribute_half_spaces(h:np.ndarray, means:np.ndarray) -> np.ndarray:
    """
    h: (n_hyperplanes, 2, ndim)
    means: (n_classes, ndim)
    """
    n_classes = means.shape[0] # = from_n_hyperplanes_to_n_classes(h.shape[0])
    couple_id = list(itertools.combinations(np.arange(n_classes), 2))
    new_h = np.empty(shape=(n_classes,n_classes-1)+h.shape[1:], dtype=h.dtype)
    for k in range(h.shape[0]):
        i, j = couple_id[k]
        c, v = h[k]
        v_ij = means[j] - means[i]
        v_i = v * np.sign(scalar(v, v_ij))
        v_j = v * np.sign(scalar(v,-v_ij))
        h_i = np.asarray((c, v_i))
        h_j = np.asarray((c, v_j))
        new_h[i,j-1] = h_i
        new_h[j,i] = h_j
    return new_h

#%%
# Personalized biased and unbiased linear SVM
###

import itertools
from typing import Tuple
from cvxopt import matrix, solvers

# Biased linear SVM solver
def linear_svm(X:np.ndarray, y:np.ndarray, C:float=1.0, verbose:bool=False) -> Tuple[np.ndarray, float]:
    """
    Simple biased linear SVM solver.
    * X: (n_samples, n_features), floats
    * y: (n_sample,), only -1.0 or 1.0 floats
    * C: tolerance cte, non-negative float\n
    Returns non-normed w, b. 
    To norm w and b, devide them both by norm(w).
    """
    if len(set(np.unique(y)) - {-1,1}) > 0:
        print("Warning: y contains other values than -1 or 1.")
    if verbose is False:
        solvers.options['show_progress'] = False

    n_samples, n_features = X.shape

    # Matrices pour la QP
    P = matrix(np.block([
        [np.eye(n_features), np.zeros((n_features, n_samples + 1))],
        [np.zeros((n_samples + 1, n_features + n_samples + 1))]
    ]), tc='d')

    q = matrix(np.hstack([np.zeros(n_features + 1), C * np.ones(n_samples)]), tc='d')

    # Contraintes Gx <= h
    G_std = np.hstack([-y[:, None] * X, -y[:, None], -np.eye(n_samples)])
    G_slack = np.hstack([np.zeros((n_samples, n_features + 1)), -np.eye(n_samples)])
    G = matrix(np.vstack([G_std, G_slack]), tc='d')

    h = matrix(np.hstack([-np.ones(n_samples), np.zeros(n_samples)]), tc='d')

    # Resolution du probleme quadratique
    sol = solvers.qp(P, q, G, h)
    sol_x = np.array(sol['x']).flatten()

    w = sol_x[:n_features]
    b = sol_x[n_features]
    return w, b

# Unbiased linear SVM solver
def unbiased_linear_svm(X:np.ndarray, y:np.ndarray, C:float=1.0, verbose:bool=False) -> np.ndarray:
    """
    Unbiased linear SVM solver.
    * X: (n_samples, n_features), floats
    * y: (n_sample,), only -1.0 or 1.0 floats
    * C: tolerance cte, non-negative float\n
    Returns non-normed w. 
    To norm w, devide it by norm(w).
    """
    if len(set(np.unique(y)) - {-1,1}) > 0:
        print("Warning: y contains other values than -1 or 1.")
    if verbose is False:
        solvers.options['show_progress'] = False

    n_samples, n_features = X.shape

    # Matrices pour la QP
    P = matrix(np.block([
        [np.eye(n_features), np.zeros((n_features, n_samples))],
        [np.zeros((n_samples, n_features + n_samples))]
    ]), tc='d')

    q = matrix(np.hstack([np.zeros(n_features), C * np.ones(n_samples)]), tc='d')

    # Contraintes Gx <= h
    G_std = np.hstack([-y[:, None] * X, -np.eye(n_samples)])
    G_slack = np.hstack([np.zeros((n_samples, n_features)), -np.eye(n_samples)])
    G = matrix(np.vstack([G_std, G_slack]), tc='d')

    h = matrix(np.hstack([-np.ones(n_samples), np.zeros(n_samples)]), tc='d')

    # Resolution du probleme quadratique
    sol = solvers.qp(P, q, G, h)
    w = np.array(sol['x'][:n_features]).flatten()
    return w

# OvO multiclass biased linear SVM solver
def ovo_linear_svm(X:np.ndarray, y:np.ndarray, C:float=1.0, verbose:bool=False) -> Tuple[np.ndarray, float]:
    """
    One-VS-One multiclass unbiased linear SVM solver.
    * X: (n_samples, n_features), floats
    * y: (n_sample,), only integer floats
    * C: tolerance cte, non-negative float\n
    Returns non-normed W, b. 
    To norm W and b, devide them both by norm(W)[...,None].
    """
    couples = list(itertools.combinations(np.unique(y), 2))
    results = [
        linear_svm(
            X = np.vstack([X[y==c[0]], X[y==c[1]]]), 
            y = np.concatenate([
                np.full(np.sum(y==c[0]), -1, dtype=X.dtype), 
                np.full(np.sum(y==c[1]), +1, dtype=X.dtype)
            ]), 
            C = C, 
            verbose = verbose
        ) for c in couples
    ]
    return np.asarray([r[0] for r in results]), np.asarray([r[1] for r in results])

# OvO multiclass unbiased linear SVM solver
def ovo_unbiased_linear_svm(X:np.ndarray, y:np.ndarray, C:float=1.0, verbose:bool=False) -> np.ndarray:
    """
    One-VS-One multiclass unbiased linear SVM solver.
    * X: (n_samples, n_features), floats
    * y: (n_sample,), only integer floats
    * C: tolerance cte, non-negative float\n
    Returns non-normed W. 
    To norm W, devide it by norm(W)[...,None].
    """
    couples = list(itertools.combinations(np.unique(y), 2))
    return np.asarray([
        unbiased_linear_svm(
            X = np.vstack([X[y==c[0]], X[y==c[1]]]), 
            y = np.concatenate([
                np.full(np.sum(y==c[0]), -1, dtype=X.dtype), 
                np.full(np.sum(y==c[1]), +1, dtype=X.dtype)
            ]), 
            C = C, 
            verbose = verbose
        ) for c in couples
    ])

#%%
# Functions to compute Polyhedral Space Partitioning! 
# Four methods to compute separation hyperplanes: 
# -> 0. using GMM to labellise data, then unbiased SVM to get frontier hyperplanes [unsupervised] ;
# -> 1. using GMM to labellise data, then   biased SVM to get frontier hyperplanes [unsupervised] ; 
# -> 2. using SVM only (to get frontier hyperplanes) on given class samples [supervised] ; 
# -> 3. using k-means to find centroids, then Voronoi diagram (hyperplanes) [unsupervised].
###

# 0. Zeroth method: GMM then unbiased SVM
def class_polyhedra_GMM_unbiasedSVM(
        data:np.ndarray, 
        init:int|np.ndarray=2, 
        n_init:int=1, 
        remove_unnecessary_couples:bool=True, 
        infos:bool=True
) -> list[np.ndarray]:
    """
    Computing 'n' polyhedron classes expressed by the intersection of half-spaces, 
    using a Gaussian Mixture Model of 'n' mixture components on input 'data', and 
    separating classes by hyperplanes computed from linear Support Vector Machine.\n
    Returns list_of_class_polyhedra.
    """
    if type(init) is np.ndarray:
        n = init.shape[0]
        init_means = init
    else:
        n = int(np.round(init))
        init_means = None
    if infos:
        print("* Fitting GMM model on data...", end=' ')
    gmm = GaussianMixture(n_components=n, covariance_type='full', init_params='kmeans', n_init=n_init, means_init=init_means).fit(X=data)
    if infos:
        print("Done!")
        print("* Predicting data on GMM model...", end=' ')
    GMM_labels = gmm.predict(X=data)
    gmm_means = gmm.means_
    if infos:
        print("Done!")
        print("* Computing SVM parameters...", end=' ')
    w = ovo_unbiased_linear_svm(X=data, y=GMM_labels)
    h_hyperplanes = to_half_space_couples(w, np.zeros(w.shape[:-1], dtype=w.dtype))
    h_half_spaces = list(distribute_half_spaces(h_hyperplanes, gmm_means))
    if remove_unnecessary_couples:
        if infos:
            print("Done!")
            print("* Computing minimum H-descriptions...", end=' ')
        for _ in range(len(h_half_spaces)):
            h0 = h_half_spaces.pop(0)
            hp = keep_only_necessary_couples(h0)
            h_half_spaces.append(hp)
    if infos:
        print("Done!")
    return h_half_spaces

# 1. First method: GMM then SVM
def class_polyhedra_GMM_SVM(
        data:np.ndarray, 
        init:int|np.ndarray=2, 
        n_init:int=1, 
        remove_unnecessary_couples:bool=True, 
        infos:bool=True
) -> list[np.ndarray]:
    """
    Computing 'n' polyhedron classes expressed by the intersection of half-spaces, 
    using a Gaussian Mixture Model of 'n' mixture components on input 'data', and 
    separating classes by hyperplanes computed from linear Support Vector Machine.\n
    Returns list_of_class_polyhedra.
    """
    if type(init) is np.ndarray:
        n = init.shape[0]
        init_means = init
    else:
        n = int(np.round(init))
        init_means = None
    if infos:
        print("* Fitting GMM model on data...", end=' ')
    gmm = GaussianMixture(n_components=n, covariance_type='full', init_params='kmeans', n_init=n_init, means_init=init_means).fit(X=data)
    if infos:
        print("Done!")
        print("* Predicting data on GMM model...", end=' ')
    GMM_labels = gmm.predict(X=data)
    gmm_means = gmm.means_
    if infos:
        print("Done!")
        print("* Computing SVM parameters...", end=' ')
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo').fit(X=data, y=GMM_labels)
    w, b = clf.coef_, clf.intercept_
    h_hyperplanes = to_half_space_couples(w, b)
    h_half_spaces = list(distribute_half_spaces(h_hyperplanes, gmm_means))
    if remove_unnecessary_couples:
        if infos:
            print("Done!")
            print("* Computing minimum H-descriptions...", end=' ')
        for _ in range(len(h_half_spaces)):
            h0 = h_half_spaces.pop(0)
            hp = keep_only_necessary_couples(h0)
            h_half_spaces.append(hp)
    if infos:
        print("Done!")
    return h_half_spaces

# 2. Second method: SVM on class samples (as windows in image 'img' of rectangle-coordinates 'coord')
def class_polyhedra_WindowSample_SVM(
        img:np.ndarray, 
        coord:np.ndarray, 
        remove_unnecessary_couples:bool=True, 
        infos:bool=True
) -> list[np.ndarray]:
    """
    Computing 'n' polyhedron classes expressed by the intersection of half-spaces, 
    for which the hyperplanes are computed from linear Support Vector Machine on 
    classes got from class-window crops on image 'img' given in crop coordinates 'coord'.\n
    Returns list_of_class_polyhedra.
    """
    if infos:
        print("* Computing sample parameters...", end=' ')
    means = np.asarray([get_mean_from_box_coord (img, coord_i[0], coord_i[1]) for coord_i in coord])
    #covar = np.asarray([get_covar_from_box_coord(img, coord_i[0], coord_i[1]) for coord_i in coord])
    window_data = np.concatenate([get_data_from_box_coord(img, coord_i[0], coord_i[1]) for coord_i in coord], axis=0)
    lens = [int(np.round((coord[i][0][0]-coord[i][1][0])*(coord[i][0][1]-coord[i][1][1]))) for i in range(len(coord))]
    labl = np.concatenate([np.full(shape=(lens[i],), fill_value=i, dtype=int) for i in range(len(coord))], axis=0)
    if infos:
        print("Done!")
        print("* Computing SVM parameters...", end=' ')
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo').fit(X=window_data, y=labl)
    w, b = clf.coef_, clf.intercept_
    h_hyperplanes = to_half_space_couples(w, b)
    h_half_spaces = list(distribute_half_spaces(h_hyperplanes, means))
    if remove_unnecessary_couples:
        if infos:
            print("Done!")
            print("* Computing minimum H-descriptions...", end=' ')
        for _ in range(len(h_half_spaces)):
            h0 = h_half_spaces.pop(0)
            hp = keep_only_necessary_couples(h0)
            h_half_spaces.append(hp)
    if infos:
        print("Done!")
    return h_half_spaces

# 3. Third method: k-means then Voronoi
def class_polyhedra_Kmeans_Voronoi(
        data:Optional[np.ndarray]=None, 
        init:int|np.ndarray=2, 
        n_init:Optional[int]=None, 
        remove_unnecessary_couples:bool=True, 
        infos:bool=True
) -> list[np.ndarray]:
    """
    If 'init' is of type numpy.ndarray and 'data' is None, then 'init' array is kept as Voronoi centroids (no K-means).\n
    Returns list_of_class_polyhedra.
    """
    if n_init is None:
        n_init = 'auto'
    if type(init) is np.ndarray:
        n = init.shape[0]
        if data is not None:
            if infos:
                print("* Computing k-means centroids...", end=' ')
            means = KMeans(n_clusters=n, init=init, n_init=n_init).fit(data).cluster_centers_
            if infos:
                print("Done!")
        else:
            means = init
    else:
        n = int(np.round(init))
        if data is not None:
            if infos:
                print("* Computing k-means centroids...", end=' ')
            means = KMeans(n_clusters=n, init='k-means++', n_init=n_init).fit(data).cluster_centers_
            if infos:
                print("Done!")
        else:
            unit_base = np.eye(n) - 1 / n * np.ones(n)
            proj_base = orthonormalize(unit_base[:-1]) / np.sqrt(2)
            means = scalar(proj_base, unit_base[:,np.newaxis])
    
    # Computation of Voronoi polyhedra
    if infos:
        print(f"* Computing Voronoi's class polyhedra from {'computed'*(data is not None)+'given'*(data is None)} centroids...", end=' ')
    n_means, ndim = means.shape
    centers = np.tile(means, (n_means, 1, 1))
    centers_refere = centers[:-1].transpose(1,0,2)
    centers_target = centers[~np.eye(n_means, dtype=bool)].reshape(n_means, n_means-1, ndim)
    c = np.empty(shape=centers_refere.shape, dtype=means.dtype)
    np.divide(centers_refere + centers_target, 2, out=c, casting='unsafe')
    v = normed(centers_target - centers_refere)

    h_half_spaces = list(np.transpose((c,v), axes=(1,2,0,3))) # (nh, nh - 1, 2, ndim)

    if remove_unnecessary_couples:
        if infos:
            print("Done!")
            print("* Computing minimum H-descriptions...", end=' ')
        for _ in range(len(h_half_spaces)):
            h = h_half_spaces.pop(0)
            hp = keep_only_necessary_couples(h)
            h_half_spaces.append(hp)
    if infos:
        print("Done!")
    return h_half_spaces

#%%
# Function to extract a random sample from data, which will be used to compute frontier hyperplanes in functions above
###

def extract_random_sample(data:np.ndarray, prop:float=0.01, return_indices:bool=False) -> np.ndarray:
    """
    Function to extract a random sample from data, to compute class polyhedra (for GMM for example).
    """
    n = data.shape[0]
    indices = random.sample(range(0,n), int(n*prop))
    if return_indices:
        return data[indices], indices
    return data[indices]

#%%
# Functions to compute the minimum-norm points to convex polyhedra and the associated signed distances
###

def distance_to_polyhedra(data:np.ndarray, h:list[np.ndarray], infos:bool=True) -> np.ndarray:
    """
    * data: ndarray of points in ndim-dimensional real vector space,
     with shape (n_samples, ndim) or (ndim,) ;
    * h: list of polyhedra represented as intersection of half_spaces described by couples (c,v),
     with shape n_classes * (n_half_spaces, 2, ndim) or (n_half_spaces, 2, ndim).\n
    Returns ndarray of Euclidean distances from data to each class polyhedra, 
    with shape (n_samples, n_classes) or (n_classes,) or (n_samples,) or scalar.
    """
    min_n_pts = minimum_norm_points_to_polyhedra_PYTHON(data, h, infos=infos) # PYTHON VERSION
    #min_n_pts, _ = minimum_norm_points_to_polyhedra(data, [h[i][:,1] for i in range(len(h))], [scalar(h[i][:,0],h[i][:,1]) for i in range(len(h))], infos=infos) # C VERSION
    distances = norm(min_n_pts - data[:,np.newaxis], keepdims=False)
    return distances

def add_negative_distance(data:np.ndarray, h:np.ndarray, distances:Optional[np.ndarray]=None) -> np.ndarray:
    n_samples = data.shape[0] # = distances.shape[0]
    n_classes = len(h) # = distances.shape[1]
    if distances is None:
        new_distances = np.zeros(shape=(n_samples, n_classes), dtype=data.dtype)
    else:
        new_distances = distances.copy()
    eps = np.finfo(data.dtype).resolution
    for c in range(n_classes):
        h_class = h[c]
        max_dist = np.max(scalar(data[np.newaxis] - h_class[:,0,np.newaxis], h_class[:,1,np.newaxis]), axis=0)
        neg_data = max_dist < - eps
        new_distances[neg_data,c] = max_dist[neg_data]
    return new_distances

#%%
# Functions to turn distances (positive and/or negative) into probability map via softmax function
###

def softmax_probability(distances:np.ndarray, multi:float=1, power:float=1) -> np.ndarray:
    """
    Turns relative distances into their corresponding probabilities using softmax function.\n
    Parameter:
    * distance: map of relative distances (negative values accepted) with shape (nb_samples, nb_clusters)\n
    Returns probability map computed with softmax function on relative distances, with shape (nb_samples, nb_clusters).
    """
    exp_distances = np.exp(multi * np.sign(distances) * np.abs(distances) ** power)

    exp_sum = np.sum(exp_distances, axis=-1, keepdims=True)
    exp_sum_inf  = (exp_sum == np.inf).reshape(exp_sum.shape[:-1])
    exp_sum_zero = (exp_sum == 0     ).reshape(exp_sum.shape[:-1])
    
    exp_distances_max = exp_distances == exp_distances.max(axis=-1, keepdims=True)
    exp_distances[exp_sum_inf] = exp_distances_max[exp_sum_inf]#.astype(exp_distances)
    exp_sum[exp_sum_inf] = np.sum(exp_distances_max[exp_sum_inf], axis=-1, keepdims=True)

    exp_distances[exp_sum_zero] = 1
    exp_sum[exp_sum_zero] = distances.shape[-1]

    return exp_distances / exp_sum

def tanh_probability(distances:np.ndarray, multi:float=1, power:float=1) -> np.ndarray:
    """
    Turns relative distances into their corresponding probabilities using tanh function.\n
    Parameter:
    * distance: map of relative distances (negative values accepted) with shape (nb_samples, nb_clusters)\n
    Returns probability map computed with tanh function on relative distances, with shape (nb_samples, nb_clusters).
    """
    tanh_distance = 1 + np.tanh(multi * np.sign(distances) * np.abs(distances) ** power)

    tanh_sum = np.sum(tanh_distance, axis=-1, keepdims=True)
    tanh_sum_zero = (tanh_sum == 0).reshape(tanh_sum.shape[:-1])

    tanh_distance[tanh_sum_zero] = 1
    tanh_sum[tanh_sum_zero] = distances.shape[-1]

    return tanh_distance / tanh_sum

def classical_distance_to_probability(distance:np.ndarray, dtype:Optional[type]=None) -> np.ndarray:
    """
    Turns a distance map into its corresponding probability map.\n
    Parameter:
    * distance: map of distances with shape (nb_samples, nb_clusters)\n
    Returns ndarray of the probability of each point to belong to the clusters from their distance, with shape (nb_samples, nb_clusters).
    """
    if dtype is None:
        dtype = distance.dtype
    if not np.issubdtype(dtype, np.floating):
        print("Warning: operation 'to_probability' may not be adapted for non-floating output dtype.")
    if np.issubdtype(distance.dtype, np.floating):
        eps = np.finfo(distance.dtype).resolution
    else:
        eps = 1
    zero_distance = distance < eps
    distance = distance * ~zero_distance
    min_distance = distance.min(axis=-1, keepdims=True)
    probability = np.empty(shape=distance.shape, dtype=dtype)
    np.divide(min_distance + zero_distance, distance + zero_distance, out=probability, casting='unsafe')
    np.divide(probability, probability.sum(axis=-1, dtype=dtype, keepdims=True), out=probability, casting='unsafe') # sum theoretically cannot be null
    return probability

#%%
# Function to turn signed distances into an abundance map via projection on the probability simplex
###

def simplex_frontier_hyperplanes(m:int, pdis:float=1.0) -> np.ndarray:
    v = (np.eye(m, m) - np.ones(m) / m) / np.sqrt((m - 1) / m)
    b = np.full((m, 1), fill_value = -1 / np.sqrt((m - 1) * m))
    return - np.append(v, pdis * b, axis = -1)

def from_Vb_to_CV(Vb:np.ndarray) -> np.ndarray:
    V = Vb[:, np.newaxis, :-1]
    C = V * Vb[:, np.newaxis, -1:]
    return np.append(C, V, axis = 1)

def simplex_projection(x:np.ndarray, pdis:float=1.0) -> np.ndarray:
    m = x.shape[-1]
    y = x + (pdis - np.sum(x, axis=-1, keepdims=True)) * np.ones(m) / m
    hVb = simplex_frontier_hyperplanes(m, pdis)
    p = minimum_norm_points_to_polyhedra_PYTHON(y, from_Vb_to_CV(hVb), infos=False)
    return p / pdis

def to_probability(x:np.ndarray, saturation:float=1.0) -> np.ndarray:
    """
    Function that projects x data onto the probability simplex.
    """
    #import scipy.stats as st
    #exp_x = np.exp(x) # softmax
    #exp_x = 1 / (1 + np.exp(-x)) # sigmoid
    #exp_x = x - x.min() + (0.1)#? # min-max
    #exp_x = st.norm.cdf(x) # Gaussian Cumulative Distribution Function
    exp_x = simplex_projection(x * saturation)

    exp_sum = np.sum(exp_x, axis=-1, keepdims=True)
    exp_sum_inf  = (exp_sum == np.inf).reshape(exp_sum.shape[:-1])
    exp_sum_zero = (exp_sum == 0     ).reshape(exp_sum.shape[:-1])
    
    exp_x_max = exp_x == exp_x.max(axis=-1, keepdims=True)
    exp_x[exp_sum_inf] = exp_x_max[exp_sum_inf]#.astype(exp_x)
    exp_sum[exp_sum_inf] = np.sum(exp_x_max[exp_sum_inf], axis=-1, keepdims=True)

    exp_x[exp_sum_zero] = 1
    exp_sum[exp_sum_zero] = x.shape[-1]

    return exp_x / exp_sum

#%%
# Evaluation metrics: SAD, RMSE, DSSIM
###

# Endmember SAD
def SAD(M_gt:np.ndarray, M_hat:np.ndarray, individual:bool=False) -> float:
    """Shape: (n_endmembers, n_bands) if not individual; any shape if individual, where each array represents one unique vector."""
    try:
        M_gt  = np.asarray(M_gt )
        M_hat = np.asarray(M_hat)
    except:
        raise ValueError("Input variables must be ndarrays")
    if M_gt.ndim != M_hat.ndim:
        raise ValueError("Input arrays must have same dimension")
    if np.sum(np.array(M_gt.shape) != np.array(M_hat.shape)) != 0:
        raise ValueError("Input arrays must have same shape")
    if M_gt.ndim == 0:
        return float(0.0)
    if individual:
        return np.arccos(scalar(normed(M_gt.flatten()),normed(M_hat.flatten()))).item()
    if M_gt.ndim not in {1,2}:
        raise ValueError("Input arrays must be of dimension 1 or 2")
    if M_gt.ndim == 1:
        print("Input arrays are of dimension 1, while individual=False. Considered as individual=True; otherwise, the SAD would be exactly 0.")
    elif M_gt.shape[0] > M_gt.shape[1]:
        print("The number of rows (n_endmembers) is larger than the number of columns (n_bands). Did you mean the transpose matrix?")
    return np.mean(np.arccos(scalar(normed(M_gt),normed(M_hat)))).item()

# Abundance RMSE
def RMSE(A_gt:np.ndarray, A_hat:np.ndarray, individual:bool=False) -> float:
    """Shape: (..., n_endmembers) if not individual, where prod(...) == n_pixels; any shape if individual, where array pixels are gray-level."""
    try:
        A_gt  = np.asarray(A_gt )
        A_hat = np.asarray(A_hat)
    except:
        raise ValueError("Input variables must be ndarrays")
    if A_gt.ndim != A_hat.ndim:
        raise ValueError("Input arrays must have same dimension")
    if np.sum(np.array(A_gt.shape) != np.array(A_hat.shape)) != 0:
        raise ValueError("Input arrays must have same shape")
    if A_gt.ndim == 0:
        return np.abs(A_gt - A_hat).item()
    if individual:
        return np.sqrt(np.mean(np.square(A_gt - A_hat))).item()
    try:
        oshape:tuple = A_gt.shape
        A_gt  = A_gt .reshape(int(np.prod(oshape[:-1])), oshape[-1])
        A_hat = A_hat.reshape(int(np.prod(oshape[:-1])), oshape[-1])
    except:
        raise ValueError("Cannot reshape evaluation arrays to 2D arrays in non-individual evaluation mode")
    if A_gt.shape[0] < A_gt.shape[1]:
        print("The number of rows (n_pixels) is less than the number of columns (n_endmembers). Did you mean the transpose matrix?")
    return np.mean(np.sqrt(np.mean(np.square(A_gt - A_hat), axis=0))).item()

#%%
# Permutation functions: 
# 1. permute_to_GT_M (permute M_hat and A_hat over GT endmembers M_gt), 
# 2. permute_to_GT_A (permute M_hat and A_hat over GT abundances A_gt), 
# 3. reorder_C (re-order classification labels over GT labels).
###

# Permute (A_hat, M_hat) to align with (_, M_gt)
def permute_to_GT_M(
        M_hat: np.ndarray, 
        A_hat: Optional[np.ndarray], 
        M_gt: np.ndarray, 
        method: Literal['all_arrangements','linear_sum'] = 'linear_sum'
) -> Tuple[np.ndarray, np.ndarray]|np.ndarray:
    """
    Permute the rows of M_hat and A_hat so that M_hat's rows are aligned with the ones of M_gt, minimizing SAD.\n
    * Endmember matrix shape: (n_endmembers, n_bands); 
    * Abundance matrix shape: (..., n_endmembers).\n
    If A_hat is None, then only permuted M_hat is returned. Otherwise, permuted M_hat and A_hat are both returned.
    """
    if M_hat is None or M_gt is None:
        raise ValueError("M_hat and M_gt arguments must both be given")

    if method.lower() == 'all_arrangements':
        n_endm = M_gt.shape[0]
        
        permutations = np.asarray(list(itertools.permutations(np.arange(n_endm), n_endm))).tolist()
    
        idx = permutations[0]
        val = SAD(M_gt, M_hat[idx], individual=False)
        for i in range(1,len(permutations)):
            new_idx = permutations[i]
            new_val = SAD(M_gt, M_hat[new_idx], individual=False)
            if new_val < val:
                idx = new_idx
                val = new_val
    
    elif method.lower() == 'linear_sum':
        exp_M_gt  = np.expand_dims(normed(M_gt ), axis=1)
        exp_M_hat = np.expand_dims(normed(M_hat), axis=0)
        matrix = np.arccos(scalar(exp_M_gt, exp_M_hat))
        del(exp_M_gt, exp_M_hat)

        _, idx = linear_sum_assignment(matrix, maximize=False)
    
    else:
        raise ValueError("Argument 'method' must be either 'all_arrangements' or 'linear_sum'")

    M_hat = M_hat[idx]
    if A_hat is None:
        return M_hat

    A_hat = A_hat[...,idx]
    return M_hat, A_hat

# Permute (A_hat, M_hat) to align with (A_gt, _)
def permute_to_GT_A(
        M_hat: Optional[np.ndarray], 
        A_hat: np.ndarray, 
        A_gt: np.ndarray, 
        method: Literal['all_arrangements','linear_sum'] = 'linear_sum'
) -> Tuple[np.ndarray, np.ndarray]|np.ndarray:
    """
    Permute the columns of M_hat and A_hat so that A_hat's columns are aligned with the ones of A_gt, minimizing RMSE.\n
    * Endmember matrix shape: (n_endmembers, n_bands); 
    * Abundance matrix shape: (..., n_endmembers).\n
    If M_hat is None, then only permuted A_hat is returned. Otherwise, permuted M_hat and A_hat are both returned.
    """
    if A_hat is None or A_gt is None:
        raise ValueError("M_hat and M_gt arguments must both be given")

    if method.lower() == 'all_arrangements':
        n_endm = A_gt.shape[-1]
        
        permutations = np.asarray(list(itertools.permutations(np.arange(n_endm), n_endm))).tolist()
    
        idx = permutations[0]
        val = RMSE(A_gt, A_hat[...,idx], individual=False)
        for i in range(1,len(permutations)):
            new_idx = permutations[i]
            new_val = RMSE(A_gt, A_hat[...,new_idx], individual=False)
            if new_val < val:
                idx = new_idx
                val = new_val

    elif method.lower() == 'linear_sum':
        diff_A = np.expand_dims(A_gt, axis=-1) - np.expand_dims(A_hat, axis=-2)
        matrix = np.sqrt(np.mean(np.square(diff_A), axis=tuple(range(A_gt.ndim-1))))
        del(diff_A)

        _, idx = linear_sum_assignment(matrix, maximize=False)
    
    else:
        raise ValueError("Argument 'method' must be either 'all_arrangements' or 'linear_sum'")

    A_hat = A_hat[...,idx]
    if M_hat is None:
        return A_hat
    
    M_hat = M_hat[idx]
    return M_hat, A_hat

# Compute classification map accuracy
def _map_accuracy(y_pred:np.ndarray, y_labl:np.ndarray) -> np.ndarray:
    """Accuracy for binary arrays of shape (..., n_classes)"""
    yp = np.zeros(y_pred.shape[:-1], np.int32)
    yl = np.zeros(y_labl.shape[:-1], np.int32)
    for val in range(y_pred.shape[-1]):
        yp[y_pred[...,val].astype(bool)] = val
    for val in range(y_labl.shape[-1]):
        yl[y_labl[...,val].astype(bool)] = val
    return accuracy_score(y_true=yl.flatten(), y_pred=yp.flatten(), normalize=True)

# Re-order predicted class labels over map accuracy
def reorder_C(
        y_pred: np.ndarray, 
        y_labl: np.ndarray, 
        adapt_values: bool = True, 
        method: Literal['all_arrangements','linear_sum'] = 'linear_sum'
) -> np.ndarray:
    """Re-order labels of predicted 2D classification map y_pred to align with y_labl (Ground-Truth)."""
    u_pred = np.unique(y_pred)
    u_labl = np.unique(y_labl)

    if method.lower() == 'all_arrangements':
        im_pred = np.asarray([y_pred.flatten()==val for val in u_pred], dtype=bool).T
        im_labl = np.asarray([y_labl.flatten()==val for val in u_labl], dtype=bool).T
        
        len_u = min(len(u_labl), len(u_pred))
        permutations = np.asarray(list(itertools.permutations(np.arange(len_u), len_u))).tolist()
        
        idx = permutations[0]
        acc = _map_accuracy(im_labl, im_pred[:,idx])
        for i in range(1,len(permutations)):
            new_idx = permutations[i]
            new_acc = _map_accuracy(im_labl, im_pred[:,new_idx])
            if new_acc > acc:
                idx = new_idx
                acc = new_acc
        
        new_y_pred = np.zeros_like(y_pred)
        for i in range(len(idx)): 
            new_val_i = u_labl[i] if adapt_values else u_pred[i]
            new_y_pred[y_pred==u_pred[idx[i]]] = new_val_i
        
        if len(u_pred) > len(u_labl):
            for k in set(range(len(u_pred))).difference(idx):
                matrix_k = [precision_score(y_true=(y_labl==val_j).flatten(), y_pred=(y_pred==k).flatten()) for val_j in u_labl]
                new_val_k = u_labl[np.argmax(matrix_k).item()] if adapt_values else u_pred[k]
                new_y_pred[y_pred==u_pred[k]] = new_val_k
        
        return new_y_pred.reshape(*y_pred.shape)

    elif method.lower() == 'linear_sum':
        score = precision_score if len(u_pred) != len(u_labl) else accuracy_score
    
        matrix = []
        for val_i in u_pred:
            mat_i = []
            for val_j in u_labl:
                val_ij = score(y_true=(y_labl==val_j).flatten(), y_pred=(y_pred==val_i).flatten())
                mat_i.append(val_ij)
            matrix.append(mat_i)
        
        idx, idy = linear_sum_assignment(matrix, maximize=True)
        
        new_y_pred = np.zeros_like(y_pred)
        
        if not adapt_values:
            idz = np.zeros_like(idx)
            if len(u_pred) <= len(u_labl):
                idz[np.argsort(idy)] = np.arange(len(idx))
            else:
                idz[np.argsort(idx)] = np.arange(len(idy))
        
        for k in range(len(idx)):
            if len(u_pred) <= len(u_labl):
                id_i = idx[k] # here, idx[k] == k
                id_j = idy[k] if adapt_values else idz[k]
            else:
                id_i = idx[k] if adapt_values else idz[k]
                id_j = idy[k]
            new_val_k = u_labl[id_j] if adapt_values else u_pred[id_j]
            new_y_pred[y_pred==u_pred[id_i]] = new_val_k
        
        if len(u_pred) > len(u_labl):
            for k in set(range(len(u_pred))).difference(idx if adapt_values else idz):
                new_val_k = u_labl[np.argmax(matrix[k]).item()] if adapt_values else u_pred[k]
                new_y_pred[y_pred==u_pred[k]] = new_val_k
        
        return new_y_pred

    else:
        raise ValueError("Argument 'method' must be either 'all_arrangements' or 'linear_sum'")

#%%
# Main function for Polyhedral Unmixing
###

def unmix(
        image: np.ndarray, 
        classes: int|List[np.ndarray], 
        mask: Optional[np.ndarray] = None, 
        preprocess_homogenize: bool = False, 
        preprocess_PCA_ndim: Optional[int] = None, 
        polyhedral_sample_prop: float = 1.0, 
        polyhedral_n_init: int = 1, 
        polyhedral_method: Literal['KMeans_Voronoi','GMM_SVM'] = 'GMM_SVM', 
        density_uniformize: bool = False, 
        density_method: Literal['abundance','probability'] = 'probability', 
        verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Main function for polyhedral unmixing.

    Given a spectral image with shape (..., n_bands), the algorithm determines the density map 
    (abundances or probabilities) of each of the given classes (number, spectra or coordinates) 
    using a polyhedral partitioning of the spectral space (given by GMM-SVM or kMeans-Voronoi) 
    and a density function on it based on the signed Euclidean distance to polyhedral classes.

    Parameters
    ----------
    image : ndarray
        Spectral image Y to unmix. Last axis for spectral bands.
    classes : int (> 0) or list of ndarray
        If int, number of polyhedral classes to extract. 
        Otherwise, list of n class spectra or coordinates.
    mask : ndarray, optional, default: None
        Binary array representing the pixels to consider. 
        Its shape must be: image.shape[:-1].
    preprocess_homogenize : bool, default: False
        Whether to homogenize luminance by normalizing input spectra.
    preprocess_PCA_ndim : int, optional, default: None
        If given, dimension of the PCA-reduced spectral space.
    polyhedral_sample_prop : float, in (0,1], default: 1.0
        Proportion of pixels taken to fit the polyhedral partitioning model.
    polyhedral_n_init : int, > 0, default: 1
        Number of times the partitioning model is fit with random initializations.
        The best result is retained.
    polyhedral_method : {'KMeans_Voronoi', 'GMM_SVM'}, default: 'GMM_SVM'
        Method used to construct the polyhedral partitioning model.
    density_uniformize : bool, default: False
        Whether to compute densities as A = M^-1 Y.
        If 'probability', harmonizes distance space on extreme values.
    density_method : {'abundance', 'probability'}, default: 'probability'
        Method used to compute density map (either a true abundance or a probability map).
    verbose : bool, default: True
        Whether to print progress and computation details.

    Returns
    -------
    densities : ndarray, A or P
        Density maps representing the proportions of the n polyhedral classes.
    segmented : ndarray, C
        Segmentation maps assigning each pixel to a polyhedral class (uint8).
    endmembers : ndarray, M
        Spectral signatures associated with each of the n polyhedral classes.
    """

    # Check input image
    if type(image) is not np.ndarray:
        if np.iterable(image):
            try:
                image = np.asarray(image)
            except:
                raise ValueError("Input image cannot be converted into a ndarray")
        else:
            raise ValueError("Input image must be iterable")
    if verbose:
        print(f"Spectral image - size: {image.shape[:-1]}; bands: {image.shape[-1]}")

    # Check input mask
    if mask is None:
        mask = np.ones(shape=image.shape[:-1], dtype=np.bool_)
    else:
        if type(mask) is not np.ndarray:
            if np.iterable(mask):
                try:
                    mask = np.asarray(mask)
                except:
                    raise ValueError("Input mask cannot be converted into a ndarray")
            else:
                raise ValueError("Input mask must be iterable")
        if mask.dtype != np.bool_:
            try:
                mask = mask.astype(np.bool_)
            except:
                raise ValueError("Dtype of input mask cannot be converted into np.bool_")
        if mask.ndim != image.ndim - 1:
            raise ValueError("Mask must be of dimension: image.ndim - 1")
        elif mask.shape != image.shape[:-1]:
            raise ValueError("Mask must be of shape: image.shape[:-1]")

    # Check input classes
    if np.iterable(classes):
        if type(classes) is not np.ndarray:
            try:
                classes = np.asarray(classes)
            except:
                raise ValueError("Iterable parameter 'classes' cannot be converted into a ndarray")
        if classes.ndim > 2:
            raise ValueError("Iterable parameter 'classes' must be of dimension 2")
        elif classes.ndim == 1:
            if image.shape[-1] == 1: # n grayscale classes
                classes = classes[:,np.newaxis]
            elif classes.shape[0] == image.shape[-1]: # 1 spectrum
                classes = classes[np.newaxis]
            elif classes.shape[0] == image.ndim - 1: # 1 coordinate
                classes = classes[np.newaxis]
            else:
                raise ValueError("Iterable parameter 'classes' must be of dimension 2")
        if classes.shape[-1] == image.shape[-1]: # (probably) spectra
            if classes.shape[-1] == image.ndim - 1 and np.issubdtype(classes.dtype, np.integer) and classes.dtype != image.dtype: # coordinates
                class_type = 2
            else: # spectra
                if classes.dtype != image.dtype:
                    try:
                        classes = classes.astype(image.dtype)
                    except:
                        raise ValueError("Iterable parameter 'classes', if list of spectra, must be of same dtype as input image")
                class_type = 1
        elif classes.shape[-1] == image.ndim - 1: # coordinates
            if not np.issubdtype(classes.dtype, np.integer):
                classes = classes.astype(np.int32)
            class_type = 2
        else:
            raise ValueError("Iterable parameter 'classes' must either respresent spectra or coordinates on the input spectral image")
    else:
        try:
            classes = int(classes)
        except:
            raise ValueError("Non-iterable parameter 'classes' cannot be converted into an integer")
        class_type = 0
    if verbose:
        class_verbose = (class_type == 0) * 'number of classes' 
        class_verbose+= (class_type == 1) * 'list of class spectra' 
        class_verbose+= (class_type == 2) * 'list of class coordinates'
        print(f"/!\\ Considering parameter 'classes' as {class_verbose}.")

    # Homogenize image luminance (no PCA dimensionality reduction)
    if verbose and preprocess_homogenize is not None and preprocess_homogenize != 0:
        print(f"Homogenizing luminance of Y (power = {float(preprocess_homogenize)})...", end=' ')
    homogenized_image = preprocess_image(
        image = image, 
        homogenize_luminance = preprocess_homogenize, # homogenize luminance
        ndim_PCA_reduction = None # !no! PCA dimensionality reduction here
    )
    if verbose and preprocess_homogenize is not None and preprocess_homogenize != 0:
        print("Done!")

    # PCA dimensionality reduction on homogenized image
    if verbose and preprocess_PCA_ndim is not None:
        print(f"Reducing dimensionality with PCA (ndim = {preprocess_PCA_ndim})...", end=' ')
    homoReduced_image = preprocess_image(
        image = homogenized_image, 
        homogenize_luminance = False, # luminance has already been homogenized
        ndim_PCA_reduction = preprocess_PCA_ndim # PCA dimensionality reduction
    )
    if verbose and preprocess_PCA_ndim is not None:
        print("Done!")

    # Reshape data to (n_data, n_features)
    homogenized_data = homogenized_image[mask]
    homoReduced_data = homoReduced_image[mask]

    # Deduce ('init','n_classes') parameters from ('classes','class_type')
    if class_type == 0: # [0] 'classes' is integer: number of classes
        init = classes
        n_classes = classes
    elif class_type == 1: # [1] 'classes' is list of spectra: ndarray of same dtype as input image
        init = classes
        n_classes = classes.shape[0]
    else: # [2] 'classes' is list of coordinates (on input image): ndarray of integers
        init = np.asarray([homoReduced_image[tuple(coord)] for coord in classes])
        n_classes = classes.shape[0]

    # Extract a random sample from homogenized reduced data to fit partitioning model
    if verbose:
        print(f"Extracting random sample (prop = {polyhedral_sample_prop})...", end=' ')
    homoReduced_sample = extract_random_sample(homoReduced_data, prop=polyhedral_sample_prop)
    if verbose:
        print("Done!")

    # Fit polyhedral partitioning model, either KMeans-Voronoi or GMM-SVM methods
    if verbose:
        print(f"Computing space polyhedral partitioning (method = '{polyhedral_method}', n_classes = {n_classes}, n_init = {polyhedral_n_init}):")
    if polyhedral_method.lower() == 'KMeans_Voronoi'.lower():
        h = class_polyhedra_Kmeans_Voronoi(data=homoReduced_sample, init=init, n_init=polyhedral_n_init, remove_unnecessary_couples=True, infos=verbose)
    else:
        h = class_polyhedra_GMM_SVM(data=homoReduced_sample, init=init, n_init=polyhedral_n_init, remove_unnecessary_couples=True, infos=verbose)
    
    # Compute original segmentation maps assigning each pixel to a polyhedral class
    if n_classes > 255:
        print("WARNING: the number of classes is greater than 255 -> uint8 dtype not supported for segmentation maps!")
    if verbose:
        print("Deducing segmentation maps from polyhedral classes...", end=' ')
    segmented = np.zeros(shape=image.shape[:-1], dtype=np.uint8)
    for i in range(n_classes):
        binary_class = data_in_polyhedron(homoReduced_data, h[i])
        segmented[mask] += binary_class.astype(np.uint8) * (i + 1)
    if verbose:
        print("Done!")
    
    # 2 main approaches: abundance-like or probability-like densities
    if density_method.lower() == 'abundance':
        
        # computing negative distances to class polyhedra (no need to compute positive parts)
        if verbose:
            print("Computing signed distances (negative only) to polyhedral classes...", end=' ')
        negative_distances = add_negative_distance(homoReduced_data, h)
        if verbose:
            print("Done!")

        # extract endmembers: points (spectra) with the minimum distance to classes
        if verbose:
            print("Deducing endmembers M with the minimum signed distance per class...", end=' ')
        endmembers = get_extrema_val(negative_distances, homogenized_data)
        if verbose:
            print("Done!")

        # compute first abundances: linearly transform data in spectral space using A = M^-1 Y
        if density_uniformize == False:
            print("WARNING: parameter density_uniformize is False while density_method is 'abundance' -> uniformization still applied!")
        if verbose:
            print("Computing first abundances by uniformization using A = M^{-1} Y ...", end=' ')
        first_abundances = uniformize_data(
            data = homogenized_data, 
            references = endmembers, 
            orthonormalize_matrix = False, 
            project_in_reference_space = True, 
            infos = False
        )
        if verbose:
            print("Done!")

        # turn first_abundances into true abundances (sum on axis -1 must be equal to 1.0)
        if verbose:
            print("Deducing true abundances...", end=' ')
        abundances = first_abundances[:,:endmembers.shape[0]].copy()
        abundances[abundances<0] = 0 # no data should be negative
        abundances_sum = abundances.sum(axis=-1, keepdims=True)
        abundances_sum_null = abundances_sum < np.finfo(abundances.dtype).resolution
        abundances = abundances * ~abundances_sum_null / (abundances_sum + abundances_sum_null) # the sum must be 1
        if verbose:
            print("Done!")
    
        # turn abundances into image
        densities = np.zeros(shape=image.shape[:-1]+abundances.shape[-1:], dtype=abundances.dtype)
        densities[mask] = abundances

    else:
        # compute distances to polyhedra
        if verbose:
            print("Computing signed distances to polyhedral classes:")
            print("=> Positive distances:")
        distances = distance_to_polyhedra(homoReduced_data, h, infos=verbose)
        if verbose:
            print("=> Negative distances...", end=' ')
        distances = add_negative_distance(homoReduced_data, h, distances )
        if verbose:
            print("Done!")

        # extract the extreme class distances (in distance space) and endmembers (in spectral space)
        if verbose:
            print("Deducing endmembers M with the minimum signed distance per class...", end=' ')
        dis_extrema = get_extrema_val(distances, distances)
        endmembers = get_extrema_val(distances, homogenized_data)
        if verbose:
            print("Done!")

        # if asked for, linearly transform distances in distance space using D = M^-1 Y
        if density_uniformize:
            if verbose:
                print("Uniformizing distances using D = D_{min}^{-1} Y ...", end=' ')
            uniformized_distances = uniformize_data(
                data = distances, 
                references = dis_extrema, 
                orthonormalize_matrix = False, 
                project_in_reference_space = True, 
                infos = False
            )
            if verbose:
                print("Done!")
        else:
            uniformized_distances = - distances # invert sign

        # turn uniformized distances into probabilities using normalized softmax function
        if verbose:
            print("Deducing true abundances...", end=' ')
        probabilities = uniformized_distances[:,:dis_extrema.shape[0]] / np.std(uniformized_distances) # devide by STD to get relative distances for softmax
        probabilities = softmax_probability(probabilities, multi=1, power=1) # allows smoothing predictions and get a probability map by definition of softmax
        probabilities = normalized(probabilities, axis=0) # we suppose that, for each class, there is at least one pixel on the image which has pure class spectrum!
        probabilities = probabilities / np.sum(probabilities, axis=-1, keepdims=True) # the sum must be 1
        if verbose:
            print("Done!")
    
        # turn probabilities into image
        densities = np.zeros(shape=image.shape[:-1]+probabilities.shape[-1:], dtype=probabilities.dtype)
        densities[mask] = probabilities
    
    return densities, segmented, endmembers

