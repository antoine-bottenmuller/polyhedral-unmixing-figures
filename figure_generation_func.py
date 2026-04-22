# Functions for polyhedral unmixing 
# and 3D figure generation.
#@author: Antoine BOTTENMULLER
#@created: Aug 2025
#@updated: Sep 2025

import itertools
import numpy as np
from typing import Tuple, Literal, Optional

from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

from scipy import ndimage
from scipy.spatial.transform import Rotation

from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from unmixing import scalar, norm, normed, orthonormalize
from unmixing import classical_distance_to_probability
from unmixing import data_in_polyhedron

#%%
# Function to generate a polyhedron using random H-representation (intersection of half-spaces)
###

# Function to generate random unit vector
def mean_unit_vector(v:np.ndarray) -> np.ndarray:
    """
    v: shape = (m, ndim)
    """
    min_float = 1e-6
    norm_v = norm(v, keepdims=True)
    norm_v_null = norm_v == 0
    w = v * ~norm_v_null / (norm_v + norm_v_null)
    x = np.mean(w, axis=0)
    h = norm(x)
    if np.isclose(h,0):
        w[0,0] += min_float
        return mean_unit_vector(w)
    return x / h

# Function to generate a random polyhedron defined as the intersection of hyperplanes 
def generate_polyhedron(m:int, n:int=2, radius:float=100, disparity:float=0.5) -> np.ndarray:
    """
    Randomly generate a polyhedron made of m half-spaces in vector space R**n.\n
    - m: number of half_spaces ;
    - n: dimension of vector space ;
    - radius: mean distance to zero point ;
    - disparity (c or v): disparity and randomness of vectors (c or v):
       * ~0.0: vectors are strongly similar (weak randomness)
       * ~0.5: vectors are fully random (strong randomness)
       * ~1.0: vectors are strongly opposed (weak randomness)
    """
    min_float = 1e-6
    c_stoch_val = 1- np.abs(disparity * 2 - 1)
    c_stoch_sgn = - np.sign(disparity * 2 - 1)
    v_stoch_val = c_stoch_val ** (1/1)
    h = np.empty(shape=(m,2,n), dtype=float)
    for i in range(m):
        if i==0:
            mean_vec = np.zeros(shape=(n,), dtype=h.dtype)
        else:
            mean_vec = mean_unit_vector(h[:i,0])

        random_vec_c = np.random.rand(n) * 2 - 1
        while np.isclose(norm(random_vec_c),0):
            random_vec_c = np.random.rand(n) * 2 - 1
        random_vec_c = random_vec_c / norm(random_vec_c)

        vec_c = c_stoch_val * random_vec_c + (1-c_stoch_val) * c_stoch_sgn * mean_vec

        if np.isclose(norm(vec_c),0):
            vec_c = random_vec_c
        else:
            vec_c = vec_c / norm(vec_c)

        c = vec_c * max(np.abs(np.random.normal(radius, radius * c_stoch_val)), min_float)

        random_vec_v = np.random.rand(n) * 2 - 1
        while np.isclose(norm(random_vec_v),0):
            random_vec_v = np.random.rand(n) * 2 - 1
        random_vec_v = random_vec_v / norm(random_vec_v)

        vec_v = v_stoch_val * random_vec_v + (1-v_stoch_val) * vec_c

        proj = np.sum(vec_c * vec_v)
        if np.isclose(proj,0):
            vec_v = vec_v + min_float * vec_c
        if np.sum(vec_c * vec_v) < 0:
            vec_v = - vec_v
        
        v = vec_v / norm(vec_v)

        h[i] = (c,v)
    return h

#%%
# Functions to generate spectral mixing data: endmembers M, abundances A and observations Y
###

# Generate random endmembers M
def generate_endmembers(m:int, ndim:int, mean_norm:float=1.0, disparity:float=0.5, add_dim:bool|float=False) -> np.ndarray:
    """Return random endmember matrix M."""
    M = generate_polyhedron(m, ndim, radius=mean_norm, disparity=disparity)[:,1].T
    if float(add_dim) > 0:
        M = np.concatenate([M, np.full(shape=(1,*M.shape[1:]), fill_value=float(add_dim), dtype=M.dtype)], axis=0)
    return M

# Generate random abundances A
def generate_abundances(n:int, m:int, cluster_std:float=2.0, noise_prop:float=0.5, concentration:float=1.0) -> np.ndarray:
    """Return random abundance matrix A."""
    blob_pts, _, blob_ctr = make_blobs(n_samples=n, n_features=m, centers=m, cluster_std=cluster_std, return_centers=True)
    A = np.linalg.norm(blob_pts[:,None] - blob_ctr, axis=-1)
    A = A * (1 + (np.random.rand(*A.shape)*2-1) * noise_prop) + concentration
    A = classical_distance_to_probability(A).T
    return A

# Compute noisy Y with residual term added to it
def compute_observations(endmembers:np.ndarray, abundances:np.ndarray, noise_on_YA:float=0.20, residual_noise:float=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Compute noisy observations Y (noise = noise_on_YA + residual_noise) and noisy abundances A (noise = noise_on_YA) from M and A."""
    Y = endmembers @ abundances
    Y+= (np.random.rand(*Y.shape) * 2 - 1) * noise_on_YA # noise cloud
    A = endmembers.T @ np.linalg.inv(endmembers @ endmembers.T) @ Y # recover abundance matrix A using the pseudo-inverse of the endmember matrix
    Y+= (np.random.rand(*Y.shape) * 2 - 1) * residual_noise # add little noise E
    return Y, A

# Global rotation of the observation data cloud Y and endmembers M regarding the principal components of Y
def rotate_on_principal_components_3D(observations:np.ndarray, endmembers:np.ndarray, additional_angle:float=0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Global rotation of the observations Y and endmembers M over the principal components of Y."""
    deg = np.deg2rad(additional_angle)

    # Space rotation
    mean_M = np.mean(endmembers, axis=1)
    ortho_vec = orthonormalize(np.array([*(endmembers.T[:2] - mean_M), mean_M], dtype=endmembers.dtype)) # == np.linalg.inv(ortho_vec.T)
    M = ortho_vec @ endmembers
    Y = ortho_vec @ observations

    # Plane orientation
    pca = PCA(n_components=2).fit(Y[:2].T)
    axr = np.eye(3); axr[:2,:2] = pca.components_.T
    axc = np.asarray([[np.cos(deg),-np.sin(deg),0],[np.sin(deg),np.cos(deg),0],[0,0,1]]) @ np.linalg.inv(axr)
    M = axc @ M
    Y = axc @ Y

    return Y, M

#%%
# Functions to compute and plot 3D visual arrows of the standard space basis
###

# rotAngles is a tuple of rotation angles about each axis in degrees 
def define_arrow(
        length:float=1.0, 
        apex:float=1.0, 
        start:Tuple=(0,0,0), 
        rotAngles:Tuple=(0,45,0), 
        heightTip:float=None, 
        widthTip:float=None, 
        heightRatio:float=0.3, 
        widthRatio:float=0.1, 
        heightRes:int=3, 
        widthRes:int=20
) -> Tuple[np.ndarray, np.ndarray]:
    # creating the Rotation object
    r = Rotation.from_euler('xyz', rotAngles, degrees=True)
    def f(x, y, height, width):
        return np.sqrt(2) * width - np.sqrt(x ** 2 + y ** 2) * height
    
    if heightTip is None:
        heightTip = np.sqrt(apex) * heightRatio * np.sign(apex)
    if widthTip is None:
        widthTip = np.sqrt(apex) * widthRatio
    u1, v1 = np.mgrid[0:2*np.pi:int(widthRes)*1j, 0:np.pi:int(heightRes)*1j]
    x1 = np.cos(u1)*np.sin(v1)
    y1 = np.sin(u1)*np.sin(v1)
    z1 = f(x1, y1, heightTip, widthTip) + length
    
    x1 *= widthTip
    y1 *= widthTip

    # applying rotations
    vals = np.dot(np.dstack((x1,y1,z1)),r.as_matrix().T) + start
    vals2 = np.dot(np.array([[0,0,0],[0,0,1]]),r.as_matrix().T) + start
    vals2 = np.asarray([vals2[0], vals2[0] + (vals2[1] - vals2[0]) * length])

    return vals, vals2 # vals is cone data, vals2 is tail data

# Function to plot the unit arrows of a regular vector space basis
def plot_arrow(
        ax:Axes3D=None, 
        length:float=1.0, 
        width:float=1.0, 
        apex:float=1.0, 
        start:Tuple=(0,0,0), 
        rotAngles:Tuple=(0,45,0), 
        heightTip:float=None, 
        widthTip:float=None, 
        heightRatio:float=0.3, 
        widthRatio:float=0.1, 
        heightRes:int=3, 
        widthRes:int=20, 
        shade:bool=False, 
        antialiased:bool=False, 
        rasterized:bool=False, 
        **kwargs
) -> None:
    covals, rovals = define_arrow(length, apex, start, rotAngles, heightTip, widthTip, heightRatio, widthRatio, heightRes, widthRes)
    zorder = 1.0
    if 'zorder' in kwargs.keys():
        zorder = kwargs.pop('zorder')
    
    art = ax.plot_surface(covals[:,:,0], covals[:,:,1], covals[:,:,2], shade=shade, antialiased=antialiased, zorder = zorder + .1, **kwargs) # head
    if rasterized:
        art.set_rasterized(True)
    
    art = ax.plot3D(rovals[:,0], rovals[:,1], rovals[:,2], antialiased=antialiased, linewidth=width, zorder = zorder, **kwargs) # tail
    if rasterized:
        for art_i in art:
            art_i.set_rasterized(True)

#%%
# Functions for Gaussian Mixture representation
###

# Compute an inverse Malahanobis-distance-based probability function to belong to a Gaussian cluster described by the mean-cov couple
def gauss_proba(mean:np.ndarray, cov:np.ndarray, x:np.ndarray, power:float=1.0) -> np.ndarray:
    """
    mean: (d,)
    cov: (d, d)
    x: (k, d)
    """
    # np.exp(-1/2 * scalar(x-mean, scalar(np.linalg.inv(cov), (x-mean)[:,np.newaxis,:]))) / np.sqrt((2*np.pi)**cov.shape[1] * np.linalg.det(cov))
    return 1 / scalar(x-mean, scalar(np.linalg.inv(cov), (x-mean)[:,np.newaxis,:])) ** power

# Randomly generate class labels of a point cloud under a 'gauss_proba'-based probability distribution of a Gaussian Mixture
def random_gauss(means:np.ndarray, covs:np.ndarray, x:np.ndarray) -> np.ndarray:
    """
    means: (n, d)
    covs: (n, d, d)
    x: (k, d)
    """
    probs = np.asarray([gauss_proba(mean=means[i], cov=covs[i], x=x) for i in range(covs.shape[0])]) # (n, k)
    probs = np.triu(np.tile(probs, (covs.shape[0], 1, 1)).T).sum(axis=1).T; probs /= probs[-1] # (n, k)
    jeton = np.random.rand(x.shape[0]); probs -= jeton; probs[probs <= 0] = 2
    return np.argmin(probs, axis=0) # (k,)

# from 3D mean-covariance couple to 2D Ellipse object ready to plot
def covariance_ellipse(mu:np.ndarray, Sigma:np.ndarray, radius:float=1.0, **patch_kwargs):
    vals, vecs = np.linalg.eigh(Sigma) # Décomposition en valeurs/vecteurs propres
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order] # Ordonner par valeur propre décroissante (grand axe d'abord)
    width, height = 2 * radius * np.sqrt(vals) # Demi-axes = radius * sqrt(valeur_propre)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])) # Angle de l'ellipse (en degrés) depuis le 1er vecteur propre
    return Ellipse(xy=mu, width=width, height=height, angle=angle, **patch_kwargs)

# from 3D mean-covariance couple to 3D ellipsoid mesh for 3D plotting
def get_ellipsoid(mu:np.ndarray, Sigma:np.ndarray, radius:float=1.0, n_latitude:int=30, n_longitude:int=20) -> np.ndarray:
    u, v = np.mgrid[0:2*np.pi:int(n_latitude)*1j, 0:np.pi:int(n_longitude)*1j]
    sphere_pts = np.asarray([np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)]).reshape(3, np.prod(u.shape))
    ellips_pts = (radius * (np.linalg.cholesky(Sigma) @ sphere_pts).T + mu).T.reshape(3, *u.shape)
    return ellips_pts

#%%
# Function for the Kemeny optimal ranking / minimum feedback arc set problem (main function: best_stack_order)
###

def score_order(M, order):
    """Nombre de paires conformes à M quand order est l'ordre de haut->bas."""
    M = np.asarray(M)
    idx = np.array(order, dtype=int)
    s = 0
    n = len(order)
    for i in range(n-1):
        a = idx[i]
        row = M[a, idx[i+1:]]
        s += int(np.sum(row == 1))
    return int(s)

def kemeny_dp_optimal(M, max_n=16):
    """
    Exact pour n<=max_n via DP par sous-ensembles.
    Retourne (order, score). order: haut -> bas.
    """
    M = np.asarray(M)
    n = M.shape[0]
    if n > max_n:
        raise ValueError(f"n={n} trop grand pour le DP exact (>{max_n}).")
    W = (M == 1).astype(np.int16) # W[i,j]=1 si i "bat" j

    size = 1 << n
    dp = np.full(size, -10**9, dtype=np.int32)
    parent = np.full(size, -1, dtype=np.int16) # qui est ajouté en dernier
    prevmask = np.full(size, -1, dtype=np.int32)

    dp[0] = 0
    for mask in range(size):
        if dp[mask] < -10**8:
            continue
        # éléments pas encore placés
        remaining = (~mask) & (size-1)
        k = remaining
        while k:
            lsb = k & -k
            j = (lsb.bit_length() - 1) # index du nouvel élément placé à la fin
            newmask = mask | lsb
            # contribution: paires (i,j) avec i déjà placé avant j
            add = 0
            mm = mask
            while mm:
                l = mm & -mm
                i = (l.bit_length() - 1)
                add += W[i, j]
                mm ^= l
            val = dp[mask] + add
            if val > dp[newmask]:
                dp[newmask] = val
                parent[newmask] = j
                prevmask[newmask] = mask
            k ^= lsb

    # reconstruction (on a placé les éléments dans l'ordre, dernier ajouté = plus bas)
    mask = size - 1
    order_rev = []
    while mask:
        j = parent[mask]
        order_rev.append(int(j))
        mask = prevmask[mask]
    order = list(reversed(order_rev)) # haut -> bas
    return order, int(dp[-1])

def copeland_init(M):
    """Ordre initial par score de Copeland : somme des +/-1 sur chaque ligne."""
    s = np.sum(M, axis=1)  # wins - losses
    return list(np.argsort(-s, kind='stable'))

def local_adjacent_search(M, order, max_passes=10):
    """Recherche locale par échanges adjacents qui améliorent le score."""
    best = order[:]
    best_s = score_order(M, best)
    n = len(order)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n-1):
            cand = best[:]
            cand[i], cand[i+1] = cand[i+1], cand[i]
            s = score_order(M, cand)
            if s > best_s:
                best, best_s = cand, s
                improved = True
    return best, best_s

def best_stack_order(M, exact_threshold=16, restarts=10, seed=0):
    """
    Cherche un ordre haut->bas maximisant les accords avec M.
    - Exact si n <= exact_threshold (DP).
    - Sinon : Copeland + recherche locale + quelques redémarrages aléatoires.
    Retourne (order, score, conflicts).
    """
    rng = np.random.default_rng(seed)
    M = np.asarray(M)
    n = M.shape[0]
    total_pairs = n*(n-1)//2

    if n <= exact_threshold:
        order, s = kemeny_dp_optimal(M, max_n=exact_threshold)
        return order, s, total_pairs - s

    # Heuristique
    best_order = copeland_init(M)
    best_order, best_s = local_adjacent_search(M, best_order)
    # redémarrages
    for _ in range(restarts):
        o = np.array(best_order)
        rng.shuffle(o)
        cand_o, cand_s = local_adjacent_search(M, list(o))
        if cand_s > best_s:
            best_order, best_s = cand_o, cand_s
    return best_order, best_s, total_pairs - best_s

#%%
# Functions to compute objects' plot sizes and objects' plot order, depending on their distance to the camera in a 3D scene
###

# Compute object sizes in a 3D scene relatively to their distance to the camera
def get_sizes(ax, x, y, z, s_base=20, min_scale=0.4, max_scale=2.0) -> np.ndarray:
    """
    Recalcule les tailles des marqueurs en fonction de la profondeur vue par la caméra.
    s_base : taille de base (points^2, convention matplotlib)
    min_scale/max_scale : bornes du facteur d'échelle (évite des tailles absurdes)
    """
    M = ax.get_proj()
    _, _, zs = proj3d.proj_transform(x, y, z, M)
    zmin, zmax = np.min(zs), np.max(zs)
    zn = (zs - zmin) / (zmax - zmin + 1e-12)
    sizes = s_base * (min_scale + (1 - zn) * (max_scale - min_scale))
    return sizes

# Compute object plotting order using Kemeny optimal ranking, from the furthest to the closest object to plot
def get_order(clusters:list, walls:list, wall_dirs:list, camera:np.ndarray, ignore_clusters:bool=True, ignore_out_cones:bool=True) -> Tuple[list, list]:
    """
    Compute lists of display order of point clusters and plane point sets (walls), from the first (furthest) to the last (closest) set to plot. 
    This function uses Kemeny optimal ranking (exact when n_objects <= 16) to compute the optimal order.
    """
    c_mean = [np.mean(c, axis=0) for c in clusters]
    w_mean = [np.mean(w, axis=0) for w in walls]
    list_m = np.concatenate([c_mean, w_mean], axis=0)
    couples = list(itertools.combinations(range(len(walls)), 2)) # to know if hyperplane j belongs to the polyhedral surface of the point cloud of class i
    Morder = np.zeros(shape=(len(list_m),)*2, dtype=np.int16)
    for i in range(0, len(list_m)):
        for j in range(i+1, len(list_m)):
            if j < len(c_mean): # Q: is 'i' closer to the camera than 'j' is?
                dis_i = norm(list_m[i] - camera)
                dis_j = norm(list_m[j] - camera)
                state = int(np.sign(dis_j - dis_i))
                Morder[i,j] = state * int(not(ignore_clusters))
            else: # Q: is 'i' in front of 'j'?
                ignore = ignore_out_cones and i < len(c_mean) and i not in couples[j-len(c_mean)]
                if ignore:
                    state = int(0) # ignore
                else:
                    dirw = wall_dirs[j - len(c_mean)]
                    dirw *= int(np.sign(scalar(dirw, camera)))
                    state = int(np.sign(scalar(dirw, list_m[i])))
                Morder[i,j] = state
    Morder.T[Morder!=0] = - Morder[Morder!=0] # antisymmetric matrix
    
    order_idx, agrees, conflicts = best_stack_order(Morder)
    #print("Ordre haut->bas :", order_idx)
    #print("Accords / Conflits :", agrees, "/", conflicts)

    order = np.zeros(len(list_m))
    order[list(order_idx)] = 1 - (1 + np.arange(len(list_m))) / (len(list_m) + 1)
    return order[:len(c_mean)], order[len(c_mean):]

#%%
# Function to plot 2D rectangular planes orthogonal to basis axes in a 3D scene with pre-fixed plot order
###

# Function to plot a 4-block sized plane orthogonal to x, y or z axis, 
# with fixed plot order between which the back and front of the planes are plot
def plot_quadrants(
        ax:Axes3D, 
        array:np.ndarray, 
        fixed_coord:Literal['x','y','z'], 
        center:tuple, 
        size:tuple, 
        alpha:float=1.0, 
        gradient:float=1.0, 
        linewidth:int|float=0.1, 
        cmap:str='viridis_r', 
        shade:bool=False, 
        antialiased:bool=False, 
        rasterized:bool=False, 
        zorder_between:int|float=3
) -> None:
    """
    For a given 3d *array* plot a plane with *fixed_coord*, using four quadrants. 
    Adapted from: https://matplotlib.org/stable/gallery/mplot3d/intersecting_planes.html.
    """
    nx, ny, nz = array.shape
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane_data = array[index]

    n0, n1 = plane_data.shape
    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]

    max_val = array.max()
    min_val = array.min() 
    min_val = min_val - (max_val - min_val) * (1 / gradient - 1) if gradient > 0 else -float('inf')

    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val)) if gradient > 0 else cmap(np.ones(shape=(n0//2,n1//2))) #0.5*
        zorder_i = zorder_between+1 if i==3 else zorder_between-0.5+0.1*i
        if fixed_coord == 'x':
            Y = np.linspace(center[1]-size[0]/2, center[1], n0//2)
            Z = np.linspace(center[2]-size[1]/2, center[2], n1//2)
            Y, Z = np.meshgrid(Y, Z, indexing='ij')
            X = center[0] * np.ones_like(Y)
            Y_offset = (i // 2) * size[0]/2
            Z_offset = (i % 2) * size[1]/2
            art = ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1, linewidth=linewidth, 
                                  facecolors=facecolors, alpha=alpha, shade=shade, antialiased=antialiased, 
                                  label='Red class limit' if i==0 else None, zorder=zorder_i)
            if rasterized:
                art.set_rasterized(True)
        elif fixed_coord == 'y':
            X = np.linspace(center[0]-size[0]/2, center[0], n0//2)
            Z = np.linspace(center[2]-size[1]/2, center[2], n1//2)
            X, Z = np.meshgrid(X, Z, indexing='ij')
            Y = center[1] * np.ones_like(X)
            X_offset = (i // 2) * size[0]/2
            Z_offset = (i % 2) * size[1]/2
            art = ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride=1, cstride=1, linewidth=linewidth, 
                                  facecolors=facecolors, alpha=alpha, shade=shade, antialiased=antialiased, 
                                  label='Green class limit' if i==1 else None, zorder=zorder_i)
            if rasterized:
                art.set_rasterized(True)
        elif fixed_coord == 'z':
            X = np.linspace(center[0]-size[0]/2, center[0], n0//2)
            Y = np.linspace(center[1]-size[1]/2, center[1], n1//2)
            X, Y = np.meshgrid(X, Y, indexing='ij')
            Z = center[2] * np.ones_like(X)
            X_offset = (i // 2) * size[0]/2
            Y_offset = (i % 2) * size[1]/2
            art = ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride=1, cstride=1, linewidth=linewidth, 
                                  facecolors=facecolors, alpha=alpha, shade=shade, antialiased=antialiased, 
                                  label='Blue class limit' if i==0 else None, zorder=zorder_i)
            if rasterized:
                art.set_rasterized(True)

#%%
# Function to plot 2D triangular planes delimiting class regions in a 3D scene with pre-fixed plot order
###

# Triangular planes delimiting the three regions led by the standard basis in 3D polyhedron-distance space
def plot_semis(
        ax:Axes3D, 
        array:np.ndarray, 
        fixed_coord:Literal['x','y','z'], 
        center:tuple, 
        size:tuple, 
        alpha:float=1.0, 
        gradient:float=1.0, 
        linewidth:int|float=0.1, 
        cmap:str='viridis_r', 
        shade:bool=False, 
        antialiased:bool=False, 
        rasterized:bool=False, 
        zorder:int|float=3
) -> None:
    """
    For a given 3d *array* plot a triangular plane with *fixed_coord*. 
    Adapted from: https://matplotlib.org/stable/gallery/mplot3d/intersecting_planes.html.
    """
    nx, ny, nz = array.shape
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane_data = array[index]

    n0, n1 = plane_data.shape

    triangle = np.tri(n0, n1, dtype=np.bool_)
    mask = triangle * np.flip(triangle, axis=0)
    quadrant = plane_data
    quadrant[:,:n1//2] = quadrant[:,n1//2:]

    max_val = array.max()
    min_val = array.min() 
    min_val = min_val - (max_val - min_val) * (1 / gradient - 1) if gradient > 0 else -float('inf')

    cmap = plt.get_cmap(cmap)

    facecolors = cmap((quadrant - min_val) / (max_val - min_val)) if gradient > 0 else cmap(np.ones(shape=(n0,n1))) #0.5*
    zorder_i = zorder
    if fixed_coord == 'x':
        Y = np.linspace(center[1], center[1]+size[0], n0)
        Z = np.linspace(center[2], center[2]+size[1], n1)
        Y, Z = np.meshgrid(Y, Z, indexing='xy')
        X, Y, Z = 1 / np.sqrt(2) * Z, Y, 1 / np.sqrt(2) * Z
        X[~mask] = np.nan
        Y[~mask] = np.nan
        Z[~mask] = np.nan
        art = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=linewidth, 
                              facecolors=facecolors, alpha=alpha, shade=shade, antialiased=antialiased, 
                              label='Red-Blue SHP', zorder=zorder_i)
        if rasterized:
            art.set_rasterized(True)
    elif fixed_coord == 'y':
        X = np.linspace(center[0], center[0]+size[0], n0)
        Y = np.linspace(center[1], center[1]+size[1], n1)
        X, Y = np.meshgrid(X, Y, indexing='xy')
        X, Y, Z = X, 1 / np.sqrt(2) * Y, 1 / np.sqrt(2) * Y
        X[~mask] = np.nan
        Y[~mask] = np.nan
        Z[~mask] = np.nan
        art = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=linewidth, 
                              facecolors=facecolors, alpha=alpha, shade=shade, antialiased=antialiased, 
                              label='Blue-Green SHP', zorder=zorder_i)
        if rasterized:
            art.set_rasterized(True)
    elif fixed_coord == 'z':
        Z = np.linspace(center[2], center[2]+size[0], n0)
        X = np.linspace(center[0], center[0]+size[1], n1)
        Z, X = np.meshgrid(Z, X, indexing='xy')
        X, Y, Z = 1 / np.sqrt(2) * X, 1 / np.sqrt(2) * X, Z
        X[~mask] = np.nan
        Y[~mask] = np.nan
        Z[~mask] = np.nan
        art = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=linewidth, 
                              facecolors=facecolors, alpha=alpha, shade=shade, antialiased=antialiased, 
                              label='Green-Red SHP', zorder=zorder_i)
        if rasterized:
            art.set_rasterized(True)

#%%
# Functions to locate points relatively to a sphere, and plot a hyperplane (half-wall) crossing that sphere
###

# Function to know whether points are inside (val: 1) or behind (val: 2) a sphere, or if they are visible to the camera (int: 0)
def locate_around_sphere(
        points:np.ndarray, 
        camera:np.ndarray, 
        proj_type:Literal['persp','ortho']='persp',
        sphere_radius:float=1.0, 
        sphere_center:Optional[Tuple]=None
) -> np.ndarray:
    """
    * points: (n_points, ndim)
    * camera: (ndim,)
    * sphere_radius: float
    * sphere_center: (ndim,)\n
    Returns array of integers of shape points.shape, 
    with 0 for 'visible', 1 for 'inside sphere', 2 for 'behind sphere'
    """
    if sphere_center is None:
        sphere_center = np.zeros_like(camera)
    sct = np.asarray(sphere_center)
    pts = np.asarray(points) - sct
    cma = np.asarray(camera) - sct

    if proj_type == 'ortho':
        dirt = normed(cma)
        dist = np.abs(scalar(pts, dirt, keepdims=True)) + 2 * sphere_radius
        cma = pts + dist * dirt

    diff = pts - cma
    scal = scalar(cma, diff)
    difn2 = np.square(norm(diff))
    cman2 = np.square(norm(cma))
    radn2 = np.square(sphere_radius)
    delta = np.square(scal) - difn2 * (cman2 - radn2)

    delta_up = delta >= 0
    lambda_1 = (- scal[delta_up] - np.sqrt(delta[delta_up])) / difn2[delta_up]
    lambda_2 = (- scal[delta_up] + np.sqrt(delta[delta_up])) / difn2[delta_up]

    lambda_inside = np.logical_and(lambda_1 >= 0, lambda_1 <= 1)
    lambda_behind = np.logical_and(lambda_2 >= 0, lambda_2 <= 1)
    
    location = np.zeros(shape=points.shape[:1], dtype=np.uint8)
    location[delta_up] += lambda_inside.astype(np.uint8)
    location[delta_up] += lambda_behind.astype(np.uint8)
    return location

# Function to plot a plane crossing a sphere in a 3D scene, with given plot order of each location
def plot_half_wall(
        ax:Axes3D, 
        array:np.ndarray, 
        camera:np.ndarray, 
        sphere_radius:float=1.0, 
        sphere_center:Tuple=(0,0,0), 
        alpha:float=1.0, 
        gradient:float=1.0, 
        lumishift:float=0.5, 
        linewidth:int|float=0.1, 
        cmap:str='viridis_r', 
        shade:bool=False, 
        antialiased:bool=False, 
        rasterized:bool=False, 
        zorder_1:int|float=3.4, 
        zorder_2:int|float=3.5, 
        zorder_3:int|float=3.6
) -> None:
    """
    For a given 3d *array* plot a plane crossing a sphere of radius *sphere_radius* and center *sphere_center*. 
    """
    _, ny, nz = array.shape
    quadrant = (np.mgrid[-1:1:1j * ny * 2, -1:1:1j * nz] ** 2).sum(0)[ny:]

    max_val = quadrant.max()
    min_val = quadrant.min()

    color_map = plt.get_cmap(cmap)
    facecolors = color_map((quadrant - min_val) / (max_val - min_val) * gradient + (1 - gradient) * (1 - lumishift))

    points = array.reshape(array.shape[0], np.prod(array.shape[1:])).T
    #inside = norm(points - np.asarray(sphere_center)) <= float(sphere_radius)
    locate = locate_around_sphere(points=points, camera=camera, sphere_radius=sphere_radius, sphere_center=sphere_center)
    locate = locate.reshape(array.shape[1:])

    be_boo = locate == 2 # behind the sphere
    in_boo = locate == 1 # insied the sphere
    ou_boo = locate == 0 # out of the sphere
    be_boo = ndimage.binary_dilation(be_boo, structure=np.ones((3,)*2, dtype=bool)) * ~in_boo
    in_boo = ndimage.binary_dilation(in_boo, structure=np.ones((3,)*2, dtype=bool))

    be_pts = array.copy()
    be_pts[:,~be_boo] = np.nan
    in_pts = array.copy()
    in_pts[:,~in_boo] = np.nan
    ou_pts = array.copy()
    ou_pts[:,~ou_boo] = np.nan

    art = ax.plot_surface(*be_pts, rstride=1, cstride=1, linewidth=linewidth, facecolors=facecolors, 
                          alpha=alpha, shade=shade, antialiased=antialiased, zorder=zorder_1)
    if rasterized:
        art.set_rasterized(True)
    art = ax.plot_surface(*in_pts, rstride=1, cstride=1, linewidth=linewidth, facecolors=facecolors, 
                          alpha=alpha, shade=shade, antialiased=antialiased, zorder=zorder_2)
    if rasterized:
        art.set_rasterized(True)
    art = ax.plot_surface(*ou_pts, rstride=1, cstride=1, linewidth=linewidth, facecolors=facecolors, 
                          alpha=alpha, shade=shade, antialiased=antialiased, zorder=zorder_3)
    if rasterized:
        art.set_rasterized(True)

#%%
# Function to quickly compute the exact signed distances between data and polyhedral sets in h made of two halfspaces
###

# Fast function to compute the distance between points and polyhedra made of two halfspaces
def quick_distances(data:np.ndarray, h:np.ndarray) -> np.ndarray:
    """
    Signed distance function for polyhedra made of two halfspaces.
    * data: (n_elements, ndim)
    * h: len(h) * (2, 2, ndim)
    """
    in_poly = [data_in_polyhedron(data, h_i) for h_i in h]
    distanc = np.zeros(shape=(data.shape[0],len(h)), dtype=np.float64)
    for k in range(len(h)):
        distanc[in_poly[k], k] = np.max(scalar(data[in_poly[k],np.newaxis] - h[k][:,0], h[k][:,1]), axis=1)
        
        dta = data[~in_poly[k]]
        
        sc1 = scalar(dta - h[k][0,0], h[k][0,1])
        co1 = dta - sc1[:,None] * h[k][0,1]
        po1 = data_in_polyhedron(co1, h[k])
        no1 = np.abs(sc1)
        no1[~po1] = np.inf

        sc2 = scalar(dta - h[k][1,0], h[k][1,1])
        co2 = dta - sc2[:,None] * h[k][1,1]
        po2 = data_in_polyhedron(co2, h[k])
        no2 = np.abs(sc2)
        no2[~po2] = np.inf

        vah = normed(h[k][1,1] - scalar(h[k][1,1], h[k][0,1], keepdims=True) * h[k][0,1])
        dah = scalar(co1 - h[k][1,0], h[k][1,1], keepdims=True) / scalar( vah, h[k][1,1])
        no3 = norm(sc1[:,None] * h[k][0,1] + dah * vah)
        
        distanc[~in_poly[k], k] = np.min([no1, no2, no3], axis=0)
    return distanc
