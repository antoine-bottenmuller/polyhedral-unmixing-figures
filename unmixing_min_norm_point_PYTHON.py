import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Literal, Any
from scipy.optimize import linprog

#%%
# Basic functions for vector manipulation
###

def scalar(a:np.ndarray, b:np.ndarray, keepdims:bool=False) -> np.ndarray:
    if keepdims:
        return np.einsum('...i,...i->...', a, b)[..., np.newaxis]
    return np.einsum('...i,...i->...', a, b)

def norm(v:np.ndarray, keepdims:bool=False) -> np.ndarray:
    return np.linalg.norm(v, axis=-1, keepdims=keepdims)

def normed(v:np.ndarray) -> np.ndarray:
    v_norm = norm(v, keepdims=True)
    v_zero = v_norm == 0
    return v * ~v_zero / (v_norm + v_zero)

def global_standardized(v:np.ndarray) -> np.ndarray:
    # globally standardize pools (axis=-2) of vectors (axis=-1)
    v_mean = np.mean(v, axis=-2, keepdims=True) # vector
    new_v_ = v - v_mean
    v_std_ = np.std(new_v_, axis=(-2,-1), keepdims=True) # scalar
    v_zero = v_std_ == 0
    return new_v_ * ~v_zero / (v_std_ + v_zero)

#%%
# Functions to convert parameters (a,b) in <a,x> + b >= 0, to parameters (c,v), v unit, in <x-c,v> <= 0 ; and reciprocally!
###

def max_indicator_array(a:np.ndarray) -> np.ndarray:
    """
    Return boolean array of same shape as a, indicating where the (unique) maximum values along the last axis are located.
    """
    max_indices = np.argmax(a, axis=-1)
    b = np.zeros(shape=a.shape, dtype=bool)
    grid = np.ogrid[tuple(slice(dim) for dim in a.shape[:-1])]
    b[(*grid, max_indices)] = True
    return b

# from inequalities to half-space couples
def to_half_space_couples(w:np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    Finds a (non-unique) vector couple (c,v), with v unite vector, 
    such that the inequality <x-c,v> <= 0 is equivalent to <w,x>+b >= 0.
    These are two ways of expressing an half-space inequality.
    We have v=-w. c exists and is unique iif w!=0 ; 
    c exists and can be any vector iif w==0 and b>=0 (c=0 by default) ; 
    c does not exist iif w==0 and b<0 (c,v are then set to np.nan).\n
    Input:
    * w: vector or matrix of shape (..., ndim) ;
    * b: scalar or vector of shape (...,) or (..., 1).\n
    Return: array h of vector couples (c,v), with shape (..., 2, ndim).
    """
    if type(w) is not np.ndarray:
        try:
            w = np.asarray(w)
        except:
            raise ValueError("Parameter 'w' must be an array-like.")
    if type(b) is not np.ndarray:
        try:
            b = np.asarray(b)
        except:
            raise ValueError("Parameter 'b' must be an array-like or a scalar.")
    if b.ndim == w.ndim-1:
        b = np.expand_dims(b, axis=-1)
    elif b.ndim != w.ndim:
        raise ValueError("Array 'b' must be of dimension w.ndim-1 or w.ndim.")
    
    sign_cv_ineq = - 1 # '-1' because inequality <x-c,v> <= 0 is negative inequality
    sign_wb_ineq = + 1 # '+1' because inequality <w,x>+b >= 0 is positive inequality
    inverted = sign_cv_ineq * sign_wb_ineq # '-1' if these two inequalities have opposite sign!

    w_max = w * max_indicator_array(np.abs(w))
    w_max_zero = w_max == 0

    w_norm = norm(w, keepdims=True)
    w_norm_zero = w_norm == 0

    c_all_b = inverted * np.ones_like(w) * b
    c = c_all_b * ~w_max_zero / (w_max + w_max_zero)

    v = inverted * w / (w_norm + w_norm_zero)

    # Cases where w is the zero vector (i.e. if max(|w_i|)==0 or ||w||==0)
    w_zero = np.prod(w_max_zero, axis=-1, dtype=bool, keepdims=True) + w_norm_zero # max(|w_i|)==0 or ||w||==0
    opposite_b = sign_wb_ineq * b < 0 # 0*x+b >= 0 has no solution iif b < 0 (inverted if inequality is negative)
    ispossible = (w_zero *~opposite_b)[...,0] # consider cases where the wb inequality is possible!
    impossible = (w_zero * opposite_b)[...,0] # consider cases where the wb inequality is impossible!
    c[ispossible], v[ispossible] = 0, 0
    c[impossible], v[impossible] = np.nan, np.nan

    h = np.asarray([c,v], dtype=v.dtype)
    h = np.transpose(h, axes=tuple(np.arange(1,v.ndim))+(0,)+(v.ndim,))

    return h

def to_half_space_inequality(h:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds a (non-unique) vector/matrix w and scalar/vector b, 
    such that the inequality <w,x>+b >= 0 is equivalent to <x-c,v> <= 0.
    These are two ways of expressing an half-space inequality.
    We have w=-v and b=<v,c> (both are always well defined).\n
    Input: array h of vector couples (c,v), with shape (..., 2, ndim).\n
    Return:
    * w: vector or matrix of shape (..., ndim) ;
    * b: scalar or vector of shape (...,).
    """
    if type(h) is not np.ndarray:
        try:
            h = np.asarray(h)
        except:
            raise ValueError("Parameter 'h' must be an array-like.")
    
    sign_cv_ineq = - 1 # '-1' because inequality <x-c,v> <= 0 is negative inequality
    sign_wb_ineq = + 1 # '+1' because inequality <w,x>+b >= 0 is positive inequality
    inverted = sign_cv_ineq * sign_wb_ineq # '-1' if these two inequalities have opposite sign!

    h = np.transpose(h, axes=(-2,)+tuple(np.arange(h.ndim-2))+(-1,))
    c = h[0] # shape: (..., ndim)
    v = h[1] # shape: (..., ndim)

    w = inverted * v
    b = - inverted * np.sum(v * c, axis=-1)

    return w, b

#%%
# Functions to check non-emptiness of polyhedron using I-rank of matrices
###

def mres(array:np.ndarray, upmul:int=0) -> float:
    dtype = array.dtype
    if np.issubdtype(dtype, np.floating):
        finfo = np.finfo(dtype)
        decim = int(finfo.precision - upmul)
        nfres = finfo.resolution * np.power(10,upmul)
        return np.round(nfres, decimals=decim)
    return dtype.type(1)

def I_minor(M:np.ndarray, r:int) -> np.ndarray:
    pos = M[:,r] > 0
    neg = M[:,r] < 0
    zer = M[:,r] ==0
    P,N = pos.sum(), neg.sum()
    new_M = np.delete(M,r,axis=1)
    pairs = np.asarray(np.meshgrid(M[pos,r], M[neg,r])).T.reshape(P*N,2)
    mat = np.asarray([np.tile(new_M[pos],(N,1,1)), np.tile(new_M[neg],(P,1,1)).transpose(1,0,2)]).T.reshape(new_M.shape[1],P*N,2)
    prod_pairs_mat = (pairs * mat[:,:,::-1]).T
    I_complement_PN = prod_pairs_mat[0] - prod_pairs_mat[1]
    I_complement_Z  = new_M[zer]
    I_complement = np.concatenate((I_complement_PN, I_complement_Z), axis=0)
    return I_complement

def I_rank(M:np.ndarray, res:Optional[float]=None) -> int:
    if res is None:
        res = mres(M, 1 + max(0, int(np.log10(np.max(np.abs(M)))))) # error committed (>0)
    else:
        res = np.abs(res)
    m = M.shape[0]
    M = M*(np.abs(M)>res)
    sgn = np.sign(M)
    ssm = sgn.sum(0)
    tpe = (ssm==m)+(ssm==-m)
    if tpe.sum()>0:
        return M.shape[1]
    matrices = [M]
    level = M.shape[1]-1
    while level>0:
        I_minors = []
        for M in matrices:
            for r in range(level+1):
                I_complement = I_minor(M,r)
                m = I_complement.shape[0]
                I_complement = I_complement*(np.abs(I_complement)>res)
                sgn = np.sign(I_complement)
                ssm = sgn.sum(0)
                tpe = (ssm==m)+(ssm==-m)
                if tpe.sum()>0:
                    return level
                I_minors.append(I_complement)
        matrices = I_minors
        level -= 1
    return level

def exists_a_solution_using_I_rank_Computation_Method(A:np.ndarray, b:Optional[np.ndarray]=None, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * A: 2D matrix, as ndarray of shape (m,n) ;
    * b: 1D vector, as ndarray of shape (m,) ;
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif a solution x exists in the system of linear inequalities: A * x + b > 0.
    """
    if b is not None:
        A = np.concatenate([A, b[:,np.newaxis]], axis=1, dtype=A.dtype)
        A = np.concatenate([A, [(0,)*(A.shape[1]-1)+(1,)]], axis=0, dtype=A.dtype)
    return I_rank(A, res) > 0

def exists_a_solution_using_Linear_Programming_Feasibility_Method(A:np.ndarray, b:Optional[np.ndarray]=None, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * A: 2D matrix, as ndarray of shape (m,n) ;
    * b: 1D vector, as ndarray of shape (m,) ;
    * res: precision float >0 to solve A * x + b > 0 (Psi open) ; float <= 0 to solve A * x + b >= 0.\n
    => Returns True iif a solution x exists in the system of linear inequalities: A * x + b > 0.
    """
    if res is None:
        res = mres(A, 1 + max(0, int(np.log10(np.max(np.abs(A)))))) # error committed (>0)
    res = 1e-6 * np.sign(res) # TODO: better adapt this parameter! ???
    return linprog(np.zeros(A.shape[1]), A_ub=-A, b_ub=b-res, bounds=(None,None), method='highs', options={'presolve':False}).success

def exists_a_solution(A:np.ndarray, b:Optional[np.ndarray]=None, res:Optional[float]=None, method:Literal['LP','IR']='LP') -> bool:
    """
    Parameters:
    * A: 2D matrix, as ndarray of shape (m,n) ;
    * b: 1D vector, as ndarray of shape (m,) ;
    * method: to check consistency of linear system, either 'LP' for Linear Programming or 'IR' for I-rank ;
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif a solution x exists in the system of linear inequalities: A * x + b > 0.
    """
    if method == 'LP':
        return exists_a_solution_using_Linear_Programming_Feasibility_Method(A, b, res)
    elif method == 'IR':
        return exists_a_solution_using_I_rank_Computation_Method(A, b, res)
    raise ValueError("Parameter 'method' must be either 'LP' or 'IR'.")

def polyhedron_is_fully_dimensional(V:np.ndarray, P:np.ndarray, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * V: family of m vectors of size n directing the m hyperplans (HP) and pointing outward the half-spaces (HS), as ndarray of shape (m,n) ;
    * P: family of m points of size n belonging to the m hyperplans, as ndarray of shape (m,n) ;
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif the interior of polyhedron Psi_h, i.e. the intersection of all half-spaces defined by couples h=(P,V), is not empty.
    """
    #V = normed(V) # important to norm direction vectors V ?
    #P = global_standardized(P) # important to standardize points P ?
    A = - V
    b = scalar(P,V)
    return exists_a_solution(A, b, res)

def couple_is_necessary(i:int, V:np.ndarray, P:np.ndarray, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * V: family of m vectors of size n directing the m hyperplans (HP) and pointing outward the half-spaces (HS), as ndarray of shape (m,n) ;
    * P: family of m points of size n belonging to the m hyperplans, as ndarray of shape (m,n) ;
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif the i-th couple in h=(P,V) is necessary for the construction of polyhedron Psi_h.
    """
    V_prime = V.copy()
    V_prime[i] = - V_prime[i]
    return polyhedron_is_fully_dimensional(V_prime, P, res)

def q_is_not_minor(q:np.ndarray, h:np.ndarray, p:Optional[np.ndarray]=None, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * q: any point in space, different from p, with shape (n,) ;
    * h: family of vector couples (c,v) defining half-spaces whose intersection is the polyhedron Psi_h to consider, with shape (m,2,n) ;
    * p: reference point outside Psi_h, with shape (n,) [zero by default];
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif the i-th couple in h=(P,V) is necessary for the construction of polyhedron Psi_h.
    """
    #p_q = normed(q-p) # not necessary (costs time), but better for precision!?
    if p is None:
        h_0 = np.array([q, q])[np.newaxis]
    else:
        h_0 = np.array([q, q-p])[np.newaxis]
    new_h = np.concatenate((h_0, h), axis=0, dtype=h.dtype)
    return polyhedron_is_fully_dimensional(new_h[:,1], new_h[:,0], res)

def keep_only_necessary_couples(h:np.ndarray, eps:Optional[float]=None) -> np.ndarray:
    i = h.shape[0] - 1
    while i >= 0:
        if not couple_is_necessary(i, h[:,1], h[:,0], eps):
            h = np.delete(h, i, axis=0)
        i -= 1
    return h

def keep_only_necessary_couples_idx(h:np.ndarray, eps:Optional[float]=None) -> np.ndarray:
    idx = np.ones(shape=h.shape[0], dtype=bool)
    i = h.shape[0] - 1
    while i >= 0:
        if not couple_is_necessary(i, h[:,1], h[:,0], eps):
            h = np.delete(h, i, axis=0)
            idx[i] = False
        i -= 1
    return idx

#%%
# Main function to compute distance from point 'p' to polyhedron 'P_h'. 
# Four versions: algo_0, algo_1, algo_2, algo_3. Fastest one is algo_0.
###

##
# Version 0
##

def __f0(h_original:np.ndarray, q:np.ndarray, h:np.ndarray, U:np.ndarray, eps:float=1e-6, steps:List=[], step:int=0) -> Tuple[np.ndarray, bool]:
    # p_original = 0
    # NO condition on family (v_i)_i
    steps.append(step)

    q_not_in_psi = np.max(scalar(q - h_original[:,0], h_original[:,1])) > eps
    U_not_fullsize = U.shape[0] < q.shape[0]
    h_not_empty = h.shape[0] > 0

    if q_not_in_psi and U_not_fullsize and h_not_empty: # we can go further in the projections!
        
        proj = scalar(q - h[:,0], h[:,1])
        positive = proj > eps # there is at least one True element, as max(proj) > 0

        v_prime = h[:,1] - scalar(scalar(h[:,1,np.newaxis], U, keepdims=True).transpose(0,2,1), U.T) # (|h|, n)
        v_prime_norm = norm(v_prime) # (|h|,)
        v_independant = v_prime_norm > eps # are the couples (c,v) in h for which v is linearly independant from U

        positive_and_independant = positive * v_independant

        if positive_and_independant.sum() == 0: # no couple for which proj is positive and v is linearly independant from U: we turn back!
            return q, False

        v_prime = v_prime[positive_and_independant] / v_prime_norm[positive_and_independant, np.newaxis] # (nbi, n)
        distances = proj[positive_and_independant] / scalar(h[positive_and_independant,1], v_prime) # == <q-c',v'>
        q_p_list = q - distances[..., np.newaxis] * v_prime # q projected on each of the positive AND independant hyperplanes

        IDX_order = list(np.argsort(distances)[::-1]) ### ARG on h_p
        IDX_pos_and_indep = np.argwhere(positive_and_independant) ### ARG on h

        q_p = q
        state = False

        while state is False and len(IDX_order) > 0:
            
            i = IDX_order.pop(0)
            v_independant[IDX_pos_and_indep[i]] = False

            q_p = q_p_list[i]
            h_p = h[v_independant]
            U_p = np.concatenate((U, [v_prime[i]]), axis=0)
            
            q_p, state = __f0(h_original, q_p, h_p, U_p, eps, steps, step+1)

        if state:
            return q_p, state
        return q, state
    
    elif q_not_in_psi:
        return q, False
    
    elif U.shape[0] <= 1: # q is necessarily minor, because only projected on one hyperplane at max!
        return q, True

    elif q_is_not_minor(q, h_original, res = 1e1 * eps): # only 1e1 (* ...) ?
        return q, False
    
    return q, True # q is in Psi AND is minor

def algo_0(h:np.ndarray, p:Optional[np.ndarray]=None, eps:Optional[float]=None, verify_h:bool=False) -> np.ndarray:
    """
    original
    """
    # precision: must be non-negative
    # verify_h: if False, no checking on the state of h and no reduction of h (Ps_i is supposed to be fully-dimensional)

    n = h.shape[-1]
    
    if p is None:
        p = np.zeros(n)

    if np.prod(h.shape) == 0: # h is empty
        return p, []

    v_norm = norm(h[:,1])
    nonull = v_norm > mres(h)

    if nonull.sum() == 0: # all v are null
        return p, []

    h = h[nonull]
    h[:,0] = h[:,0] - p
    h[:,1] = h[:,1] / v_norm[nonull,np.newaxis]
    
    if eps is None:
        eps = min(1e-6, mres(h, 1 + max(0, int(np.log10(n * np.max(np.abs(h[:,0]))))))) # is it a good 'eps' ???
    
    if np.min(scalar(h[:,0], h[:,1])) >= -eps:
        return p, []

    if not verify_h:

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f0(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps
    
    elif polyhedron_is_fully_dimensional(h[:,1], h[:,0], eps): # polyhedron_is_fully_dimensional: costs time!
        
        h = keep_only_necessary_couples(h, eps) # keep_only_necessary_couples: costs time!

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f0(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps

    elif True: # TODO: Case where Int(Psi) is empty, but not Psi!
        ... # TODO: also remove unnecessary Polyhedron by checking if their Hyperplane touch Psi (eps < 0)
        print("Warning: polyhedron is not fully-dimensional!")
        return None, []
    
    else:
        print("Warning: polyhedron is empty!")
        return np.full(shape=p.shape, fill_value=np.nan), []
    
    # TODO: if several p, once a 'q' is computed, we only have to check q_is_valid(q, p_i)
    #       to know if q is also the closest norm point to p_i, for eahc p_i in p (sort q_i by their distance to p_computed)

##
# Version 1
##

def __f1(h_original:np.ndarray, q:np.ndarray, h:np.ndarray, U:np.ndarray, eps:float=1e-6, steps:List=[], step:int=0) -> Tuple[np.ndarray, bool]:
    # p_original = 0
    # NO condition on family (v_i)_i
    steps.append(step)

    q_not_in_psi = np.max(scalar(q - h_original[:,0], h_original[:,1])) > eps
    U_not_fullsize = U.shape[0] < q.shape[0]
    h_not_empty = h.shape[0] > 0

    if q_not_in_psi and U_not_fullsize and h_not_empty: # we can go further in the projections!

        proj = scalar(q - h[:,0], h[:,1])
        positive = proj > eps # there is at least one True element, as max(proj) > 0

        v_prime = h[:,1] - scalar(scalar(h[:,1,np.newaxis], U, keepdims=True).transpose(0,2,1), U.T) # (|h|, n)
        v_prime_norm = norm(v_prime) # (|h|,)
        v_independant = v_prime_norm > eps # are the couples (c,v) in h for which v is linearly independant from U

        positive_and_independant = positive * v_independant

        if positive_and_independant.sum() == 0: # no couple for which proj is positive and v is linearly independant from U: we turn back!
            return q, False

        v_prime = v_prime[v_independant] / v_prime_norm[v_independant, np.newaxis] # (nbi, n)
        distances = proj[v_independant] / scalar(h[v_independant,1], v_prime) # == <q-c',v'>
        q_p_list = q - distances[..., np.newaxis] * v_prime # q projected on each of the positive AND independant hyperplanes

        touchPsi = keep_only_necessary_couples_idx(np.transpose((q_p_list, v_prime), axes=(1,0,2)), eps) # costs time to compute!!!
        positive_and_independant[v_independant] = touchPsi #TODO: verify that the length is not zero???
        v_independant[v_independant] = touchPsi
        v_prime = v_prime[touchPsi]
        distances = distances[touchPsi]
        q_p_list = q_p_list[touchPsi]

        IDX_pos_and_indep = np.argwhere(positive_and_independant) ### ARG on h

        remaining_space_is_2D_or_less = U.shape[0] >= q.shape[0] - 2
        
        if remaining_space_is_2D_or_less:
            
            i = np.argmax(distances)
            v_independant[IDX_pos_and_indep[i]] = False

            q_p = q_p_list[i]
            h_p = h[v_independant] # all hyperplanes must touch Psi_h_new!
            U_p = np.concatenate((U, [v_prime[i]]), axis=0)
            
            q_p, state = __f1(h_original, q_p, h_p, U_p, eps, steps, step+1)
        
        else:

            IDX_order = list(np.argsort(distances)[::-1]) ### ARG on h_p

            q_p = q
            state = False

            while state is False and len(IDX_order) > 0:
                
                i = IDX_order.pop(0)
                v_independant[IDX_pos_and_indep[i]] = False

                q_p = q_p_list[i]
                h_p = h[v_independant]
                U_p = np.concatenate((U, [v_prime[i]]), axis=0)
                
                q_p, state = __f1(h_original, q_p, h_p, U_p, eps, steps, step+1)

        if state:
            return q_p, state
        return q, state
    
    elif q_not_in_psi:
        return q, False
    
    elif U.shape[0] <= 1: # q is necessarily minor, because only projected on one hyperplane at max!
        return q, True

    elif q_is_not_minor(q, h_original, res = 1e1 * eps): # only 1e1 (* ...) ?
        return q, False
    
    return q, True # q is in Psi AND is minor

def algo_1(h:np.ndarray, p:Optional[np.ndarray]=None, eps:Optional[float]=None, verify_h:bool=False) -> np.ndarray:
    """
    original 
    + remove unnecessary couples at each step AND only take the maximum in 2D and in 1D
    """
    # precision: must be non-negative
    # verify_h: if False, no checking on the state of h and no reduction of h (Ps_i is supposed to be fully-dimensional)

    n = h.shape[-1]
    
    if p is None:
        p = np.zeros(n)

    if np.prod(h.shape) == 0: # h is empty
        return p, []

    v_norm = norm(h[:,1])
    nonull = v_norm > mres(h)

    if nonull.sum() == 0: # all v are null
        return p, []

    h = h[nonull]
    h[:,0] = h[:,0] - p
    h[:,1] = h[:,1] / v_norm[nonull,np.newaxis]
    
    if eps is None:
        eps = min(1e-6, mres(h, 1 + max(0, int(np.log10(n * np.max(np.abs(h[:,0]))))))) # is it a good 'eps' ???
    
    if np.min(scalar(h[:,0], h[:,1])) >= -eps:
        return p, []

    if not verify_h:

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f1(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps
    
    elif polyhedron_is_fully_dimensional(h[:,1], h[:,0], eps): # polyhedron_is_fully_dimensional: costs time!
        
        #h = keep_only_necessary_couples(h, eps) # not needed anymore, as it is made in __f1! # keep_only_necessary_couples: costs time!

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f1(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps

    elif True: # TODO: Case where Int(Psi) is empty, but not Psi!
        ... # TODO: also remove unnecessary Polyhedron by checking if their Hyperplane touch Psi (eps < 0)
        print("Warning: polyhedron is not fully-dimensional!")
        return None, []
    
    else:
        print("Warning: polyhedron is empty!")
        return np.full(shape=p.shape, fill_value=np.nan), []
    
    # TODO: if several p, once a 'q' is computed, we only have to check q_is_valid(q, p_i)
    #       to know if q is also the closest norm point to p_i, for eahc p_i in p (sort q_i by their distance to p_computed)

###
# Version 2
###

def __f2(h_original:np.ndarray, q:np.ndarray, h:np.ndarray, U:np.ndarray, eps:float=1e-6, steps:List=[], step:int=0) -> Tuple[np.ndarray, bool]:
    # p_original = 0
    # NO condition on family (v_i)_i
    steps.append(step)

    q_is_minor = U.shape[0] <= 1
    if not q_is_minor:
        q_is_minor = not q_is_not_minor(q, h_original, res = 1e1 * eps) # only 1e1 (* ...) ?
    q_not_in_psi = np.max(scalar(q - h_original[:,0], h_original[:,1])) > eps
    U_not_fullsize = U.shape[0] < q.shape[0]
    h_not_empty = h.shape[0] > 0

    if q_not_in_psi and q_is_minor and U_not_fullsize and h_not_empty: # we can go further in the projections!

        proj = scalar(q - h[:,0], h[:,1])
        positive = proj > eps # there is at least one True element, as max(proj) > 0

        v_prime = h[:,1] - scalar(scalar(h[:,1,np.newaxis], U, keepdims=True).transpose(0,2,1), U.T) # (|h|, n)
        v_prime_norm = norm(v_prime) # (|h|,)
        v_independant = v_prime_norm > eps # are the couples (c,v) in h for which v is linearly independant from U

        positive_and_independant = positive * v_independant

        if positive_and_independant.sum() == 0: # no couple for which proj is positive and v is linearly independant from U: we turn back!
            return q, False

        v_prime = v_prime[v_independant] / v_prime_norm[v_independant, np.newaxis] # (nbi, n)
        distances = proj[v_independant] / scalar(h[v_independant,1], v_prime) # == <q-c',v'>
        q_p_list = q - distances[..., np.newaxis] * v_prime # q projected on each of the positive AND independant hyperplanes

        touchPsi = keep_only_necessary_couples_idx(np.transpose((q_p_list, v_prime), axes=(1,0,2)), eps) # costs time to compute!!!
        positive_and_independant[v_independant] = touchPsi #TODO: verify that the sum is not zero???
        v_independant[v_independant] = touchPsi
        v_prime = v_prime[touchPsi]
        distances = distances[touchPsi]
        q_p_list = q_p_list[touchPsi]

        IDX_pos_and_indep = np.argwhere(positive_and_independant) ### ARG on h

        remaining_space_is_2D_or_less = U.shape[0] >= q.shape[0] - 2
        
        if remaining_space_is_2D_or_less:
            
            i = np.argmax(distances)
            v_independant[IDX_pos_and_indep[i]] = False

            q_p = q_p_list[i]
            h_p = h[v_independant] # all hyperplanes must touch Psi_h_new!
            U_p = np.concatenate((U, [v_prime[i]]), axis=0)
            
            q_p, state = __f2(h_original, q_p, h_p, U_p, eps, steps, step+1)
        
        else:

            IDX_order = list(np.argsort(distances)[::-1]) ### ARG on h_p

            q_p = q
            state = False

            while state is False and len(IDX_order) > 0:
                
                i = IDX_order.pop(0)
                v_independant[IDX_pos_and_indep[i]] = False

                q_p = q_p_list[i]
                h_p = h[v_independant]
                U_p = np.concatenate((U, [v_prime[i]]), axis=0)
                
                q_p, state = __f2(h_original, q_p, h_p, U_p, eps, steps, step+1)

        if state:
            return q_p, state
        return q, state
    
    elif q_not_in_psi or not q_is_minor:
        return q, False
    
    return q, True # q is in Psi AND is minor

def algo_2(h:np.ndarray, p:Optional[np.ndarray]=None, eps:Optional[float]=None, verify_h:bool=False) -> np.ndarray:
    """
    original 
    + remove unnecessary couples at each step AND only take the maximum in 2D and in 1D 
    + check if q is minor at each step to continue
    """
    # precision: must be non-negative
    # verify_h: if False, no checking on the state of h and no reduction of h (Ps_i is supposed to be fully-dimensional)

    n = h.shape[-1]
    
    if p is None:
        p = np.zeros(n)

    if np.prod(h.shape) == 0: # h is empty
        return p, []

    v_norm = norm(h[:,1])
    nonull = v_norm > mres(h)

    if nonull.sum() == 0: # all v are null
        return p, []

    h = h[nonull]
    h[:,0] = h[:,0] - p
    h[:,1] = h[:,1] / v_norm[nonull,np.newaxis]
    
    if eps is None:
        eps = min(1e-6, mres(h, 1 + max(0, int(np.log10(n * np.max(np.abs(h[:,0]))))))) # is it a good 'eps' ???
    
    if np.min(scalar(h[:,0], h[:,1])) >= -eps:
        return p, []

    if not verify_h:

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f2(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps
    
    elif polyhedron_is_fully_dimensional(h[:,1], h[:,0], eps): # polyhedron_is_fully_dimensional: costs time!
        
        #h = keep_only_necessary_couples(h, eps) # not needed anymore, as it is made in __f1! # keep_only_necessary_couples: costs time!

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f2(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps

    elif True: # TODO: Case where Int(Psi) is empty, but not Psi!
        ... # TODO: also remove unnecessary Polyhedron by checking if their Hyperplane touch Psi (eps < 0)
        print("Warning: polyhedron is not fully-dimensional!")
        return None, []
    
    else:
        print("Warning: polyhedron is empty!")
        return np.full(shape=p.shape, fill_value=np.nan), []
    
    # TODO: if several p, once a 'q' is computed, we only have to check q_is_valid(q, p_i)
    #       to know if q is also the closest norm point to p_i, for eahc p_i in p (sort q_i by their distance to p_computed)

###
# Version 3
###

def __f3(h_original:np.ndarray, q:np.ndarray, h:np.ndarray, U:np.ndarray, eps:float=1e-6, steps:List=[], step:int=0) -> Tuple[np.ndarray, bool]:
    # p_original = 0
    # NO condition on family (v_i)_i
    steps.append(step)

    q_is_minor = U.shape[0] <= 1
    if not q_is_minor:
        q_is_minor = not q_is_not_minor(q, h_original, res = 1e1 * eps) # only 1e1 (* ...) ?
    q_not_in_psi = np.max(scalar(q - h_original[:,0], h_original[:,1])) > eps
    U_not_fullsize = U.shape[0] < q.shape[0]
    h_not_empty = h.shape[0] > 0

    if q_not_in_psi and q_is_minor and U_not_fullsize and h_not_empty: # we can go further in the projections!
        
        proj = scalar(q - h[:,0], h[:,1])
        positive = proj > eps # there is at least one True element, as max(proj) > 0

        v_prime = h[:,1] - scalar(scalar(h[:,1,np.newaxis], U, keepdims=True).transpose(0,2,1), U.T) # (|h|, n)
        v_prime_norm = norm(v_prime) # (|h|,)
        v_independant = v_prime_norm > eps # are the couples (c,v) in h for which v is linearly independant from U

        positive_and_independant = positive * v_independant

        if positive_and_independant.sum() == 0: # no couple for which proj is positive and v is linearly independant from U: we turn back!
            return q, False

        v_prime = v_prime[positive_and_independant] / v_prime_norm[positive_and_independant, np.newaxis] # (nbi, n)
        distances = proj[positive_and_independant] / scalar(h[positive_and_independant,1], v_prime) # == <q-c',v'>
        q_p_list = q - distances[..., np.newaxis] * v_prime # q projected on each of the positive AND independant hyperplanes

        IDX_order = list(np.argsort(distances)[::-1]) ### ARG on h_p
        IDX_pos_and_indep = np.argwhere(positive_and_independant) ### ARG on h

        q_p = q
        state = False

        while state is False and len(IDX_order) > 0:
            
            i = IDX_order.pop(0)
            v_independant[IDX_pos_and_indep[i]] = False

            q_p = q_p_list[i]
            h_p = h[v_independant]
            U_p = np.concatenate((U, [v_prime[i]]), axis=0)
            
            q_p, state = __f3(h_original, q_p, h_p, U_p, eps, steps, step+1)

        if state:
            return q_p, state
        return q, state
    
    elif q_not_in_psi or not q_is_minor:
        return q, False
    
    return q, True # q is in Psi AND is minor

def algo_3(h:np.ndarray, p:Optional[np.ndarray]=None, eps:Optional[float]=None, verify_h:bool=False) -> np.ndarray:
    """
    original 
    + check if q is minor at each step to continue
    """
    # precision: must be non-negative
    # verify_h: if False, no checking on the state of h and no reduction of h (Ps_i is supposed to be fully-dimensional)

    n = h.shape[-1]
    
    if p is None:
        p = np.zeros(n)

    if np.prod(h.shape) == 0: # h is empty
        return p, []

    v_norm = norm(h[:,1])
    nonull = v_norm > mres(h)

    if nonull.sum() == 0: # all v are null
        return p, []

    h = h[nonull]
    h[:,0] = h[:,0] - p
    h[:,1] = h[:,1] / v_norm[nonull,np.newaxis]
    
    if eps is None:
        eps = min(1e-6, mres(h, 1 + max(0, int(np.log10(n * np.max(np.abs(h[:,0]))))))) # is it a good 'eps' ???
    
    if np.min(scalar(h[:,0], h[:,1])) >= -eps:
        return p, []

    if not verify_h:

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f3(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps
    
    elif polyhedron_is_fully_dimensional(h[:,1], h[:,0], eps): # polyhedron_is_fully_dimensional: costs time!
        
        h = keep_only_necessary_couples(h, eps) # keep_only_necessary_couples: costs time!

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f3(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps

    elif True: # TODO: Case where Int(Psi) is empty, but not Psi!
        ... # TODO: also remove unnecessary Polyhedron by checking if their Hyperplane touch Psi (eps < 0)
        print("Warning: polyhedron is not fully-dimensional!")
        return None, []
    
    else:
        print("Warning: polyhedron is empty!")
        return np.full(shape=p.shape, fill_value=np.nan), []
    
    # TODO: if several p, once a 'q' is computed, we only have to check q_is_valid(q, p_i)
    #       to know if q is also the closest norm point to p_i, for eahc p_i in p (sort q_i by their distance to p_computed)

#%%
# MAIN FUNCTION TO COMPUTE THE MINIMUM-NORM POINTS TO CONVEX POLYHEDRA (Python version -> very slow!)
###

def data_in_polyhedron(data:np.ndarray, polyhedron:np.ndarray) -> np.ndarray:
    eps = np.finfo(data.dtype).resolution
    data_in_poly = np.max(scalar(data[np.newaxis] - polyhedron[:,0,np.newaxis], polyhedron[:,1,np.newaxis]), axis=0) < eps
    return data_in_poly

def minimum_norm_points_to_polyhedra_PYTHON(data:np.ndarray, h:list[np.ndarray], method:Literal['0','1','2','3']='0', infos:bool=True) -> np.ndarray:
    """
    Main Python function to compute the minimum-norm points to convex polyhedra. 
    Default version of the algorithm is 'algo_0'.
    * data: ndarray of points in ndim-dimensional real vector space,
     with shape (n_samples, ndim) or (ndim,) ;
    * h: list of polyhedra represented as intersection of half_spaces described by couples (c,v),
     with shape n_classes * (n_half_spaces, 2, ndim) or (n_half_spaces, 2, ndim).\n
    Returns ndarray of minimum-norm points from data to each class polyhedra, 
    with shape (n_samples, n_classes, ndim) or (n_classes, ndim) or (n_samples, ndim) or (ndim,).
    """
    data_is_1D = data.ndim == 1
    h_is_3D = type(h) is np.ndarray and h.ndim == 3

    if data_is_1D:
        data = data[np.newaxis]
    if h_is_3D:
        h = h[np.newaxis]
    
    if '0' in method:
        algo = algo_0
    elif '1' in method:
        algo = algo_1
    elif '2' in method:
        algo = algo_2
    elif '3' in method:
        algo = algo_3
    else:
        algo = algo_0

    n_samples, ndim = data.shape
    n_classes = len(h)

    min_n_pts = np.empty(shape=(n_samples, n_classes, ndim), dtype=data.dtype)

    estimated = False
    start = time.time()

    # For each polyhedron in h, compute the minimum-norm points for all points in data
    for c in range(n_classes):

        if infos:
            print(f"* Processing class: {c+1} / {n_classes}...", end=' ')
        
        # Get half-space vector couples forming the polyhedron related to class c
        h_class = h[c]

        # Directly associate p themselves to points p which belong to h_class
        p_in_h = data_in_polyhedron(data=data, polyhedron=h_class)
        min_n_pts[p_in_h,c] = data[p_in_h]
        arg_data_not_in_h = np.argwhere(~p_in_h)[:,0]
        n_samples_not_in_h = arg_data_not_in_h.shape[0]

        # Compute corresponding minimum-norm points q to the other points p
        for i in range(n_samples_not_in_h):
            
            # Estimate the computation time from the already-computed sample
            if infos and not estimated and time.time() - start > 1 and c * n_samples + i > 0:
                delta_time = time.time() - start
                total_iter = n_samples * n_classes
                n_iter_now = c * n_samples + i
                total_time = total_iter / n_iter_now * delta_time
                if total_time > 60:
                    total_time = total_time / 60
                    if total_time > 60:
                        total_time = total_time / 60
                        unit = "hours"
                    else:
                        unit = "minutes"
                else:
                    unit = "seconds"
                decimals = 2
                total_time = np.round(total_time, decimals)
                print(f"\nEstimated computation time: {total_time} {unit}")
                estimated = True
            
            # Compute the minimum-norm point q from p to h_class with algo
            arg = arg_data_not_in_h[i]
            p = data[arg]
            q, _ = algo(h_class, p, eps=1e-8)
            min_n_pts[arg,c] = q

        if infos:
            print("Done!")
    
    if data_is_1D:
        min_n_pts = min_n_pts[0]
        if h_is_3D:
            min_n_pts = min_n_pts[0]
    elif h_is_3D:
        min_n_pts = min_n_pts[:,0]

    return min_n_pts

#%% ####################################################################################################
# Functions to generate and display polyhedron using H-representation (intersection of half-spaces)
###

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

def generate_half_spaces(m:int, n:int=2, radius:float=100, disparity:float=0.5) -> np.ndarray:
    """
    Randomly generate m half-space couples in vector space R**n.\n
    - m: number of half_spaces ;
    - n: dimension of vector space ;
    - radius: max distance to zero point ;
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
            max_vec = np.zeros(shape=(n,), dtype=h.dtype)
        else:
            max_vec = mean_unit_vector(h[:i,0])

        random_vec_c = np.random.rand(n) * 2 - 1
        while np.isclose(norm(random_vec_c),0):
            random_vec_c = np.random.rand(n) * 2 - 1
        random_vec_c = random_vec_c / norm(random_vec_c)

        vec_c = c_stoch_val * random_vec_c + (1-c_stoch_val) * c_stoch_sgn * max_vec

        if np.isclose(norm(vec_c),0):
            vec_c = random_vec_c
        else:
            vec_c = vec_c / norm(vec_c)

        c = vec_c * radius# max(np.abs(np.random.normal(radius, radius * c_stoch_val)), min_float)

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

        c = v * radius

        h[i] = (c,v)
    return h

def visualize_2D(h:np.ndarray, random_color:bool=False, thickness:float=1, resolution:int=1000, ax:Any=None, window_size:int=8) -> None:
    """
    Function to visualize 2D polyhedron defined by family h.
    """
    if h.shape[-1] != 2:
        print("WARNING: space is not 2-dimensional.")
        return None

    if random_color:
        img = np.zeros(shape=(resolution,resolution,3), dtype=np.uint8)
        img_flat = img.reshape(resolution**2,3)
    else:
        img = np.full(shape=(resolution,resolution), fill_value=255, dtype=np.uint8)
        img_flat = img.reshape(resolution**2)

    x = np.arange(resolution)
    yc, xc = np.meshgrid(x,x)
    xv = xc.reshape(resolution**2)
    yv = yc.reshape(resolution**2)
    coord = np.array([xv,yv]).swapaxes(0,1)

    hs = np.empty(shape=(len(h), resolution**2))

    # plot hyperplanes
    if random_color:
        color = (np.random.rand(len(h),3)*255).astype(np.uint8)
    else:
        color = [0]*len(h)
    for i in range(len(h)):
        ci, vi = h[i]

        proj_i = np.sum((coord - ci) * vi, axis=-1)
        hyperplane_i = np.abs(proj_i) <= thickness
        img_flat[hyperplane_i] = color[i]

        half_space_i = proj_i < -thickness
        hs[i] = half_space_i
    
    # plot polyhedron
    gray_val = 0.7 - 0.4 * random_color
    color_hs = [np.uint8(int(255*gray_val))] * (3 * random_color - random_color + 1)
    polyhedron = np.prod(hs, axis=0) > 0
    img_flat[polyhedron] = color_hs

    if ax is None:
        plt.figure(figsize=(window_size,window_size))
        plt.gca().set_aspect('equal')
        if random_color: plt.imshow(img)
        else: plt.imshow(img, cmap='gray')
        plt.show()
    else:
        if random_color: ax.imshow(img)
        else: ax.imshow(img, cmap='gray')

#%% ####################################################################################################
# !!! WORK IN PROGRESS !!!
# Function to compute V-representation from H-representation (WARNING: computationaly expensive!)
###

import itertools

def from_H_to_V_representation(h:np.ndarray, precision:Optional[float]=None) -> np.ndarray:
    """
    Variables:
    * h: array of vector couples (c,v), with shape (m,2,n) ;\n
    * precision: adapted positive float to check the linear independancy of vectors and the belonging of vertices to Psi_h.\n
    m: number of half-spaces | n: number of dimensions in space.\n
    => Returns v, array of all the k vertices of the polyhedron Psi_h, with shape (k,n).
    """
    
    if h.shape[0] == 0:
        print("Warning: array h of vector couples (c,v) is empty! Polyhedron Psi_h is the entire space V. No possible vertex to compute.")
        return np.zeros(shape=(0,n), dtype=h.dtype)

    h_norm_non_null = norm(h[:,1]) > 0
    if h_norm_non_null.sum() == 0:
        print("Warning: array h contains no couple (c,v) for which v is non zero! Polyhedron Psi_h is the entire space V. No possible vertex to compute.")
        return np.zeros(shape=(0,n), dtype=h.dtype)

    h = h[h_norm_non_null] # remove couples (c,v) for which v is zero (creates new array h)

    if polyhedron_is_fully_dimensional(h[:,1], h[:,0]): # Case where Int(Psi) is not empty!
        h = keep_only_necessary_couples(h) # remove unnecessary couples if Psi is fully-dim
    ... # TODO: Case where Int(Psi) is empty, but not Psi!
    
    A, b = to_half_space_inequality(h) # a and b vectors are well defined, as no v is zero!

    m, n = A.shape
    if m < n:
        print("Warning: the number of couples is lower than the number of dimensions. Polyhedron Psi_i contains no vertex. No possible vertex to compute.")
        return np.zeros(shape=(0,n), dtype=h.dtype)

    arange = np.arange(m)
    indices = list(np.asarray(list(itertools.combinations(arange, n))))

    if precision is None:
        precision_1 = mres(A, 1 + max(0, int(np.log10(n*np.max(np.abs(A)))))) # precision for linear independancy validation
        precision_2 = mres(h, 1 + max(0, int(np.log10(n*np.max(np.abs(h)))))) # precision for belonging to Psi_h validation
    else:
        precision_1 = precision # precision for linear independancy validation
        precision_2 = precision # precision for belonging to Psi_h validation

    v = []
    for idc in indices:
        # solution: x = - M^(-1) * b_
        M = A[idc]
        det_M = np.linalg.det(M)
        # CONDITION 1: all the vectors in M must be linearly independant
        if np.abs(det_M) > precision_1:
            M_inv = np.linalg.inv(M)
            b_sel = b[idc]
            vertex = - M_inv @ b_sel
            # CONDITION 2: the computed vertex must belong to Psi_h to effectively be one of its vertices
            if np.max(scalar(vertex - h[:,0], h[:,1])) <= precision_2:
                v.append(vertex)
    
    if len(v) <= n:
        print("Polytope resulted from computed vertices is not fully-dimensional!")
    
    return np.asarray(v)

#%% ####################################################################################################
# !!! WORK IN PROGRESS !!!
# WOLFE algorithm for the nearest point problem in convex polyhedron (works on V-representation only!)
###

from typing import Literal
from scipy.optimize import lsq_linear

def wolfe(
        P:np.ndarray, 
        initRule:Literal['first','minnorm']='minnorm', 
        addRule:Literal['first','minnorm','linopt']='linopt', 
        displayOn:Literal['on','off']='off'
    ) -> Tuple[np.ndarray, np.ndarray]:
    # P: m x n matrix containing points as rows (m points in R^n)
    # The point from which the minimum norm point in a convex polytope is computed is considered as the null point 0_V
    
    data = []
    corrals = []
    corralnum = 0

    majorCycle = 0
    minorCycle = 0

    if displayOn not in ['on', 'off']:
        print('Check your displayOn value.')
        return float('inf'), data

    if initRule == 'first':
        i = 0
    elif initRule == 'minnorm':
        i = np.argmin(norm(P))
    else:
        print('Check your initRule value.')
        return float('inf'), data
    
    x = P[i, :]
    C = np.array([x])
    lambda_ = np.array([1])
    Cind = [i]

    if displayOn == 'on':
        print(f'Step 0: {majorCycle} {minorCycle} {x}')
        print(np.column_stack((Cind, C)))

    data.append([majorCycle + 0.1 * minorCycle, C.shape[0]])
    corrals.append(Cind)
    corralnum += 1

    while (np.linalg.norm(x, 2) >= np.finfo(float).eps) and (np.min(P @ x) + np.finfo(float).eps < x @ x):
        Cprevious = C.copy()

        majorCycle += 1
        minorCycle = 0

        indOptions = np.where(P @ x < x @ x)[0]

        if addRule == 'first':
            i = 0
            while indOptions.size > 0 and indOptions[i] in Cind:
                indOptions = indOptions[1:]
        elif addRule == 'minnorm':
            i = np.argmin(norm(P[indOptions, :]))
            while indOptions.size > 0 and indOptions[i] in Cind:
                indOptions = np.delete(indOptions, i)
                if indOptions.size > 0:
                    i = np.argmin(norm(P[indOptions, :]))
        elif addRule == 'linopt':
            i = np.argmin(P[indOptions, :] @ x)
            while indOptions.size > 0 and indOptions[i] in Cind:
                indOptions = np.delete(indOptions, i)
                if indOptions.size > 0:
                    i = np.argmin(P[indOptions, :] @ x)
        else:
            print('Check your addRule value.')
            return float('inf'), data
        
        if indOptions.size > 0:
            j = indOptions[i]
            C = np.vstack([C, P[j, :]])  # add this vertex to potential corral
            lambda_ = np.append(lambda_, 0)
            Cind.append(j)

        if Cprevious.shape == C.shape and np.array_equal(Cprevious, C):
            solution = x
            print('Repeated a corral.')
            return solution, data

        if displayOn == 'on':
            print(f'Step 1: {majorCycle} {minorCycle} {x}')
            print(np.column_stack((Cind, C)))

        k = C.shape[0]
        M_ = np.concatenate((np.array([0] + [1] * k)[:,np.newaxis], np.concatenate((np.ones((1,k)), C @ C.T), axis=0)), axis=1)
        b_ = np.array([1] + [0] * k)
        alpha = lsq_linear(M_, b_).x[1:]
        y = (C.T @ alpha)

        if displayOn == 'on':
            print(f'Step 2: {majorCycle} {minorCycle} {x} {y}')
            print(np.column_stack((Cind, C)))

        data.append([majorCycle + 0.1 * minorCycle, C.shape[0]])

        while np.min(alpha) < -np.finfo(float).eps:
            minorCycle += 1
            negind = np.where(alpha < -np.finfo(float).eps)[0]
            lambda_ = lsq_linear(C.T, x).x
            theta = np.min(lambda_[negind] / (lambda_[negind] - alpha[negind]))

            x = (C.T @ (theta * alpha + (1 - theta) * lambda_))

            i = np.where(theta * alpha + (1 - theta) * lambda_ <= np.finfo(float).eps)[0]
            lambda_ = theta * alpha + (1 - theta) * lambda_

            if i.size > 0:
                C = np.delete(C, i[0], axis=0)
                lambda_ = np.delete(lambda_, i[0])
                Cind.pop(i[0])
            else:
                i = negind[0]
                C = np.delete(C, i, axis=0)
                lambda_ = np.delete(lambda_, i)
                Cind.pop(i)

            if displayOn == 'on':
                print(f'Step 3: {majorCycle} {minorCycle} {x}')
                print(np.column_stack((Cind, C)))

            k = C.shape[0]
            M_ = np.concatenate((np.array([0] + [1] * k)[:,np.newaxis], np.concatenate((np.ones((1,k)), C @ C.T), axis=0)), axis=1)
            b_ = np.array([1] + [0] * k)
            alpha = lsq_linear(M_, b_).x[1:]
            y = (C.T @ alpha)

            if displayOn == 'on':
                print(f'Step 2: {majorCycle} {minorCycle} {x} {y}')
                print(np.column_stack((Cind, C)))

            data.append([majorCycle + 0.1 * minorCycle, C.shape[0]])

        x = y
        lambda_ = alpha
        corrals.append(Cind)
        corralnum += 1

    solution = x
    if displayOn == 'on':
        print(f'Corrals: {len(corrals)}')
        for i in range(len(corrals)):
            print(corrals[i])

    return solution, data

