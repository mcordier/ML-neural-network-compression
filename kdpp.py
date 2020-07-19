import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics.pairwise import polynomial_kernel, cosine_similarity, sigmoid_kernel, linear_kernel, laplacian_kernel
import scipy.linalg as la
from numpy.linalg import eig
import pdb
np.random.seed(1)




class kDPP():
    """

    Attributes
    ----------
    A : PSD/Symmetric Kernel


    Usage:
    ------

    >>> from pydpp.dpp import DPP
    >>> import numpy as np

    >>> X = np.random.random((10,10))
    >>> dpp = kDPP(X,kernel)
    >>> ksamples = kdpp.sample_k(5)
    """

    def __init__(self, X=None, kernel=None, **kwargs):
        self.kernel = kernel
        self.X = X
        self.K = None
        self.compute_kernel(self.kernel)
        self.eig = np.linalg.eigh(self.K)

    def compute_kernel(self, kernel_func, *args, **kwargs):
        if kernel_func==None:
            self.K = self.kernel_se(self.X, self.X, **kwargs)
        else:
            self.K = kernel_func(self.X, self.X, **kwargs)

    def sample(self, k=5):
        # Select _k eigen vectors (k elementary dpp)
        N,d = self.X.shape[0], self.X.shape[1]
        K = self.kernel(self.X,self.X)
        l_eig, v_eig = self.eig
        J = []
        l = k
        e = self.ele_sym_polynomials(l_eig, k)
        for n in range(N-1,-1,-1):
            if l==0:
                break
            u = np.random.uniform(0,1,1)
            if u < l_eig[n]*e[n][l-1]/e[n+1][l]:
                J += [n]
                l -= 1
        V = v_eig[J]
        Y = np.array([], dtype=int)

        # Sampling simulation
        while len(V) > 0:
            ### Random sampling of new element to add i, according to the distribution Pr
            Pr = np.sum((V@K)**2, axis=0)
            Pr = Pr/np.sum(Pr)
            i = np.random.choice(np.arange(len(K)), p = Pr)
            
            if i not in Y: # If i is already in the Y, we start again
                Y = np.append(Y, i)
                ei = K[i]
                
                ### Calculating subspace of V orthogonal to e_i
                # First, choose a vector with a non zero ith component.
                random_index = np.random.choice(np.arange(len(V)))
                vector_to_substract = V[random_index]
                scalar_product = vector_to_substract@ei
                while scalar_product == 0:
                    ### We assume that there is at least one eigenvector with a non zero ith coefficient
                    random_index = np.random.choice(np.arange(len(V)))
                    vector_to_substract = V[random_index]
                    scalar_product = vector_to_substract@ei
                mult_coeff = scalar_product
                subspace = np.delete(V, random_index, axis=0)
                multipliers = (subspace@ei)/mult_coeff
                subspace = subspace - np.outer(vector_to_substract, multipliers).T
    #             print(subspace@ei)
    #             print(X[Y], Y)
                if np.shape(subspace)[0] > 0:
                    V = la.orth(subspace.T).T
                else:
                    V = []
    #     print(Y)
        return self.X[Y], Y

    def ele_sym_polynomials(self, l_eig, k):
        N = len(l_eig)
        e = np.zeros((N+1,k+1))
        for n in range(N+1):
            e[n,0]=1
        for l in range(1, k+1):
            for n in range(1, N+1):
                e[n,l]=e[n-1,l] + l_eig[n-1]*e[n-1,l-1]
        return(e)

    def kernel_se(self, _X1,_X2,_hyp={'gain':1,'len':1,'noise':1e-8}):
        hyp_gain = float(_hyp['gain'])**2
        hyp_len  = 1/float(_hyp['len'])
        pairwise_dists = cdist(_X1,_X2,'euclidean')
        K = hyp_gain*np.exp(-pairwise_dists ** 2 / (hyp_len**2))
        return K