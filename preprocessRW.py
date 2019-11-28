from scipy.sparse import diags
import scipy as sp
import numpy as np


def computeRep(G,K,c):
    A = G
    
    
    degreeDict = np.sum(G, axis=1)
    
    InvdegreeList = []
    for i in range(len(degreeDict)):
        if degreeDict[i]==0:
            degreeDict[i] = 0.1 #Any number can be used, just to avoid the error
        InvdegreeList.append(float(1)/degreeDict[i])
        
    InvDegree = diags(InvdegreeList)
    
    P_0 = diags( (np.ones((1,G.shape[0]))).tolist()[0] ) #Initialize P_0
    
    intermedMat = sp.sparse.csr_matrix(InvDegree)*sp.sparse.csr_matrix(A) #Matrix multiplication
    
    P = P_0
    for k in range(K):
        P = c * sp.sparse.csr_matrix(P)*sp.sparse.csr_matrix(intermedMat) + (1 - c) * P_0
        
    return P.todense()