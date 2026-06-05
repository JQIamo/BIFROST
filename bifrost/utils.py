import numpy as np
from scipy import optimize as opt
import numpy.typing as npt
import typing

'''
The following four functions are dedicated to converting Jones Vectors to Stokes Vectors
and vice versa. Since BIFROST does not simulate unpolarized light, both Stokes and Jones
vectors correspond to the same physical properties, defined here as delta and phi, which
correspond respectively to the phase difference between the defined axes and the arctangent
of the y-amplitude to the arctangent of the x-amplitude.

'''
def JonesVdeltaPhi (JonesV):

    #takes a jones vector and returns an array containing delta, phi in that order.

    return [-np.angle(JonesV[1][0]),np.arccos(np.abs(JonesV[0][0]))]

def StokesVdeltaPhi (StokesV):

     #takes a stokes vector object and returns a 2xN array with delta on the first row and phi on the bottom.
    delta_list = []
    phi_list = []

    if StokesV[1][0] ==1:
        delta_list.append(0)
        phi_list.append(0)
    else:
        delta_list.append(np.arctan2(StokesV[3][0],StokesV[2][0]))
        phi_list.append(1/2 * np.arccos(StokesV[1][0]))
    
    return [delta_list,phi_list]

def JonesVtoStokesV (JonesV):

    deltaphi = JonesVdeltaPhi(JonesV)

    return np.array([
                     [1],
                     [np.cos(2*deltaphi[1])],
                     [np.sin(2*deltaphi[1])*np.cos(deltaphi[0])],
                     [np.sin(2*deltaphi[1])*np.sin(deltaphi[0])]
                     ])

def StokesVtoJonesV (StokesV):

    deltaphi = StokesVdeltaPhi(StokesV)

    return np.array([
        [np.cos(deltaphi[1][0])],
        [np.conj(np.exp(1j*deltaphi[0][0])*np.sin(deltaphi[1][0]))]
    ])

'''
The following two functions convert between Jones matrices to Mueller matrices.
For the Jones to Mueller conversion, a simple set of matrix operations accomplishes
the task, while the Mueller to Jones conversion is a bit more involved. It uses the 
covariance matrix method developed by Cloude; it produces a covariance matrix from the
Mueller matrix provided (here named H) which is isomorphic to an equivalent Jones
matrix, and then derives the Jones matrix from said matrix.
'''

#defines the Pauli matrices. Maybe useful?

sig0 = np.array([[1,0],
                [0,1]])
sig1 = np.array([[1,0],
                 [0,-1]])
sig2 = np.array([[0,1],
                 [1,0]])
sig3 = np.array([[0,-1j],
                [1j,0]])

#Two functions which convert Jones to Mueller Matrices and vice versa.
def JonesMtoMuellerM (JonesM):

    #takes a Jones matrix,
    #returns a Mueller matrix.

    U = np.array([[1,0,0,1],
                  [1,0,0,-1],
                  [0,1,1,0],
                  [0,-1j,1j,0]])

    Jxj = np.kron(JonesM,JonesM.conj())

    step1 = np.matmul(U,Jxj)
    final = np.matmul(step1,np.linalg.inv(U))

    return final

def MuellerMtoJonesM (M):

    #takes a Mueller matrix,
    #defines a covariance matrix based off of said Mueller Matrix,
    #derives the Jones matrix from the eigenvector corresponding to 
    #its nonzero eigenvalue, and returns it.

    H = 0.25 * np.array([[
        M[0,0] + M[1,1] + M[2,2] + M[3,3],
        M[0,1] + M[1,0] + 1j*M[2,3] - 1j*M[3,2],
        M[0,2] + M[2,0] - 1j*M[1,3] + 1j*M[3,1],
        M[0,3] + M[3,0] + 1j*M[1,2] - 1j*M[2,1]
    ], [
        M[0,1] + M[1,0] - 1j*M[2,3] + 1j*M[3,2],
        M[0,0] + M[1,1] - M[2,2] - M[3,3],
        M[1,2] + M[2,1] - 1j*M[0,3] + 1j*M[3,0],
        M[1,3] + M[3,1] + 1j*M[0,2] - 1j*M[2,0]
    ],[
        M[0,2] + M[2,0] + 1j*M[1,3] - 1j*M[3,1],
        M[1,2] + M[2,1] + 1j*M[0,3] - 1j*M[3,0],
        M[0,0] - M[1,1] + M[2,2] - M[3,3],
        M[2,3] + M[3,2] - 1j*M[0,1] + 1j*M[1,0]
    ],[
        M[0,3] + M[3,0] - 1j*M[1,2] + 1j*M[2,1],
        M[1,3] + M[3,1] - 1j*M[0,2] + 1j*M[2,0],
        M[2,3] + M[3,2] + 1j*M[0,1] - 1j*M[1,0],
        M[0,0] - M[1,1] - M[2,2] + M[3,3]
    ]])

    if np.allclose(H, H.conj().T) == False:
        return "error: covariance not hermitian."
    else:

        eigenvals, eigenvects = np.linalg.eigh(H)
        max = -1

        if eigenvals[3] <= 1e-10:
            return "error: zero matrix"

    k = eigenvects[:, 3]
    k = k* np.sqrt(eigenvals[3])


    final = np.array([
        [k[0] + k[1], k[2] - 1j*k[3]],
        [k[2] + 1j*k[3], k[0] - k[1]]
    ])

    return final