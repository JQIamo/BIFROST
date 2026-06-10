'''
A module for BIFROST containing
useful utility functions.

William Jin
Swarthmore '27
'''



import numpy as np
from scipy import optimize as opt
import numpy.typing as npt
import typing

 # Defines the four Pauli matrices.
 # While not used within utils.py itself, they are common
 # enough that we have provided them here.
 # ----------------------------------------
PAULI_I = np.array([[1,0],
                [0,1]])
PAULI_Z = np.array([[1,0],
                 [0,-1]])
PAULI_X = np.array([[0,1],
                 [1,0]])
PAULI_Y = np.array([[0,-1j],
                [1j,0]])

 # A function that normalizes a given vector into a unit vector.
 # Not used often, but not included in numpy (for some reason).
 
def normalize(vector):
    magnitudesq = 0
    for i in range(np.shape(vector)[0]):
        magnitudesq = magnitudesq + vector[i] ** 2
    
    mag = np.sqrt(magnitudesq)

    return 1/mag * vector

 # The following functions generate common Stokes/Jones vectors
 # and Jones/Mueller matrices. 

 # Linear light: Functions which return an np.array designating
 # light linearly polarized at some angle theta.

def linJonesVec (theta):
    """
    Creates a general linear polarized light state in the Jones parameterization.

    Parameters
    ----------
    theta: float
        Angle of polarization wrt. horizontal.

    Returns
    -------
    np.array[2]
        2-vector containing a set of corresponding Jones parameters for a linear
        polarized state at angle theta.
    """
    return np.array([np.cos(theta), np.sin(theta)])

def linStokesVec (theta):
    """
    Creates a general linear polarized light state in the Stokes parameterization.

    Parameters
    ----------
    theta: float
        Angle of polarization wrt. horizontal.

    Returns
    -------
    np.array[4]
        4-vector containing a set of corresponding Stokes parameters for a linear
        polarized state at angle theta.
    """
    return np.array([1,np.cos(2*theta),np.sin(2*theta),0])

 # Circular light:

def LcircularJonesVec ():
    """
    Creates a left circular polarized light state in the Jones parameterization.

    Parameters
    ----------
    N/A

    Returns
    -------
    np.array[2]
        2-vector containing a set of corresponding Jones parameters for a
        left-circular polarized state.
    """
    return np.array([1/np.sqrt(2), 1j/np.sqrt(2)],dtype=complex)

def RcircularJonesVec ():
    """
    Creates a right circular polarized light state in the Jones parameterization.

    Parameters
    ----------
    N/A

    Returns
    -------
    np.array[2]
        2-vector containing a set of corresponding Jones parameters for a
        right-circular polarized state.
    """
    return np.array([1/np.sqrt(2), -1j/np.sqrt(2)],dtype=complex)

def LcircularStokesVec ():
    """
    Creates a left circular polarized light state in the Stokes parameterization.

    Parameters
    ----------
    N/A

    Returns
    -------
    np.array[4]
        4-vector containing a set of corresponding Stokes parameters for a
        left-circular polarized state.
    """
    return np.array([1,0,0,-1])

def RcircularStokesVec ():
    """
    Creates a right circular polarized light state in the Stokes parameterization.

    Parameters
    ----------
    N/A

    Returns
    -------
    np.array[4]
        4-vector containing a set of corresponding Stokes parameters for a
        right-circular polarized state.
    """
    return np.array([1,0,0,1])

 # Linear polarizers:

def linJonesMat (theta):
    """
    Creates a polarizer theta to the horizontal in the Jones paramterization.

    Parameters
    ----------
    theta: float
        Angle of polarization wrt. horizontal.

    Returns
    -------
    np.array[2x2]
        2x2 matrix containing the Jones representation of a linear polarization
        theta to the horizontal.
    """
    return np.array([[np.cos(theta) ** 2, np.cos(theta) * np.sin(theta)],
                     [np.cos(theta) * np.sin(theta), np.sin(theta) ** 2]])

def linStokesMat (theta):
    """
    Creates a polarizer theta to the horizontal in the Stokes paramterization.

    Parameters
    ----------
    theta: float
        Angle of polarization wrt. horizontal.

    Returns
    -------
    np.array[4x4]
        4x4 matrix containing the Stokes representation of a linear polarization
        theta to the horizontal.
    """

    return 1/2 * np.array([[1,np.cos(2*theta),np.sin(2*theta),0],
                           [np.cos(2*theta),np.cos(2*theta) **2, np.cos(2*theta) * np.sin(2*theta), 0],
                           [np.sin(2*theta),np.cos(2*theta) * np.sin(2*theta), np.sin(2*theta) **2, 0],
                           [0              , 0                               , 0                  , 0]])

 # Half-wave/Quarter-wave plates:

def JonesPhaseRetarder(theta, delta):
    """
    Creates the Jones representation of a phase retarder with 
    phase different delta and major axis theta to the horizontal.
    The fast axis is at angle theta to the established horizontal,
    and is faster than the slow axis by a phase of delta.

    Ex: a QWP and a HWP have deltas of pi/2, pi, respectively.

    Parameters
    ----------
    theta: float
        Angle of polarization wrt. horizontal.
    delta: float
        Phase difference created by polarizer.

    Returns
    -------
    np.array[2x2]
        2x2 matrix containing the Jones representation of a phase retarder with 
    phase different delta and major axis theta to the horizontal.
    """

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    exp_i_eta = np.exp(1j * delta)
    
    m11 = cos_t**2 + exp_i_eta * sin_t**2
    m12 = (1 - exp_i_eta) * cos_t * sin_t
    m21 = m12  
    m22 = sin_t**2 + exp_i_eta * cos_t**2
    
    matrix = np.array([[m11, m12],
                       [m21, m22]], dtype = complex)
    
    scalar = np.exp(-1j * delta / 2)
    
    return scalar * matrix

def MuellerPhaseRetarder(theta, delta):
    """
    Creates the Stokes representation of a phase retarder with 
    phase different delta and major axis theta to the horizontal.
    The fast axis is at angle theta to the established horizontal,
    and is faster than the slow axis by a phase of delta.

    Ex: a QWP with a 

    Parameters
    ----------
    theta: float
        Angle of polarization wrt. horizontal.
    delta: float
        Phase difference created by polarizer.

    Returns
    -------
    np.array[4x4]
        4x4 matrix containing the Stokes representation of a phase retarder with 
    phase different delta and major axis theta to the horizontal.
    """

    cos_2t = np.cos(2 * theta)
    sin_2t = np.sin(2 * theta)
    cos_d = np.cos(delta)
    sin_d = np.sin(delta)
    
    m11 = cos_2t**2 + (sin_2t**2 * cos_d)
    m12 = cos_2t * sin_2t * (1 - cos_d)
    m13 = -sin_2t * sin_d
    
    m22 = (cos_2t**2 * cos_d) + sin_2t**2
    m23 = cos_2t * sin_d
    
    matrix = np.array([
        [1,    0,    0,   0],
        [0,  m11,  m12, m13],
        [0,  m12,  m22, m23],
        [0, -m13, -m23,  cos_d]
    ])

    return matrix

 # Methods for converting between Stokes/Jones vectors
 # ---------------------------------------------------
'''
The following four functions are dedicated to converting Jones Vectors to Stokes Vectors
and vice versa. Since BIFROST does not simulate unpolarized light, both Stokes and Jones
vectors correspond to the same physical properties, defined here as delta and phi, which
correspond respectively to the phase difference between the defined axes and the arctangent
of the y-amplitude to the x-amplitude. If delta is positive, the y-component is behind the
x-component and vise versa.
'''

def JonesVdeltaPhi (JonesV):

    """
    Converts from a Jones vector parameterization of a polarization state to
    delta (phase difference between defined perpendicular axes) and phi
    (arctangent of the amplitude in the defined y-direction to the amplitude
    in the x-direction). 
    
    Delta arises from the phase difference indicated by the imaginary
    number e^i(delta), phi arises from a simple arccos calculation of the x-amplitude
    (Jones vectors are always normalized).

    Parameters
    ----------
    JonesV: np.array[2]
        2-vector containing a set of Jones parameters. 

    Returns
    -------
    np.array[2]
        2-vector containing delta (phase difference between defined perpendicular axes)
        and phi (arctangent of the amplitude in the defined y-direction to the amplitude
    in the x-direction) in that order.
    """

    normerror = np.absolute(np.sqrt(np.absolute(JonesV[1]) ** 2 + np.absolute(JonesV[0]) ** 2) - 1)

    if normerror > 1e-12:
        print ("Warning: Jones vector is not normalized. Output will normalize polarization state.")

    return np.array([np.angle(JonesV[1])-np.angle(JonesV[0]),
                        np.arctan2(np.absolute(JonesV[1]),np.absolute(JonesV[0]))])

def StokesVdeltaPhi (StokesV):

    """
    Converts from a Stokes vector parameterization of a polarization state to
    delta (phase difference between defined perpendicular axes) and phi
    (arctangent of the amplitude in the defined y-direction to the amplitude
    in the x-direction). 

    Parameters
    ----------
    StokesV: np.array[4]
        4-vector containing a set of Stokes parameters. 

    Returns
    -------
    np.array[2]
        2-vector containing delta (phase difference between defined perpendicular axes)
        and phi (arctangent of the amplitude in the defined y-direction to the amplitude
    in the x-direction) in that order.
    """

    S0error = np.absolute(StokesV[0]-1)

    normerror = np.absolute(np.sqrt(StokesV[1] ** 2 + StokesV[2] ** 2
                                    + StokesV[3] ** 2) - 1)

    if np.absolute(np.imag(StokesV[0])) > 1e-12 or np.absolute(np.imag(StokesV[1])) > 1e-12 or np.absolute(np.imag(StokesV[2])) > 1e-12 or np.absolute(np.imag(StokesV[3])) > 1e-12:
       raise "Invalid input: Stokes vector with imaginary components"


    if normerror > 1e-12 or S0error > 1e-12:
        print ("Warning: Stokes vector is not normalized. Output will normalize polarization state.")


    if -1e-12 < StokesV[1] - 1 < 1e-12:
        return [0,0]
    else:
        return np.array([np.arctan2(StokesV[3],StokesV[2]),1/2 * np.arccos(StokesV[1])])

def JonesVtoStokesV (JonesV):
    """
    Converts from a Jones vector parameterization of a polarization state to
    a normalized Stokes vector. Global phase is not conserved as a 
    result of the differences inherent between the formulations.
    We assume the input state is normalized; if not, a warning
    is printed.

    Parameters
    ----------
    JonesV: np.array[2]
        2-vector containing a set of Jones parameters. 

    Returns
    -------
    np.array[4]
        4-vector containing a set of corresponding Stokes parameters for a normalized state.
    """

    deltaphi = JonesVdeltaPhi(JonesV)

    return np.array([1,np.cos(2*deltaphi[1]),np.sin(2*deltaphi[1])*np.cos(deltaphi[0]),
                     -np.sin(2*deltaphi[1])*np.sin(deltaphi[0])])

def StokesVtoJonesV (StokesV):
    """
    Converts from a Jones vector parameterization of a polarization state to
    a normalized Stokes vector. Global phase is not conserved as a 
    result of the differences inherent between the formulations.
    We assume the input state is normalized; if not, a warning
    is printed.

    Parameters
    ----------
    StokesV: np.array[4]
        4-vector containing a set of Stokes parameters. 

    Returns
    -------
    np.array[2]
        2-vector containing a set of corresponding Jones parameters for a normalized state.
    """

    deltaphi = StokesVdeltaPhi(StokesV)

    return np.array([np.cos(deltaphi[1]), np.conjugate(np.exp(deltaphi[0]*1j)*np.sin(deltaphi[1]))], dtype=complex)

'''
The following two functions convert between Jones matrices to Mueller matrices.
For the Jones to Mueller conversion, a simple set of matrix operations accomplishes
the task, while the Mueller to Jones conversion is a bit more involved, si it uses the 
covariance matrix method developed by Cloude. It produces a covariance matrix from the
Mueller matrix provided (H) which is isomorphic to the Jones convention, and then
derives the Jones matrix from said matrix.
'''

def JonesMtoMuellerM (JonesM):

    """
    Converts from a Jones parameterization of a polarizer to
    a Stokes parameterization. Uses the well
    known formula of S = A Kronecker(J, Jconjugate) Ainverse, where Kronecker() indicates
    the Kroncker product.

    Parameters
    ----------
    JonesM: np.array[2,2]
        2x2 matrix containing the Jones parameterization of a polaizer. 

    Returns
    -------
    np.array[4,4]
        4x4 matrix containing the Mueller parameterization of a polarizer.
    """

    if np.allclose(JonesM, np.transpose(np.conj(JonesM))) != True:
        print("WARNING: Inputted Jones matrix not unitary, not power-conserving")

    U = np.array([[1,0,0,1],
                  [1,0,0,-1],
                  [0,1,1,0],
                  [0,-1j,1j,0]], dtype=complex)
    
    Uinv = np.array([[ 0.5+0.j,   0.5+0.j,   0. +0.j,   0. +0.j ],
                    [ 0. +0.j ,  0. +0.j   ,0.5+0.j   ,0. +0.5j],
                    [ 0. +0.j  , 0. +0.j ,  0.5+0.j  , 0. -0.5j],
                    [ 0.5+0.j,  -0.5-0.j ,  0. -0.j  , 0. -0.j ]], dtype=complex)

    Jxj = np.kron(JonesM,JonesM.conj())

    step1 = np.matmul(U,Jxj)
    final = np.matmul(step1,Uinv)

    if np.allclose(final,np.transpose(np.conj(final))):
        print("WARNING: Output matrix not unitary, is depolarizing")

    return np.real(final)
         

def MuellerMtoJonesM (M):

    """
    Converts from a Mueller parameterization of a polarizer to
    a Stokes parameterization. Uses a method first outlined by Cloude
    in Conditions For The Physical Realisability Of Matrix Operators In Polarimetry (1990);
    generates a coherence matrix based of of the Mueller matrix. If the input matrix is
    nondepolarizing, there should be only one nonzero eigenvalue; a corresponding Jones
    matrix can then be derived from the corresponding eigenvector. This process is 
    directly outlined in an unpublished 2019 Arxiv note by Kuntman and Kuntman, linked
    here: https://doi.org/10.48550/arXiv.1906.11198

    Parameters
    ----------
    M: np.array[4,4]
        4x4 matrix containing the Mueller parameterization of a polarizer.

    Returns
    -------
    JonesM: np.array[2,2]
        2x2 matrix containing the Jones parameterization of a polaizer. 
    """
    if np.allclose(M,np.transpose(np.conj(M))):
        print("WARNING: Input matrix not unitary, is depolarizing")


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
    ]], dtype=complex)

    if np.allclose(H, H.conj().T) == False:
        print( "WARNING: covariance not hermitian.")

    eigenvals, eigenvects = np.linalg.eigh(H)

    if -1e-12 <= eigenvals[3] <= 1e-12:
        raise "error: zero matrix"

    k = eigenvects[:, 3]
    k = k* np.sqrt(eigenvals[3])

    final = np.array([
        [k[0] + k[1], k[2] - 1j*k[3]],
        [k[2] + 1j*k[3], k[0] - k[1]]
    ], dtype=complex)

    if np.allclose(final,np.transpose(np.conj(final))):
        print("WARNING: Input matrix not unitary, is depolarizing")

    return final

def MuellertoAxisAngle(M):

    '''
    Takes a Mueller matrix corresponding to a rotation on the poincare sphere (eg a phase retarder)
    and returns the axis and angle of said matrix in that order.

    The axis of a rotation is the vector for which the rotation has no effect, eg Rv = v.
    Thus, (R-I)v = 0, where I is the identity matrix, and hence v is the eigenvector of R
    corresponding to an eigenvalue of 1. This leads eventually to the Rodrigues formula
    and its equivalents, which are used here. A particular form involving sin(theta)
    is used in order to keep the signs consistent. The angle of rotation is given by
    abs(theta) = arccos((tr(R)-1)/2), which would otherwise create ambiguity in sign
    without the sin(theta) component in the vector.

    Parameters
    ----------
    M: np.array[4,4]
        4x4 matrix containing the Mueller parameterization of a polarizer.

    Returns
    -------
    Axis: np.array[4]
        4-vector containing the axis of rotation on the Poincare sphere.
    
    Angle: float
        Angle of rotation on the Poincare sphere.
    '''

    

    angle = np.arccos((np.trace(M) - 1 - 1)/2) #an additional -1 is added to account for the 
    #1 on the upper-left of the matrix.

    fact = 1/(2 * np.sin(angle))
    
    axis = -fact * np.array([-1/fact,
                                             M[3][2] - M[2][3],
                                             M[1][3] - M[3][1],
                                             M[2][1] - M[1][2]])

    return axis, angle

def AxisAngletoMueller(Ax, Ang):
    '''
    Here, we use a well-known formula to create the Mueller matrix of a phase retarder corresponding
    to a rotation of angle Ang around a defined axis on the poincare sphere.

    Parameters
    ----------
    Ax: np.array[4]
        4-vector containing the axis of rotation on the Poincare sphere.
    
    Ang: float
        Angle of rotation on the Poincare sphere.

    Returns
    -------

    '''

    cs = np.cos(Ang)
    sn = np.sin(Ang)

    col0 = np.array([1,
                     0,
                     0,
                     0])

    col1 = np.array([0,
        Ax[1] **2 * (1-cs) + cs,
        Ax[1]*Ax[2]*(1-cs) + Ax[3]*sn,
        Ax[1]*Ax[3]*(1-cs) - Ax[2]*sn
    ])

    col2 = np.array([0,
        Ax[1]*Ax[2]*(1-cs) - Ax[3] * sn,
        Ax[2]**2 * (1-cs) + cs,
        Ax[2]*Ax[3]*(1-cs) + Ax[1]*sn
    ])

    col3 = np.array([0,
        Ax[1]*Ax[3]*(1-cs) + Ax[2] * sn,
        Ax[2]*Ax[3] * (1-cs) - Ax[1] * sn,
        Ax[3]**2 * (1-cs) + cs
    ])

    return np.array([col0,col1,col2,col3])