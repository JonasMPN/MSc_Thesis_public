#! /usr/bin/env python
""" This Python module contains the aeroelastic model presented in the paper
'Fundamental aeroelastic properties of a bend-twist coupled blade section'
by Alexander R. Staeblein, Morten H. Hansen, and Georg Pirrung.
The paper has been submitted to the Journal of Fluid and Structures
and it is currently under review."""

from __future__ import division
import numpy as np

def def_para(c       = 3.292, # structural parameters
            e_ac     = 0.113,
            e_cg     = 0.304,
            r        = 0.785, 
            m        = 203.,
            omg_x    = 0.93*2*np.pi, # frequencies
            omg_y    = 0.61*2*np.pi,
            # omg_phi  = 6.66*2*np.pi,
            omg_phi  = 5.69*2*np.pi,
            c_x      = 0.0049, # damping
            c_y      = 0.0047,
            c_phi    = 0.0093,
            U0       = 45.0, # inflow velocities
            V0       =  5.6,
            gam_x    = 0., # coupling parameters
            gam_y    = 0.,            
            aero_mod = 'us',# aerodynamic model quasi-stady (qs) or unsteady (us)
            flut_val = False):# Flag for flutter validation

    para = {'c'        : c,
            'e_ac'     : e_ac,
            'e_cg'     : e_cg,
            'r'        : r,      
            'm'        : m,      
            'omg_x'    : omg_x,  
            'omg_y'    : omg_y,  
            'omg_phi'  : omg_phi,
            'c_x'      : c_x,    
            'c_y'      : c_y,    
            'c_phi'    : c_phi,  
            'U0'       : U0,      
            'V0'       : V0,
            'gam_x'    : gam_x,  
            'gam_y'    : gam_y,  
            'aero_mod' : aero_mod,
            'flut_val' : flut_val}
            
    return para

def structure(para):
    '''Returns the linear structural model as a function of structural, and aerodynamic states,
    and wind speed variation.'''
    e_cg     = para['e_cg']
    m        = para['m']
    r        = para['r'] 
    wx       = para['omg_x'] 
    wy       = para['omg_y'] 
    wz       = para['omg_phi']  
    cx       = para['c_x'] 
    cy       = para['c_y'] 
    cz       = para['c_phi'] 
    gam_x    = para['gam_x'] 
    gam_y    = para['gam_y']
    flut_val = para['flut_val']

    # Define mass and stiffness matrix
    M = np.array([[m, 0, 0],[0, m, -m*e_cg],[0,-m*e_cg, m*(e_cg**2+r**2)]])
    K = np.array([[m*wx**2, 0, 0],[0, m*wy**2, 0],[0, 0, m*r**2*wz**2]])
    K[0,2] = K[2,0] = -gam_x*np.sqrt(K[0,0]*K[2,2]) # edge-twist coupling term
    K[1,2] = K[2,1] = -gam_y*np.sqrt(K[1,1]*K[2,2]) # flap-twist coupling term

    # Calculate, sort and mass-normalize eigenvectors
    A = np.dot(np.linalg.inv(M),K)
    lam, phi = np.linalg.eig(A)
    lam = np.sqrt(lam)
    i0 = abs(phi[0,:]).argmax()
    i1 = abs(phi[1,:]).argmax()
    i2 = abs(phi[2,:]).argmax()
    phi = phi[:,(i0,i1,i2)]
    for i in range(3):
        n = np.dot(np.dot(phi[:,i].T,M),phi[:,i])
        phi[:,i] = phi[:,i]/np.sqrt(n)

    # Obtain modal damping matrix
    C = 2*np.diag([lam[i0]*cx, lam[i1]*cy, lam[i2]*cz])
    C = np.dot(np.dot(np.linalg.inv(phi.T),C),np.linalg.inv(phi))
    print(M)
    print(C)
    print(K)
    # Structure depending on ddx, ddy, ddt, dx, dy, dt, x, y, t, x1, x2, V1
    S = np.hstack((M, C, K, np.zeros((3,3))))
    if flut_val:
        S = np.hstack((M, np.dot(M,2*np.diag([wx*cx, wy*cy, wz*cz])),
                np.dot(M, np.diag([wx**2, wy**2, wz**2])),np.zeros((3,3))))
        S[1,5] = S[1,8] = S[2,4] = S[2,7] = 0

    return S

def aero_force(para):
    '''Returns the aerodynamic forces, unsteady flow model and inflow angles as a function
    of structural, and aerodynamic states, and wind speed variation.'''
    c        = para['c']
    e_ac     = para['e_ac']
    U0       = para['U0']
    V0       = para['V0']
    aero_mod = para['aero_mod']
    flut_val = para['flut_val']
    
    # No aerodynamic model for 0 inflow velocity
    if abs(U0)<1e-3:
        Q  = np.zeros((3,12))
        aero = np.array([[0,0,0, 0,0,0, 0,0,0, 1.,0, 0],
                         [0,0,0, 0,0,0, 0,0,0, 0,1., 0]])
        alphas = np.zeros((3,12))
        return Q, aero, alphas, para

    # Aerodynamic parameters
    rho = 1.225 # air density
    A1 = 0.165 # factors for the unsteady model (assuming flat plate)
    A2 = 0.335
    b1 = 0.0455
    b2 = 0.3000

    # Aerodynamic coefficients
    if flut_val:
        dCL = 2*np.pi 
        CD0 = lambda alpha: 0.
        CM0 = lambda alpha: 0.
    else:            
        dCL = 7.15 
        CD0 = lambda alpha:  .01
        CM0 = lambda alpha: -.1
    CL0 = lambda alpha: dCL*alpha

    # Inflow velocities
    W0  = np.sqrt(U0**2 + V0**2) # inflow velocity
    para['W0'] = W0 # save inflow velocity as model parameter
    W1  = 1./W0*np.array([0,0,0, U0,-V0,0, 0,0,0, 0,0, V0])
    dW1 = 1./W0*np.array([U0,-V0,0, 0,0,0, 0,0,0, 0,0,  0])

    # Angles
    alpha0   = np.arctan(V0/U0)
    alpha1   = np.array([0,0,0, -V0/W0**2,-U0/W0**2,0, 0,0,1., 0,0, U0/W0**2])
    alpha1QS = alpha1 + np.array([0,0,0, 0,0,(c/2.-e_ac)*U0/W0**2, 0,0,0, 0,0, 0])
    alpha1E  = alpha1QS*(1.-A1-A2) + np.array([0,0,0, 0,0,0, 0,0,0, 1.,1., 0])
    alphas = np.vstack((alpha1, alpha1QS, alpha1E))

    # Aerodynamic models
    if aero_mod=='us': # unsteady model depending on ddx, ddy, ddt, dx, dy, dt, x, y, t, x1, x2, V1
        aero = np.vstack((2.*W0*b1/c*A1*alpha1QS - A1*alpha0/W0*dW1 -2.*W0*b1/c*np.array([0,0,0, 0,0,0, 0,0,0, 1.,0, 0]),
                        2.*W0*b2/c*A2*alpha1QS - A2*alpha0/W0*dW1 -2.*W0*b2/c*np.array([0,0,0, 0,0,0, 0,0,0, 0,1., 0])))
    elif aero_mod=='qs': # quasy steady model
        alpha1E = alpha1QS
        aero = np.array([[0,0,0, 0,0,0, 0,0,0, 1.,0, 0],
                         [0,0,0, 0,0,0, 0,0,0, 0,1., 0]])

    # Aerodynamic forces
    L0 = 0.5*rho*W0**2*CL0(alpha0) # steady state lift
    L1 = rho*W0*W1*c*CL0(alpha0)+0.5*rho*W0**2*c*dCL*alpha1E \
       + rho*np.pi*c**2/4.*(np.array([0,-1.,(.25*c-e_ac), 0,0,W0, 0,0,0, 0,0, 0])) # lift variation
    D1 = rho*W0*W1*c*CD0(alpha0)+L0*(-alpha1E) # drag variation
    M1 = rho*W0*W1*c*CM0(alpha0) \
       - rho*np.pi*c**3/8.*(np.array([0,-.5,.5*(3.*c/8.-e_ac), 0,0,W0, 0,0,0, 0,0, 0])) # moment variation
    Q = np.vstack((-D1, L1, M1+e_ac*L1)) # aerodynamic force vector
    T = np.array([[ np.cos(alpha0), np.sin(alpha0), 0.],
                  [-np.sin(alpha0), np.cos(alpha0), 0.],
                  [             0.,              0., 1.]]) # rotation matrix
    Q = np.dot(T,Q) # Rotate forces into chord coordinates

    return Q, aero, alphas, para

def ssm(para):
    '''Returns the aeroelastic state space model.'''
    S = structure(para)
    Q, aero, alphas, para = aero_force(para)

    # System + Input Matrix
    A = np.vstack((Q-S, np.zeros((3,12)), aero)) # RHS of EOM
    A[3:6,3:6] = np.identity(3) 
    A[:3,:] = np.dot(np.linalg.inv(S[:,:3]-Q[:,:3]),A[:3,:]) # invert mass matrix to identity
    for i in range(3): # eliminate mass matrix of aerodynamic model
        A[6,:] -= A[6,i]*A[i,:]
        A[7,:] -= A[7,i]*A[i,:]
    B = A[:,11:] # input matrix
    A = A[:, 3:11] # system matrix

    # Output Matrix (forces + angles)
    C = np.vstack((Q[:,3:11]+np.dot(Q[:,:3],A[:3,:]), # aerodynamic forces
                   np.hstack((-S[:,3:6],np.zeros((3,5)))), # structural damping forces                   
              alphas[:,3:11]+np.dot(alphas[:,:3],A[:3,:]))) # angles

    # Feedtrough Matrix
    D = np.zeros((9,1))

    return A, B, C, D, para

def eom(para):
    '''Returns the aeroelastic matrices of the equation of motion.'''
    S = structure(para)
    Q, aero, alphas, para = aero_force(para)

    M = S[:,0:3]-Q[:,0:3]
    C = S[:,3:6]-Q[:,3:6]
    K = S[:,6:9]-Q[:,6:9]

    return M, C, K

if __name__ == "__main__":
    '''Initiate a model and run an eigenvalue analysis for demonstration purposes.'''
    para = def_para(gam_y=-0.4)
    A, B, C, D, para = ssm(para)
    w , v = np.linalg.eig(A)
    # print('Modal frequencies [Hz] and damping ratios [-] for the')
    # print('    structural modes of the default blade section model:\n')
    # print('Mode  | Freq. | Damp. ')
    # print('----------------------')
    # print('edge  |  %.2f | %.4f' %(w[2].imag/(2*np.pi), -w[2].real/abs(w[2])))
    # print('flap  |  %.2f | %.4f' %(w[4].imag/(2*np.pi), -w[4].real/abs(w[4])))
    # print('twist |  %.2f | %.4f' %(w[0].imag/(2*np.pi), -w[0].real/abs(w[0])))