# -*- coding: utf-8 -*-
"""

@author: hara
"""
import Pyrochlore_Lib as mod
import matplotlib.pyplot as plt


from scipy.optimize import fsolve
import numpy as np
import pandas as pd

J_int=-1
D_int=0.3
K_int=-0.3

#zero field configuration
#-0.3	1.547705951521779	1.593886702068014	1.593886702068014	1.547705951521779	0.02309653323068469	0.02309653323068469	6.260088773948902	6.260088773948902

J_Matrix=mod.Interaction_Matrix(J_int, D_int, K_int)
for B in [0.001,0.005,0.01,0.05,0.1,0.2]:

    #B_ext=B*np.array([1,1,0])/np.sqrt(2)
    #B=0.001
    B_ext=B*np.array([1,1,0])/np.linalg.norm([1,1,0])    
    B_dir=B_ext/np.linalg.norm(B_ext)
    


    Energies = []
    Angles = []
    N = 10
    
    
    ran = np.random.rand(8*N)
    for i in range(N):
        guess = [np.pi*ran[8*i], np.pi*ran[8*i+1], np.pi*ran[8*i+2], np.pi*ran[8*i+3], 2*np.pi*ran[8*i+4], 2*np.pi*ran[8*i+5], 2*np.pi*ran[8*i+6], 2*np.pi*ran[8*i+7]]
        res =mod.Energy_minimization(guess, B_ext, J_Matrix)
        Energies += [res.fun]
        Angles += [res.x]
        
    Energies = np.array(Energies)
    y=Angles[Energies.argmin()]
    
    print("\n")
    print(B_ext)
    root1=fsolve(mod.Linear_Terms,y,args=(B_ext,J_Matrix))
    print(root1)
    print(mod.Classical_Energy_at(root1, B_ext, J_Matrix))
    print(mod.Linear_Terms(root1, B_ext, J_Matrix))
    mod.Coordinate_Transform(root1)
    
    print("\n")
    
    
    guess1=[1.547705951521779,1.593886702068014,1.593886702068014,1.547705951521779,0.02309653323068469,0.02309653323068469,6.260088773948902,6.260088773948902]
    root=fsolve(mod.Linear_Terms,guess1,args=(B_ext,J_Matrix))
    print(root)
    print(mod.Classical_Energy_at(root, B_ext, J_Matrix))
    print(mod.Linear_Terms(root, B_ext, J_Matrix))
    mod.Coordinate_Transform(root)
    
    print("\n")    

    print(mod.splay_check(root1))
    print(mod.splay_check(root))
    #mod.Draw_Band(['G','X','W','G','L','W','U','X','K','G'], B_ext, J_Matrix, J_int, D_int, K_int, root, 'FM_splay_Figures')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    VecStart_x = [0,0,0,2,2,2]
    VecStart_y = [0,0,0,2,2,0]
    VecStart_z = [0,0,0,0,0,2]
    VecEnd_x = [2,0,2,0,2,0]
    VecEnd_y = [2,2,0,2,0,2]
    VecEnd_z  =[0,2,2,2,2,2]
    ax.quiver(-np.cos(root[4])*np.sin(root[0])/2, -np.sin(root[4])*np.sin(root[0])/2, -np.cos(root[0])/2, np.cos(root[4])*np.sin(root[0]), np.sin(root[4])*np.sin(root[0]), np.cos(root[0]), color='c')
    ax.quiver(2-np.cos(root[5])*np.sin(root[1])/2, 2-np.sin(root[5])*np.sin(root[1])/2, -np.cos(root[1])/2, np.cos(root[5])*np.sin(root[1]), np.sin(root[5])*np.sin(root[1]), np.cos(root[1]), color='m')
    ax.quiver(-np.cos(root[6])*np.sin(root[2])/2, 2-np.sin(root[6])*np.sin(root[2])/2, 2-np.cos(root[2])/2, np.cos(root[6])*np.sin(root[2]), np.sin(root[6])*np.sin(root[2]), np.cos(root[2]), color='y')
    ax.quiver(2-np.cos(root[7])*np.sin(root[3])/2, -np.sin(root[7])*np.sin(root[3])/2, 2-np.cos(root[3])/2, np.cos(root[7])*np.sin(root[3]), np.sin(root[7])*np.sin(root[3]), np.cos(root[3]), color='g')
    #ax.quiver(2,2,2, B_dir[0], B_dir[1], B_dir[2], color='black')
    for i in range(6):
        ax.plot([VecStart_x[i], VecEnd_x[i]], [VecStart_y[i],VecEnd_y[i]],zs=[VecStart_z[i],VecEnd_z[i]], color='grey')
    ax.set_xlim([-1, 3])
    ax.set_ylim([-1, 3])
    ax.set_zlim([-1, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.rcParams['figure.dpi'] = 800
    #plt.title(f'J={J_int}, D={D_int}, K={K_int}, B={B_mag}[{B_dir[0]}, {B_dir[1]}, {B_dir[2]}]')
    plt.show()