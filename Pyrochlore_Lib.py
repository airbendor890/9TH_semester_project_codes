import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from  scipy.optimize import  fsolve,minimize

def Interaction_Matrix(J_int, D_int, K_int):
    fo = open('input_matrix_JDK_hamiltonian_pyrochlore.dat', "r")
    ls_eq = fo.read().split('\n')
    fo.close()
    Matrix = np.zeros(3888).reshape(4,3,3,3,4,3,3)
    No_data = len(ls_eq)
    i = 0
    for row in ls_eq:
        i += 1
        data = [float(j) for j in row.split()]
        if(i in range(73)):
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = J_int*data[-1]/2
        elif(i in range(73,217)):
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = D_int*data[-1]/2
        else:
            Matrix[int(data[0]-1), int(data[1]+1), int(data[2]+1), int(data[3]+1), int(data[4]-1), int(data[5]-1), int(data[6]-1)] = K_int*data[-1]
    del row, ls_eq, data
    return Matrix

def e_trans(m,Lambda,theta,phi):
    #m::local coordinate index
    #lambda::fixed coordinate index
    if(m==1):
        if(Lambda==1):
            return np.cos(theta)*np.cos(phi)
        elif(Lambda==2):
            return np.cos(theta)*np.sin(phi)
        elif(Lambda==3):
            return -np.sin(theta)
    elif(m==2):
        if(Lambda==1):
            return -np.sin(phi)
        elif(Lambda==2):
            return np.cos(phi)
        elif(Lambda==3):
            return 0
    elif(m==3):
        if(Lambda==1):
            return np.sin(theta)*np.cos(phi)
        elif(Lambda==2):
            return np.sin(theta)*np.sin(phi)
        elif(Lambda==3):
            return np.cos(theta)

def J_curly(J_Matrix,alpha, n_1, n_2, n_3, beta, Lambda, Mu):
    return J_Matrix[(alpha)-1,(n_1)+1,(n_2)+1,(n_3)+1,(beta)-1,(Lambda)-1,(Mu)-1]


def D(J_Matrix, alpha, n_1, n_2, n_3, beta, m, n, theta_alpha, phi_alpha, theta_beta, phi_beta):
    #m,n::local coordinate
    #alpha,beta:sublattice sites
    #i,j :lattice sites
    sum = 0
    for Lambda in [1,2,3]:
        for Mu in [1,2,3]:
            sum += J_curly(J_Matrix,alpha, n_1, n_2, n_3, beta, Lambda, Mu)*e_trans(m, Lambda, theta_alpha, phi_alpha)*e_trans(n, Mu, theta_beta, phi_beta)
    return sum


def B(m, theta, phi, B_ext):
    sum = 0
    for Lambda in range(1,4):
        sum += B_ext[Lambda-1]*e_trans(m, Lambda, theta, phi)
    return sum


def Linear_Terms(x, B_ext, J_Matrix):
    S = 1
    F = np.zeros(8)
    for alpha in [1,2,3,4]:
        sum1, sum2 = 0, 0
        for n_1 in [-1,0,1]:
            for n_2 in [-1,0,1]:
                for n_3 in [-1,0,1]:
                    for beta in [1,2,3,4]:
                        sum1 += (S)*(D(J_Matrix, alpha, -n_1, -n_2, -n_3, beta, 1, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
                        sum2 += (S)*(D(J_Matrix, alpha, -n_1, -n_2, -n_3, beta, 2, 3, x[alpha-1], x[alpha+3], x[beta-1], x[beta+3]))
        sum1 -= B(1, x[alpha-1], x[alpha+3], B_ext)/2
        sum2 -= B(2, x[alpha-1], x[alpha+3], B_ext)/2
        F[alpha-1] = sum1
        F[alpha+3] = sum2
    return F


def S_Lambda(Lambda, theta_alpha, phi_alpha):
    S = 1
    if(Lambda==3):
        return S*np.cos(theta_alpha)
    elif(Lambda==1):
        return S*np.sin(theta_alpha)*np.cos(phi_alpha)
    elif(Lambda==2):
        return S*np.sin(theta_alpha)*np.sin(phi_alpha)
    else:
        print('Error')


def Classical_Energy_at(Angles, B_ext, J_Matrix):
    sum = 0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                for alpha in range(1,5):
                    for beta in range(1,5):
                        for Lambda in range(1,4):
                            for Mu in range(1,4):
                                sum += J_curly(J_Matrix,alpha, n1, n2, n3, beta, Lambda, Mu)*S_Lambda(Lambda, Angles[alpha-1], Angles[alpha+3])*S_Lambda(Mu, Angles[beta-1], Angles[beta+3]) 
    sum2 =0
    for alpha in range(1,5):
        for Lambda in range(1,4):
            sum2 -= B_ext[Lambda-1]*S_Lambda(Lambda, Angles[alpha-1], Angles[alpha+3])
    return sum/4 + sum2/4

def Energy_minimization(guess, B_ext, J_Matrix):
    res = minimize(Classical_Energy_at, guess, args=(B_ext, J_Matrix), bounds=((0,np.pi), (0,np.pi), (0,np.pi), (0,np.pi), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi), (0,2*np.pi)))
    return res


############### For Spectrum ################

def J_pp(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0 + 1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0+1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) -D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

def J_nn(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0 + 1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0+1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) -D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))    

def J_pn(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0 + 1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0+1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) +D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

def J_np(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/4)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0 + 1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0+1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) +D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 2, theta_alpha, phi_alpha, theta_beta, phi_beta))

def J_p3(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0 + 1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) )

def J_n3(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 1, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0 + 1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 2, 3, theta_alpha, phi_alpha, theta_beta, phi_beta) )

def J_3p(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) - (0 + 1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) )

def J_3n(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return (1/2)*(D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 1, theta_alpha, phi_alpha, theta_beta, phi_beta) + (0 + 1j)*D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 2, theta_alpha, phi_alpha, theta_beta, phi_beta) )

def J_33(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    return D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 3, theta_alpha, phi_alpha, theta_beta, phi_beta)

## 1j = +    and     -1j = -
def J_mn(J_Matrix, mu, nu, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    if(mu==1j):
        if(nu==1j):
            return J_pp(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_pn(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==3):
            return J_p3(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    elif(mu==-1j):
        if(nu==1j):
            return J_np(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_nn(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==3):
            return J_n3(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    elif(mu==3):
        if(nu==3):
            return D(J_Matrix, alpha, n_1, n_2, n_3, beta, 3, 3, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==1j):
            return J_3p(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        elif(nu==-1j):
            return J_3n(J_Matrix, alpha, n_1, n_2, n_3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)
        else:
            print('Error')
    else:
            print('Error')


def J_q(J_Matrix, q, mu, nu, alpha, beta, theta_alpha, phi_alpha, theta_beta, phi_beta):
    sum = 0
    for n1 in [-1,0,1]:
        for n2 in [-1,0,1]:
            for n3 in [-1,0,1]:
                kernal = np.exp((0-1j)*(n1*np.dot(q,np.array([0.5,0.5,0])) + n2*np.dot(q,np.array([0,0.5,0.5])) + n3*np.dot(q,np.array([0.5,0,0.5]))))
                sum += J_mn(J_Matrix, mu, nu, alpha, n1, n2, n3, beta, theta_alpha, phi_alpha, theta_beta, phi_beta)*kernal
    return sum


def Elements_A_q(J_Matrix, q, alpha, beta, Theta_s, Phi_s):
    sum = 0
    if(alpha==beta):
        for gamma in range(1,5):
            sum += J_q(J_Matrix, np.array([0,0,0]), 3, 3, alpha, gamma, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[gamma-1], Phi_s[gamma-1]) + J_q(J_Matrix, np.array([0,0,0]), 3, 3, gamma, alpha, Theta_s[gamma-1], Phi_s[gamma-1], Theta_s[alpha-1], Phi_s[alpha-1])
    sum = -sum + 2*(J_q(J_Matrix, -q, -1j, 1j, alpha, beta, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[beta-1], Phi_s[beta-1]) + J_q(J_Matrix, q, 1j, -1j, beta, alpha, Theta_s[beta-1], Phi_s[beta-1], Theta_s[alpha-1], Phi_s[alpha-1]))
    return sum


def Elements_B_q(J_Matrix, q, alpha, beta, Theta_s, Phi_s):
    return 2*(J_q(J_Matrix, -q, -1j, -1j, alpha, beta, Theta_s[alpha-1], Phi_s[alpha-1], Theta_s[beta-1], Phi_s[beta-1]) + J_q(J_Matrix, q, -1j, -1j, beta, alpha, Theta_s[beta-1], Phi_s[beta-1], Theta_s[alpha-1], Phi_s[alpha-1]))


def Eigen_Value_finder(q, B_ext, J_Matrix, Theta_s, Phi_s, K_int):
    A_q = np.zeros((4,4), dtype=np.csingle)
    B_q = np.zeros((4,4), dtype=np.csingle)
    A_nq= np.zeros((4,4), dtype=np.csingle)

    for alpha in range(4):
        for beta in range(4):
            A_q[alpha, beta] = Elements_A_q(J_Matrix, q, alpha+1, beta+1, Theta_s, Phi_s)
            B_q[alpha, beta] = Elements_B_q(J_Matrix, q, alpha+1, beta+1, Theta_s, Phi_s)
            A_nq[alpha, beta] = Elements_A_q(J_Matrix, -q, alpha+1, beta+1, Theta_s, Phi_s)

    B_q_H = np.conjugate(B_q).T
    A_nq_T = A_nq.T

    if(np.allclose(np.conjugate(A_q).T, A_q) == False):
        print('A_q is NOT Hermitian, Error')

    D_q = np.zeros((8,8), dtype=np.csingle)
    for i in range(4):
        for j in range(4):
            if(i==j):
                D_q[i, j] = A_q[i,j] + B(3, Theta_s[i], Phi_s[i], B_ext)
                D_q[i+4, j+4] = -A_nq_T[i, j] - B(3, Theta_s[i], Phi_s[i], B_ext)
            else:
                D_q[i, j] = A_q[i,j]
                D_q[i+4, j+4] = -A_nq_T[i, j] 
            D_q[i, 4+j] = B_q[i,j]
            D_q[i+4, j] = -B_q_H[i, j]
    Eigen_Values = np.linalg.eigvals(D_q)
    if(Eigen_Values.imag.max()>1e-4): 
        print(f'Complex Eigen Value Error for K = {K_int} at q = ', print(q))
        fi = open("Error.txt", "a")
        fi.write(f"Complex Eigen Value Error for K = {K_int} at q = {print(q)}\n")
        #print('Complex Eigen Value Error for K = ',K_int,' at q = ', q)
        #fi = open("Error.txt", "a")
        #fi.write("Complex Eigen Value Error for K =" ,K_int, 'at q =', q,"\n")

    return np.real_if_close(Eigen_Values, 1e-4)


def Draw_Band(Path, B_ext, J_Matrix, J_int, D_int, K_int, Angles, File_Path):
    Theta_s = [Angles[0], Angles[1], Angles[2], Angles[3]]
    Phi_s =  [Angles[4], Angles[5], Angles[6], Angles[7]]
    '''X splay phase'''
    #'''
    G = np.array([0,0,0])
    X = np.array([2*np.pi,0,0])
    W = np.array([2*np.pi,np.pi,0])
    K = np.array([3*np.pi/2,3*np.pi/2,0])
    L = np.array([np.pi, np.pi, np.pi])
    U = np.array([2*np.pi,np.pi/2, np.pi/2])
    #'''

    '''Y Splay phase'''
    '''
    G = np.array([0,0,0])
    X = np.array([0,2*np.pi,0])
    W = np.array([np.pi,2*np.pi,0])
    K = np.array([3*np.pi/2,3*np.pi/2,0])
    L = np.array([np.pi, np.pi, np.pi])
    U = np.array([np.pi/2,2*np.pi, np.pi/2])
    '''
    
    '''Z Splay phase'''
    '''
    G = np.array([0,0,0])
    X = np.array([0,0,2*np.pi])
    W = np.array([0,np.pi,2*np.pi])
    K = np.array([3*np.pi/2,3*np.pi/2,0])
    L = np.array([np.pi, np.pi, np.pi])
    U = np.array([np.pi/2, np.pi/2,2*np.pi])
    '''


    ''''''
    # locals()[str]
    No_points = len(Path) 
    x = []
    y = [[] for i in range(8)]
    x_lebal = []
    q = np.array([0,0,0])
    #Points_in_1_path = 20
    x_l = 0
    for p in range(No_points-1):
        Points_in_1_path = int(np.linalg.norm((locals()[Path[p+1]]-locals()[Path[p]])))*4
        for i in range(Points_in_1_path):
            q = locals()[Path[p]] + (locals()[Path[p+1]]-locals()[Path[p]])*i/Points_in_1_path
            #print(q)
            EV = np.sort(abs(Eigen_Value_finder(q, B_ext, J_Matrix, Theta_s, Phi_s, K_int)))
            x += [i+x_l]
            if(i==0): x_lebal += [Path[p]]
            else: x_lebal += [' ']
            for j in range(8):
                y[j] += [EV[j]]
        x_l = x[-1]+1
    EV = np.sort(abs(Eigen_Value_finder(locals()[Path[No_points-1]], B_ext, J_Matrix, Theta_s, Phi_s, K_int)))
    x += [x_l]
    x_lebal += [Path[No_points-1]]
    for j in range(8):
        y[j] += [EV[j]]
        
    plt.figure()
    plt.xticks(x, x_lebal)
    plt.plot(x,y[0],color='k')
    plt.plot(x,y[1],color='k')
    plt.plot(x,y[2],color='r')
    plt.plot(x,y[3],color='r')
    plt.plot(x,y[4],color='g')
    plt.plot(x,y[5],color='g')
    plt.plot(x,y[6],color='b')
    plt.plot(x,y[7],color='b')
    plt.title(f'J = {J_int}, D = {D_int}, K = {K_int} , B =[{np.round(B_ext[0],4)}, {np.round(B_ext[1],4)}, {np.round(B_ext[2],4)} ]')
    #plt.savefig(f'{File_Path}\\{J_int}_{D_int}_{K_int}_{np.round(np.linalg.norm(B_ext),3)}_{111}.png',dpi=300)
    plt.show()
    plt.clf()

def Ground_State_Finder(B_ext, J_Matrix):
    Energies = []
    Angles = []
    N = 10
    ran = np.random.rand(8*N)
    guess = np.array(8)
    for i in range(N):
        guess = [np.pi*ran[8*i], np.pi*ran[8*i+1], np.pi*ran[8*i+2], np.pi*ran[8*i+3], 2*np.pi*ran[8*i+4], 2*np.pi*ran[8*i+5], 2*np.pi*ran[8*i+6], 2*np.pi*ran[8*i+7]]
        res = Energy_minimization(guess, B_ext, J_Matrix)
        Energies += [res.fun]
        Angles += [res.x]
    Energies = np.array(Energies)
    root = fsolve(Linear_Terms, Angles[Energies.argmin()], args=(B_ext, J_Matrix))
    return root

def Coordinate_Transform(Angles):
    for i in range(4):
        print(f'[x,y,z]_{i+1} = [{np.sin(Angles[i])*np.cos(Angles[i+4])},{np.sin(Angles[i])*np.sin(Angles[i+4])},{np.cos(Angles[i])}]')
    print('\n')

def splay_check(Angles):
    m1=[np.sin(Angles[0])*np.cos(Angles[4]),np.sin(Angles[0])*np.sin(Angles[4]),np.cos(Angles[0])]
    m2=[np.sin(Angles[1])*np.cos(Angles[5]),np.sin(Angles[1])*np.sin(Angles[5]),np.cos(Angles[1])]
    m3=[np.sin(Angles[2])*np.cos(Angles[6]),np.sin(Angles[2])*np.sin(Angles[6]),np.cos(Angles[2])]
    m4=[np.sin(Angles[3])*np.cos(Angles[7]),np.sin(Angles[3])*np.sin(Angles[7]),np.cos(Angles[3])]
    m=[]
    for i in range(3):
        temp=m1[i]+m2[i]+m3[i]+m4[i]
        m.append(temp)
    m=m/np.linalg.norm(m)
    u=np.array([0,-m[2],m[1]])
    phi=m[0]
    W=np.array([ [ 0,-u[2],u[1] ] , [ u[2],0,-u[0] ] ,[ -u[1],u[0],0 ]])
    R=np.add(np.identity(3),np.sin(phi)*W,2*np.sin(phi/2)*np.sin(phi/2)*np.dot(W,W))
    m1_=np.dot(R,m1)
    m2_=np.dot(R,m2)
    m3_=np.dot(R,m3)
    m4_=np.dot(R,m4)
    print(m1_)
    print(m2_)
    print(m3_)
    print(m4_)
    m_=[]
    for i in range(3):
        temp=m1_[i]+m2_[i]+m3_[i]+m4_[i]
        m_.append(temp)
    m_=m_/np.linalg.norm(m_)    
    
    return m_