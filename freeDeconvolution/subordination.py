
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 18:24:05 2018

@author: Ours blanc
"""
import numpy as np
# import scipy.signal
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import matrix
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pickle
from cmath import sqrt
import cvxopt
from cvxopt import matrix


# Parameters :
#mu : input sample (eigenvalue distribution of mu_3 from our paper)
# Gnoise : Cauchy transform of the noise mu_1
#sigma_noise : value of the imaginary part on which the deconvolution is done. Same sigma as in the paper (the smaller the sigma the better, one can usually do much better that the theoretic threshold) 
#returnmin, returnmax: intervalle on which the first step is achieved
#precisionmesh : mesh size of the latter intervalle
# precisioniteration : precision of the fixed point procedure. Can choose something very small since the precision is exponential (like 1e-12).
  
def additiveDeconvol(mu,Gnoise,sigma_noise,returnmin,returnmax,precisionmesh,precisioniteration):
    # Definingparameters

     #Definition of the H-transform

    def Gemp(z,mu):
        return sum(1./(z-mu))/len(mu)
    def tildeHmeasure(z):
        H=1/Gemp(z,mu)+z
        return H
    def Hnoise(z):
        H=z-1/Gnoise(z)
        return H
    #Definition of Kz
    def K(z,w):
        K=Hnoise(tildeHmeasure(w)-z)+z
        return K
    #Defintion of the imaginary line
    line=2*np.sqrt(2)*sigma_noise
    #Construction of w3 on the imaginary line
    sizemesh=int((returnmax-returnmin)/precisionmesh)
    w3real=np.zeros(sizemesh)
    w3imag=np.zeros(sizemesh)
    for i in range(sizemesh):
        z=returnmin+precisionmesh*i+line*1j
        w0=z
        err=abs(z)
        while err >=precisioniteration:
            w1=K(z,w0)
            err=abs(w1-w0)
            w0=w1
            if abs(w0-2)<0.00000001:
                print(w0)
        w3real[i]=w0.real
        w3imag[i]=w0.imag
    #construcion of the original measure convolved with Cauchy
    tilde_distri=np.zeros(sizemesh,dtype=complex)
#    init_distri=np.zeros(sizemesh)
    for i in range(sizemesh):
        tilde_distri[i]=Gemp(w3real[i]+w3imag[i]*1j,mu)
        #init_distri[i]=-1/np.pi*Gmeasure(returnmin+precisionmesh*i).imag
    #plt.plot(np.linspace(returnmin,returnmax,sizemesh),init_distri)
    print(line)
    return tilde_distri




def multiplicativeDeconvol(mu,Gnoise,line,returnmin,returnmax,precisionmesh,precisioniteration):
    # Definingparameters
    def Gemp(z,mu):
        return sum(1./(z-mu))/len(mu)
    def Kmeasure(z):
        H=z-z**2*1/Gemp(z**(-1),mu)
        return H
    def Hmeasure(z):
       H=z**(-1)-1/Gemp(z**(-1),mu)
       return H
    def Hnoise(z):
       H=z**(-1)-1/Gnoise(z**(-1))
       return H
    #Definition of Kz
    def K(z,w):
       K=z*Hnoise(Kmeasure(w)*z**(-1))**(-1)
       return K

    #Defintion of the imaginary line
#    R=max(2*np.sqrt(second_jacobi_noise),first_jacobi_noise)
#    line=max((6*(2*sigma_noise**2+np.sqrt(5*sigma_noise**4+2*sigma_measure**2*sigma_noise**2))),(R+3/2*np.sqrt(R**2+4*R*sigma_measure**2)))
#    print(line)
    #Construction of w3 on the imaginary line
    sizemesh=int((returnmax-returnmin)/precisionmesh)
    result=np.zeros(sizemesh,dtype=complex)
    for i in range(sizemesh):
        z=(returnmin+precisionmesh*i+line*1j)
        y=z**(-1)
        w0=y
        err=abs(y)
        while err >=precisioniteration:
            w1=K(y,w0)
            err=abs(w1-w0)
            w0=w1
            #print(w1)
            #if int(z+precisionmesh)-int(z)==1:
             #   print(w0)
        result[i]=(z-z*w0*Hmeasure(w0))**(-1)
    #construcion of the original measure convolved with Cauchy
    #print(line)
    return result


# Multiplicative deconvolution when the density of mu_3 is known (i.e not a sample of a random matrix)
def multiplicativeAnalDeconvol(density,domain,Gnoise,line,returnmin,returnmax,precisionmesh,precisioniteration):
    # Definingparameters
    def Gemp(z,domain,density):
        G=sum(1/(z-domain)*density)/sum(density)
        return G
    def Kmeasure(z):
        H=z-z**2*1/Gemp(z**(-1),domain,density)
        return H
    def Hmeasure(z):
       H=z**(-1)-1/Gemp(z**(-1),domain,density)
       return H
    def Hnoise(z):
       H=z**(-1)-1/Gnoise(z**(-1))
       return H
    #Definition of Kz
    def K(z,w):
       K=z*Hnoise(Kmeasure(w)*z**(-1))**(-1)
       return K

    #Defintion of the imaginary line
#    R=max(2*np.sqrt(second_jacobi_noise),first_jacobi_noise)
#    line=max((6*(2*sigma_noise**2+np.sqrt(5*sigma_noise**4+2*sigma_measure**2*sigma_noise**2))),(R+3/2*np.sqrt(R**2+4*R*sigma_measure**2)))
#    print(line)
    #Construction of w3 on the imaginary line
    sizemesh=int((returnmax-returnmin)/precisionmesh)
    result=np.zeros(sizemesh,dtype=complex)
    for i in range(sizemesh):
        z=(returnmin+precisionmesh*i+line*1j)
        print(z)
        y=z**(-1)
        w0=y
        err=abs(y)
        while err >=precisioniteration:
            w1=K(y,w0)
            err=abs(w1-w0)
            w0=w1
            #print(w1)
            #if int(z+precisionmesh)-int(z)==1:
             #   print(w0)
        result[i]=(z-z*w0*Hmeasure(w0))**(-1)
    #construcion of the original measure convolved with Cauchy
    #print(line)
    return result
    
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['reltol']=1e-13
cvxopt.solvers.options['abstol']=1e-13
cvxopt.solvers.options['maxiters']=1000
cvxopt.solvers.options['feastol']=1e-13
def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

#Definition ideal noise
def Gnoise(z):
    G=(z-sqrt(z)*sqrt(z-4))/(2*z)
    return G


h=10
htarget=h
precisionmesh=0.06
precisioniteration=1e-13
returnmin=-h
returnmax=h
sigma=13
#Parameters : htarget, sigma and precision have to the same as in the first step
precision=precisionmesh
hsource=10
#regularization parameter (to choose ! still work fairly well without the regularization parameter with the Levy distance; to elucidate...)
l=0.00001

import time

def freedeconvolutionresult(a):
		#step_1
                st = time.time()
                distribution=multiplicativeDeconvol(a,Gnoise,sigma,returnmin,returnmax,precisionmesh,precisioniteration) 
                t1 = time.time()
                #Cauchy Kernel
                n=int(2*htarget/precision)
                m=int(2*hsource/precision)
                K=np.zeros((n,m))
                Delta=1e3
                for i in range(n):
                        for j in range(m):
                                K[i,j]=Delta*sigma/(np.pi*((((j-i)*precision-hsource+htarget)**2+sigma**2)))

                #Optimization setting
                P = np.dot(np.transpose(K),K)+l**2*np.identity(m)
                q = -np.dot(np.transpose(K),-1*distribution.imag/np.pi).reshape((m,))

                q=matrix(q, (m,1),'d')

                np.shape(q)
                G =-np.identity(m)
                h = np.zeros(m).reshape((m,))
                A=Delta*np.ones(m)
                A = matrix(A, (1, m), 'd')
                b=np.ones(1)

                #solution
                R=cvxopt_solve_qp(P, q,G,h,A,b)
                if R is None:
                        R=np.array([1,0,0])

        
                #Plot
                y=np.linspace(-hsource,hsource,len(R))
                t2=time.time()
                print(" Tarrago : step1 = "+ str(t1-st) +  " step2 " + str(t2-t1))
                return(y,R)
                #plt.plot(y,R)
                #plt.show()

 

