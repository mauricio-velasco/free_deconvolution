import cvxpy as cp
import numpy as np
import scipy.optimize as opt

# Functions re, im, sum and other formal computation things
from sympy import *

def stieltjes(z_array, empirical_measure):
    array = z_array[...,None]-empirical_measure[...,:]
    return np.sum( -1/array, axis=-1)/len(empirical_measure)

def build_dictionary( nu, c, spectrum):
    Z = []
    for k in range(len(nu)):
        def f(z):
            z = complex(z[0],z[1])
            stieljes   = np.sum(1/(spectrum-z))/len(spectrum)
            expression = -(1-c)/z + c*stieljes - nu[k]
            return((re(expression),im(expression)))
        (r,i) = opt.fsolve(f,(1,1))
        z = complex(r,i)
        Z.append(z)
    Z = np.array( Z )
    return Z

def perform_cvx_optimization( dictionary, T, c, norm_type, verbose=False):
  Z, nu  = dictionary
  assert(len(Z) == len(nu))

  #print( "Building cvxpy program..." )
  # Weights
  W = cp.Variable(len(T),complex=False)
  # Constrains
  const=[]

  # Form array of errors e_j's
  e_array = []
  for i in range(len(nu)):
      # Does not work
      # e = 1/nu[i] + Z[i] - c*sum([T[j]*W[j]/(1+T[j]*nu[i]) for j in range(len(T)) ])
      # Strangely works ==> cvxpy has a bad overloading of operators * / + - ?
      integrand = T/(1+T*nu[i])
      summand   = [ W[j]*integrand[j] for j in range(len(T)) ]
      e_i = 1/nu[i] + Z[i] - c*sum(summand)
      e_array.append( e_i )
  e_vector = cp.bmat( [e_array] )

  # Form objective
  if norm_type=='linfty':
    # Mute variable for minimization
    u = cp.Variable(1)
    objective= cp.Minimize(u)
    for e in e_array:
        const.append(cp.real(e)<=u)
        const.append(cp.real(e)>=-u)
        const.append(cp.imag(e)<=u)
        const.append(cp.imag(e)>=-u)
    # end for
  elif norm_type=='l1':
    #e = sum( [ cp.abs(e_j) for e_j in e_array ] )/len(nu)
    e = cp.norm1( e_vector )/len(nu)
    objective = cp.Minimize(e)
  elif norm_type=='l2':
    e = cp.norm2( e_vector )/np.sqrt( len(nu) )
    objective = cp.Minimize(e)

  # Final constrains
  const.append(W>=0)
  const.append(sum(W)==1)

  #print( "Solving the convex problem...")
  problem = cp.Problem(objective, const)
  result  = problem.solve(verbose=verbose)

  return W.value, objective.value