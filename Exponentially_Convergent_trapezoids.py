import sympy as sp
import numpy as np
import math as math
from Tchebyshev_interpolation import Fourier_basis_vectors
from newton_homotopy import homotopy_plus_Newton_implicit_function_computation

if __name__ == "__main__":
    N = 500 #Order of roots of unity
    roots_of_unity = np.array([sp.exp(2*sp.I*sp.pi* k/N).evalf() for k in range(N)])
    #Ridiculously good Example below: we want to test the residue computation with a nontrivial infinite Laurent series
    #We know that the residue of exp(1/z)*z**(j) is 2piI/(j-1)!, let's see it computationally
    z = sp.symbols("z")
    j=8
    target_function = sp.exp(1/z)*z**(j)
    #target_function = z**30
    internal_function = target_function*z #Over unit circle this is the only change
    internal_function_samples = np.array([internal_function.subs(z,point).evalf() for point in roots_of_unity])
    estimate = internal_function_samples.mean()
    target = 1/math.factorial(j+1)
    estimate-target 
    #For a tougher example lets get all residues of degree up to 30 and see biggest error:
    differences = []
    for j in range(99):
        target_function = sp.exp(1/z)*z**(j)
        internal_function = target_function*z #Over unit circle this is the only change
        internal_function_samples = np.array([internal_function.subs(z,point).evalf() for point in roots_of_unity])
        estimate = internal_function_samples.mean()
        target = 1/math.factorial(j+1)
        differences.append(estimate-target)
    np.max(np.absolute(differences)) #This is ridiculously good, it matches up to coefficient 99 with perfect accuracy, using 100-th roots

    #Example 2: Exponentially convergent rule on smaller circle...
    z = sp.symbols("z")
    s = 0.5 #radius of the circle
    target_function = 1/z + 1/(z-1) 
    internal_function = target_function * z   
    internal_function_scaled = internal_function.subs(z,s*z)
    internal_function_scaled_samples =np.array([internal_function_scaled.subs(z,v).evalf() for v in roots_of_unity])
    estimate = internal_function_scaled_samples.mean()
    target = 1.0
    estimate-target
    #perfect agreement, if contour is sufficiently far from other poles, 
    # looses accuracy if the contur passes close to other poles of the function, say s=0.99 
    # Higher order of roots does compensate for this.

    #Example 3: Fourier expansion for the semicircle law from contour representation (assumed to be known at radius 3.0)
    # We wish to estimate the integrals:
    N = 200 #Order of roots of unity
    roots_of_unity = np.array([sp.exp(2*sp.I*sp.pi* k/N).evalf() for k in range(N)])
    s = 1/3 #radius of the w-image of our radius 3 circle
    scaled_roots_of_unity =  roots_of_unity * s
    inverse_sU = np.array([(1/p).evalf() for p in scaled_roots_of_unity],dtype=complex)
    #To use our exponentially convergent rule we want the mean of
    # (G(1/sv)F_k(1/sv))*1/sv over the roots of unity, we build it step by step...
    # We begin with the values of G...
    # Recall that for the semicircle law, G(z) = 2(z-sqrt(z**2-1)), where
    # the square root should be analytic in the complement of [-1,1]. 
    # For a proper choice of square root we must describe it implicitly
    z,y = sp.symbols("z,y")       
    F = y**2-(z**2-1)#Implicit equation defining y as function of z, in this case the square root.
    #We wish to solve F(z,y)=0 for y as function of z starting at
    initial_value_z = inverse_sU[0]
    initial_value_y = sp.sqrt(initial_value_z**2-1).evalf() 
    #We compute the lifted path...
    implicit_function_values = homotopy_plus_Newton_implicit_function_computation(
        dependent_variable = y,
        independent_variable = z,
        implicit_equation = F,
        independent_variable_sample_path = inverse_sU,
        dependent_var_initial_value = initial_value_y,
        num_newton_steps_per_point = 8
    )
    #Now we compute the values of the function G along the path, obtaining the contour representation we wanted
    G_function_values = np.array((inverse_sU-implicit_function_values)*2.0, dtype = complex)
    #Next, we want the Fourier expansion coefficients,
    max_degree = 5 #We want Fourier expansion in frequencies -max_degre,...,max_degree
    fourier_basis = Fourier_basis_vectors(z,max_degree, symmetric_interval_length = 1.0)
    fourier_basis_values_list = []
    #Compute the values of the Fourier functions
    for fourier_function in fourier_basis:
        fourier_basic_function_scaled_values = np.array([fourier_function.subs(z,point).evalf() for point in inverse_sU],dtype = complex)
        fourier_basis_values_list.append(fourier_basic_function_scaled_values)        
    #Finally compute the corresponding fourier coefficients:
    coefficients=[]
    numerator = G_function_values * inverse_sU
    for fourier_basic_values in fourier_basis_values_list:
        function_values = np.array(numerator * fourier_basic_values,dtype=complex)
        coefficients.append(function_values.mean()/2.0) #Normalization 2.0 comes from Fourier coefficient in [-1,1]
    #Problem: The coefficients give bad approximations because the function grows too much...




