import sympy as sp
import numpy as np
import math as math
import matplotlib.pyplot as plt
from Tchebyshev_interpolation import Fourier_basis_vectors, legendre_polynomials_vector, Legendre_coefficients_vector, evaluator_for_legendre_series
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
    N = 500 #Order of roots of unity
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
    legendre_basis = Fourier_basis_vectors(z,max_degree, symmetric_interval_length = 1.0)
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
    #Problem: The coefficients give bad approximations because the function grows too much, they seem to diverge at degree 5

    #EXAMPLE 4: Legendre expansion for the semicircle law from contour representation (assumed to be known at radius 3.0)
    # We wish to estimate the integrals:
    N = 500 #Order of roots of unity
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
        num_newton_steps_per_point = 10
    )
    #Now we compute the values of the function G along the path, obtaining the contour representation we wanted
    G_function_values = np.array((inverse_sU-implicit_function_values)*2.0, dtype = complex)
    
    #Next, we want the Legendre expansion coefficients,
    max_degree = 20 #We want Legendre expansion in terms of the first max_degree Legendre polynomials
    legendre_basis = legendre_polynomials_vector(z, max_degree=max_degree)
    legendre_basis_values_list = []
    #Compute the values of the Legendre functions
    for legendre_function in legendre_basis:
        legendre_basic_function_scaled_values = np.array([legendre_function.subs(z,point).evalf() for point in inverse_sU],dtype = complex)
        legendre_basis_values_list.append(legendre_basic_function_scaled_values)        
    #Finally compute the corresponding Legendre coefficients:
    coefficients=[]
    numerator = G_function_values * inverse_sU
    for legendre_basic_function_values in legendre_basis_values_list:
        function_values = np.array(numerator * legendre_basic_function_values,dtype=complex)
        coefficients.append(function_values.mean()) 
    coefficients = np.real(coefficients) #Are these coefficients any good?
    normalized_legendre_coefficients_from_contour = [coefficients[k] * ((2*k+1)/(2.0)) for k in range(len(coefficients))] #Adjust with Legendre normalizing constant
    #And compare them with the coefficients we obtain from integration agains the density...
    x = sp.symbols("x")
    h = 2.0*sp.sqrt(1-x**2)/(sp.pi) #Semicircle law density
    #We compute the Legendre coefficients
    h_legendre_coeffs_vec = Legendre_coefficients_vector(
        x,
        h, 
        max_degree = max_degree, 
        num_segments_for_approximation = 100)    
    h_legendre_coeffs_vec = np.array(h_legendre_coeffs_vec,dtype=complex)
    h_legendre_coeffs_vec = np.real(h_legendre_coeffs_vec)
    test_error = normalized_legendre_coefficients_from_contour - h_legendre_coeffs_vec 
    #test_error should be small... and it is, on the order of 0.01
    #Plot results:
    #And we plot the resulting approximation... which turns out to be pretty good
    approximation_degrees = [20]
    colors = ["r","g","m"]
    space_grid = np.linspace(-1.0, 1.0, 200)
    fig = plt.figure( figsize = (12,7) )
    ax = fig.add_subplot( 111 )
    original_function_values = [h.subs(x,v).evalf() for v in space_grid]
    ax.plot(space_grid, original_function_values, c="b", label="semicircle", linewidth=3.5)

    for index in range(len(approximation_degrees)):
        degree_limit = approximation_degrees[index]
        #We have already computed the approximation degrees up to 20, but we plot them gradually to see the improvement
        partial_coeffs_vec = normalized_legendre_coefficients_from_contour[0:degree_limit]
        legendre_approx_evaluator = evaluator_for_legendre_series(real_coeffs_vector = partial_coeffs_vec)
        current_color = colors[index]
        legendre_approx_function_values = [legendre_approx_evaluator(v) for v in space_grid]
        ax.plot(space_grid, legendre_approx_function_values, "--", c=current_color, label=f"Legendre_approx d<= {degree_limit}", linewidth=2.0) 


    ax.set(xlabel='Space (x)', ylabel='Value',
        title='Density')
    ax.grid()
    fig.legend()
    fig.show()

