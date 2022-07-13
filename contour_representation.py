import sympy as sp
import numpy as np
from path_integrals import path_integral
from newton_homotopy import homotopy_plus_Newton_implicit_function_computation
from path_integrals import path_integral
from Tchebyshev_interpolation import tchebyshev_coefficients_vector, tchebyshev_polynomials_vector

def cauchy_estimate_for_point_outside(path_samples, G_samples, new_point):
    """Given a contour representation (path_samples, G_samples) 
    and a new point z=new_point OUTSIDE the contour estimate the value of G(new_point)"""
    w = sp.symbols("w")
    new_path_samples = [(1/p).evalf() for p in path_samples]#with respect to w = 1/z
    w0 = (1/new_point).evalf()
    den = 1/(w-w0)
    denominators = np.array([den.subs(w,p).evalf() for p in new_path_samples])
    G_samples_np = np.array(G_samples)
    function_values = denominators * G_samples_np
    #Note that there should be a (-1) for preserving orientation
    return ((path_integral(new_path_samples, function_values))/((-2.0)*sp.pi*sp.I)).evalf()

if __name__=="__main__":
    #Example 1: Semicircle density in [-1,1]    
    #We sample G along a circular contour and try to recover properties of G and 
    # the underlying density from the contour info alone
    radius = 2.0 #should be at least one
    t = sp.symbols("t")       
    path = radius * (sp.cos(t) + sp.I*sp.sin(t))
    M = 500 #Increasing the number of samples seems to improve accuacy more than having several newton steps.
    path_sample_points = np.array([path.subs(t,k*(2*sp.pi)/M).evalf() for k in range(1,M)])
    # Recall that G(z) = 2(z-sqrt(z**2-1)), where
    # the square root should be analytic in the complement of [-1,1]. 
    # For a proper choice of square root we must describe it implicitly
    z,y = sp.symbols("z,y")       
    F = y**2-(z**2-1)#Implicit equation defining y as function of z, in this case the square root.
    #We wish to solve F(z,y)=0 for y as function of z starting at
    initial_value_z = path_sample_points[0]
    initial_value_y = sp.sqrt(initial_value_z**2-1).evalf() 
    #We compute the lifted path...
    implicit_function_values = homotopy_plus_Newton_implicit_function_computation(
        dependent_variable = y,
        independent_variable = z,
        implicit_equation = F,
        independent_variable_sample_path = path_sample_points,
        dependent_var_initial_value = initial_value_y,
        num_newton_steps_per_point = 8
    )
    #Now we compute the values of the function G along the path, obtaining the contour representation we wanted
    G_function_values = (path_sample_points-implicit_function_values)*2.0
    #Example 1.1: We compute the value of G at a point ABOVE/OUTSIDE from its contour representation
    new_point = (3.0+3.0*sp.I).evalf() # Outside the loop but on the first quadrant where wrong_G and the actual G function agree
    res = cauchy_estimate_for_point_outside(path_sample_points, G_function_values, new_point)
    wrong_G = 2.0*(z-sp.sqrt(z**2-1))
    target = wrong_G.subs(z,new_point).evalf()
    res - target #Indeed very small error.
    #Example 1.2: We compute the expected values from our contour:
    #Example 1.2.1: Expected value of constant function should be 1:
    res = (path_integral(
        path_sample_points, 
        G_function_values, 
        is_path_a_contour = True) /(2*sp.pi*sp.I)).evalf()
    target = 1.0
    res - target #Small error 
    #Example 1.2.2: Expected value of function z, or any odd power should be 0:
    function_values = G_function_values * path_sample_points
    res = (path_integral(
        path_sample_points, 
        function_values, 
        is_path_a_contour = True) /(2*sp.pi*sp.I)).evalf()
    target = 0.0
    res - target #Small error indeed 
    #Example 1.2.3: Expected value of function z**2 should be 1/4
    z_sq_path = path_sample_points * path_sample_points
    function_values = G_function_values * z_sq_path
    res = (path_integral(
        path_sample_points, 
        function_values, 
        is_path_a_contour = True) /(2*sp.pi*sp.I)).evalf()
    target = 0.25
    res - target #Small error indeed again...
    #Example 1.2.3: Expected value of function z**8 should be 1/8 see general formula here
    #https://en.wikipedia.org/wiki/Wigner_semicircle_distribution 
    z_4_path = np.array([(x**4).evalf() for x in path_sample_points])
    function_values = G_function_values * z_4_path
    res = (path_integral(
        path_sample_points, 
        function_values, 
        is_path_a_contour = True) /(2*sp.pi*sp.I)).evalf()
    target = 1.0/8.0
    res-target #Again remarkably small...

    #Example 1.3: Attempt at density recovery...
    #For density recovery we need an augmented path including segments towards the endpoints
    #For simplicity we use a disc of radius exactly one...
    t = sp.symbols("t")       
    path = sp.cos(t) + sp.I*sp.sin(t)
    M = 500 #Increasing the number of samples seems to improve accuacy more than having several newton steps.
    path_sample_points = np.array([path.subs(t,k*(2*sp.pi)/M).evalf() for k in range(1,M)])
    # For a proper choice of square root we must describe it implicitly
    z,y = sp.symbols("z,y")       
    F = y**2-(z**2-1)#Implicit equation defining y as function of z, in this case the square root.
    #We wish to construct the holomorphic sqrt(z**2-1) analytic on the complement of [-1,1] on the path
    initial_value_z = path_sample_points[0]
    initial_value_y = sp.sqrt(initial_value_z**2-1).evalf() 
    #We compute the lifted path...
    sqrt_function_values = homotopy_plus_Newton_implicit_function_computation(
        dependent_variable = y,
        independent_variable = z,
        implicit_equation = F,
        independent_variable_sample_path = path_sample_points,
        dependent_var_initial_value = initial_value_y,
        num_newton_steps_per_point = 8
    )
    G_function_values = (path_sample_points-sqrt_function_values)*2.0
    #Next we compute A(z):=z**2*(square root evaluated in 1/z) and B(z):=1/A(z)
    reciprocals_path = np.array([(1/z).evalf() for z in path_sample_points])
    initial_value_z = reciprocals_path[0]
    initial_value_y = sp.sqrt(initial_value_z**2-1).evalf() 
    sqrt_function_values_at_reciprocal = homotopy_plus_Newton_implicit_function_computation(
        dependent_variable = y,
        independent_variable = z,
        implicit_equation = F,
        independent_variable_sample_path = reciprocals_path,
        dependent_var_initial_value = initial_value_y,
        num_newton_steps_per_point = 8
    )
    #k=30
    #(sqrt_function_values_at_reciprocal[k]**2-reciprocals_path[k]**2+1).evalf()
    #((sqrt_function_values_at_reciprocal[k]*path_sample_points[k]**2).evalf())**2

    A_function_values = path_sample_points * sqrt_function_values_at_reciprocal
    test = A_function_values**2+path_sample_points**2-1.0
    test_f = np.array([x.evalf() for x in test])
    B_function_values = [(1/a).evalf() for a in A_function_values] 
    #B above is a version of 1/sqrt(1-z**2) analytic except on (-\inf,-1]\cup [1,\infty)
    # small test: The following vector should contain only zeroes...
    test = np.array([ ((1- path_sample_points[k]**2)*B_function_values[k]**2-1).evalf() for k in range(len(path_sample_points))])
    np.argmax(np.absolute(test)) #98,99,100,101 are problematic, not the others thou...
    #Remark: The function velues of B behave very well away from index 99 (exactly equal to -1)
    # We compute the path integrals on two regions, 
    # the upper half and the lower half independently removing the bad indices.
    # For comparison we compute them from the actual density first: 
    # Computation of the Tchebyshev series from semicircle law density
    x = sp.symbols("x")
    h = sp.sqrt(1-x**2)*(2/sp.pi) #Semicircle law density
    left_bound = -1.0
    right_bound = 1.0
    max_degree = 20
    #Next we compute the corresponding Tchebyshev coefficients up to degree max_degree
    h_coeffs_vec = tchebyshev_coefficients_vector(
        x,
        h, 
        left_bound= left_bound, 
        right_bound = right_bound, 
        max_degree = max_degree, 
        num_segments_for_approximation =50)    
    #The components of h_coeffs_vec are the a_j of the semicircle dist wrt to the Tchebyshev basis
    h_coeffs_vec = np.array(h_coeffs_vec)
    #Next we estimate them from contour integrals...
    tch_List = tchebyshev_polynomials_vector(x, max_degree, expand=True)
    #We begin by computing their values on a contour
    TValues_List = [] #This will be a list of arrays
    for k in range(max_degree):
        p = tch_List[k]
        values_vector = np.array([p.subs(x,z).evalf() for z in path_sample_points])
        TValues_List.append(values_vector)
    #Finally we compute the coefficients via contour integrals
    problematic_index = np.argmax(np.absolute(test))
    upper_range = slice(0,problematic_index-2)
    lower_range = slice(problematic_index+2,-1)
    slices = [lower_range, upper_range]
    estimated_coeffs = []
    #k=0
    #slice = lower_range

    for k in range(max_degree):
        res = 0.0
        #We will do the integral in two parts and add them for numerical accuracy        
        for slice in slices:
            path_points = path_sample_points[slice]
            T_values = TValues_List[k][slice]
            B_values = B_function_values[slice]
            G_values = G_function_values[slice]
            function_values = T_values * B_values * G_values
            res1 = path_integral(path_points, function_values, is_path_a_contour = False)
            res = res + (res1/(sp.I* sp.pi**2)).evalf()

        estimated_coeffs.append(res)        
    
    estimated_coeffs = np.array(estimated_coeffs)