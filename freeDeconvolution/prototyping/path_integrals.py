import numpy as np
import sympy as sp
from newton_homotopy import homotopy_plus_Newton_implicit_function_computation


def path_integral(path_sample_points, function_values_on_points, is_path_a_contour = True , use_simpson = True):
    """Given a vector of complex (a+ib) sample points and their values via a complex-valued 
    function it estimates the corresponding path integral.
    The flag is_path_a_contour creates one extra difference between first and last points    
    """
    assert len(path_sample_points) == len(function_values_on_points)#Checks that we have the same number of sample points and function evaluations    
    number_evaluations = len(function_values_on_points)-1
    p = path_sample_points
    diffs_vector = [p[k+1]-p[k] for k in range(number_evaluations)]
    if not use_simpson:
        values_vector = [function_values_on_points[k] for k in range(number_evaluations)]    
    if use_simpson:
        values_vector = [(function_values_on_points[k]+function_values_on_points[k+1])/2 for k in range(number_evaluations)]    
    
    if is_path_a_contour:
        #on a contour we need to add one more difference and its corresponding function value
        diffs_vector.append(p[0]-p[-1])
        if not use_simpson:
            values_vector.append(function_values_on_points[-1])
        if use_simpson:
            values_vector.append((function_values_on_points[-1]+function_values_on_points[0])/2)

    summands = np.multiply(diffs_vector, values_vector)
    return np.sum(summands).evalf()


if __name__ == "__main__":
    #Example 1: Cauchy integral formula numerically...
    #Construction of M sample points in a loop around the origin
    t = sp.symbols("t")       
    path = 10*sp.cos(t)+ 12*sp.I*sp.sin(t) 
    M=100
    path_sample_points = [path.subs(t,k*(2*sp.pi)/M).evalf() for k in range(M)]
    #Construction of a function 
    z = sp.symbols("z")       
    f = 1/(z-9)
    function_values_on_points = [f.subs(z,v).evalf() for v in path_sample_points]
    #Computation of path integral
    result = path_integral(path_sample_points, function_values_on_points, is_path_a_contour = True)
    print((result/(2*sp.pi*sp.I)).evalf())#Notice excellent accuracy and robustness to changes in the chosen contour.

    #EXAMPLE 2:
    #Computation of an analytic branch of sqrt(z**2-1) in the complement of [-1,1]
    #NOTE: The principal branch of the logarithm defines a formula which fails to be analytic 
    #on purely imaginary numbers (leading to TWO different branches at each side of the imaginary 
    # numbers) so this is NOT THE FORMULA WE WANT. 
    # The following example provides a proper numerical construction
    z = sp.symbols("z")       
    y = sp.symbols("y")
    F = y**2-(z**2-1)#Implicit equation. We wish to solve F(z,y)=0 for y as function of z
    #z-path we wish to lift:
    radius = 3.0 #should be at least one
    t = sp.symbols("t")       
    path = radius * (sp.cos(t) + sp.I*sp.sin(t))
    M = 100
    path_sample_points = [path.subs(t,k*(2*sp.pi)/M).evalf() for k in range(1,M)]
    initial_point = path_sample_points[1]
    initial_value = sp.sqrt(initial_point**2-1).evalf()
    #We compute the lifted path...
    implicit_function_values = homotopy_plus_Newton_implicit_function_computation(
        dependent_variable = y,
        independent_variable = z,
        implicit_equation = F,
        independent_variable_sample_path=path_sample_points,
        dependent_var_initial_value=initial_value,
        num_newton_steps_per_point=10
    )
    # If radius is one then there is a bit of numerical error in the homotopy as we approach 
    # the branch points 1,-1. Not too much thou...

    #Finally we compute the value of the integral of G(z)=2(z-sqrt(z**2-1)) along the given contour.
    #More precisely we integrate only the second summand and comes out -1/2 as it should be!
    result = path_integral(path_sample_points, implicit_function_values, is_path_a_contour = True)
    print((result/(2*sp.pi*sp.I)).evalf()) #Accuracy becomes amazing as we consider larger radii


