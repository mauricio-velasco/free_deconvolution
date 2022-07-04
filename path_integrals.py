import numpy as np
import sympy as sp

def path_integral(path_sample_points, function_values_on_points, is_path_a_contour = False , use_simpson = True):
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
    #Construction of M sample points in a loop around the origin...
    t = sp.symbols("t")       
    path = sp.cos(t)+ sp.I*sp.sin(t) 
    M=100
    path_sample_points = [path.subs(t,k*(2*sp.pi)/M).evalf() for k in range(M)]
    #Construction of a functions
    z = sp.symbols("z")       
    f = 1/z
    function_values_on_points = [f.subs(z,v).evalf() for v in path_sample_points]
    #Computation of path integral
    result = path_integral(path_sample_points, function_values_on_points, is_path_a_contour = True)
    print((result/(2*sp.pi*sp.I)).evalf())
