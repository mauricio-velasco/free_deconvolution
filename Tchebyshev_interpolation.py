import numpy as np
import sympy as sp
import pdb
import matplotlib.pyplot as plt
from path_integrals import path_integral
from newton_homotopy import homotopy_plus_Newton_implicit_function_computation

def trigonometric_chebyshev_polynomials_vector(z,max_degree):
    """Given a sympy symbol z and a max_degree
    returns the vector of trigonometric Tchebyshev functions of degree at most max_degree,
    that is the functions trig_Tcheb_j satisfying
    T_j(cos(theta)) = trig_Tcheb_j(theta)
    """
    return ([sp.cos(k*z) for k in range(max_degree)])


def tchebyshev_polynomials_vector(z, max_degree, expand=True):
    """Given a sympy symbol z and a max_degree
    returns the vector of Tchebyshev polynomials of degree 
    at most max_degree    
    """
    results = []
    for k in range(max_degree):
        if k==0:
            results.append(z**0)
        if k==1:
            results.append(z)
        if k>1:
            new_poly = 2*z*results[-1]-results[-2]
            if expand:
                results.append(new_poly.expand()) 
            else:
                results.append(new_poly) 

    return results


def scaled_tchebyshev_polynomials_vector(z, max_degree, left_bound, right_bound, expand=True):
    Tch_List = tchebyshev_polynomials_vector(z, max_degree, expand=expand)
    a = left_bound
    b = right_bound
    assert a < b
    new_z = (1/(b-a))*(2*z-a-b) 
    scaled_Tch_List = [p.subs(z,new_z) for p in Tch_List]
    return scaled_Tch_List    

def scaled_weight_function(z,left_bound,right_bound):
    a = left_bound
    b = right_bound
    assert a < b
    f = 1/sp.sqrt(1-z**2)
    new_z = (1/(b-a))*(2*z-a-b) 
    return f.subs(z,new_z)


def parallelogram_integration(function_values,step_size):
    #Given the function values at a sequence of points we compute a parallelogram_integration
    number_evaluations = len(function_values)-1
    values_vector = np.array([(function_values[k]+function_values[k+1])/2 for k in range(number_evaluations)])    
    values_vector = values_vector * step_size
    return np.sum(values_vector).evalf()



def tchebyshev_coefficients_vector(z,sp_func, left_bound, right_bound, max_degree, num_segments_for_approximation):
    """We do numerical integration on the circle because on the interval the convergence is EXTREMELY slow
    so we use the trigonometric version of tchebyshev polys. This function computes the coefficients in the Chebyshev expansion of a function
    """
    a = left_bound
    b = right_bound
    angular_step_size = sp.pi.evalf()/(num_segments_for_approximation)
    angles = np.arange(0.0, sp.pi.evalf()+angular_step_size,angular_step_size)#it is essential to add the extra angular step size, otherwise we miss one rectangle due to python conventions.
    cosines = np.array([sp.cos(v).evalf() for v in angles])
    new_z = ((1+z)/2)*(b-a) + a #here z denotes the cosine of the angle
    trig_chebys_list = trigonometric_chebyshev_polynomials_vector(z,max_degree)
    sp_new_func = sp_func.subs(z,new_z)
    sp_new_func_values = np.array([sp_new_func.subs(z,cosine_value).evalf() for cosine_value in cosines])    
    coeffs = []
    for k in range(max_degree):
        angular_tcheby_poly = trig_chebys_list[k]
        angular_tcheby_poly_values = np.array([angular_tcheby_poly.subs(z,angle).evalf() for angle in angles])
        function_values = sp_new_func_values*angular_tcheby_poly_values
        coeffs.append((4/((b-a)*np.pi))*((b-a)/2)*parallelogram_integration(function_values, angular_step_size))
        #The factor (b-a)/2 comes from the change of variables formula of the substitution new_z above
    return coeffs

def test_ortogonality(max_degree, left_bound, right_bound, num_segments_for_approximation = 100):
    """Creates the scaled Tchebyshev polynomial and verifies numerically that they satisfy the 
    expected orthonormality conditions"""
    a = left_bound
    b = right_bound
    #We make a matrix of desired results
    target = np.array([[(sp.pi/2).evalf() if i == j else 0.0 for i in range(max_degree)] for j in range(max_degree)])    
    target[0,0] = sp.pi.evalf()
    target_matrix = target * ((b-a)/2)
    target_matrix = np.array(target_matrix, dtype=float)
    #Next we compute the Tchebyshev coefficients of Tchebyshev polynomials...
    x = sp.symbols("x") #all functions will have x as variable
    Scaled_tch_List = scaled_tchebyshev_polynomials_vector(x, max_degree, left_bound=left_bound,right_bound = right_bound)
    Coeffs_matrix_list = []
    for k in range(max_degree):        
        tcheby_poly = Scaled_tch_List[k]
        coeffs_vec = tchebyshev_coefficients_vector(
            x,
            tcheby_poly, 
            left_bound= left_bound, 
            right_bound = right_bound, 
            max_degree = max_degree, 
            num_segments_for_approximation = num_segments_for_approximation)    
        Coeffs_matrix_list.append(np.array(coeffs_vec))

    result_matrix = np.array(Coeffs_matrix_list)
    result_matrix = np.array(result_matrix, dtype=float)
    print(np.linalg.norm(result_matrix-target_matrix))
    assert np.allclose(result_matrix,target_matrix)

def evaluator_for_tchebyshev_series(coeffs_vector, left_bound,right_bound):
    x = sp.symbols("x")
    max_degree = len(coeffs_vector)
    Scaled_tch_List = scaled_tchebyshev_polynomials_vector(
        x, 
        max_degree = max_degree, 
        left_bound = left_bound,
        right_bound = right_bound)
    adjusted_vector = coeffs_vector.copy()
    adjusted_vector[0] = adjusted_vector[0]/2 #Formulas are correct when the constant coefficient is divided by 2
    def function_value(point):
        return np.sum([adjusted_vector[k]*Scaled_tch_List[k].subs(x,point).evalf() for k in range(max_degree)])        
    return function_value

#FOURIER SERIES on interval computation

def Fourier_basis_vectors(z,max_degree, symmetric_interval_length = 1.0):
    #Given a maximum degree returns a vector of the complex fourier basis
    M = max_degree
    L = symmetric_interval_length
    return([sp.exp((sp.pi*sp.I*k*z)/L) for k in range(-max_degree, max_degree+1)])

def Fourier_coefficients_vector(z,sp_func, max_degree, num_segments_for_approximation, symmetric_interval_length = 1.0):
    #TODO: Readjust for arbitrary interval, not just [-1,1]    
    L = symmetric_interval_length
    angular_step_size = (2.0*L)/(num_segments_for_approximation)
    angles = np.arange((-1.0)*L, L+angular_step_size, angular_step_size)#it is essential to add the extra angular step size, otherwise we miss one rectangle due to python conventions.
    sp_new_func_values = [sp_func.subs(z,angle).evalf() for angle in angles]
    fourier_basis = Fourier_basis_vectors(z,max_degree)
    fourier_basis.reverse() #reversing conjugates the list, makng exponents negative
    coeffs = []
    #Next we compute the integrals of the products
    for k in range(len(fourier_basis)):
            fourier_poly = fourier_basis[k]
            fourier_poly_values = np.array([fourier_poly.subs(z,angle).evalf() for angle in angles])
            function_values = sp_new_func_values * fourier_poly_values
            integral_estimate = path_integral(angles, function_values, is_path_a_contour=False)
            coeffs.append((1/(2.0*L))*integral_estimate)
    return coeffs
    

def evaluator_for_fourier_series(coeffs_vector):
    x = sp.symbols("x")
    max_degree = int((len(coeffs_vector)-1)/2)
    fourier_basis_vectors = Fourier_basis_vectors(x, max_degree = max_degree)

    def function_value(point):
        return sp.re(np.sum([coeffs_vector[k]*(fourier_basis_vectors[k].subs(x,point).evalf()) for k in range(len(coeffs_vector))]))       

    return function_value



if __name__=="__main__":
    #Usage examples:
    #Construction of Tchebyshev polynomials of degree at most N-1:
    x = sp.symbols("x")
    N = 8
    tch_List = tchebyshev_polynomials_vector(x, N, expand=True)
    #We can also scale them to lie between a and b
    Scaled_tch_List = scaled_tchebyshev_polynomials_vector(x, N, left_bound=-1.0,right_bound = 1.0)
    weight = scaled_weight_function(x,left_bound=-1.0,right_bound = 1.0)
    #Trigonometric chebyshev polynomials vector:
    trig_tchebyshevs_list = trigonometric_chebyshev_polynomials_vector(z,N)
    #Orthogonality verification of scaled polynomials:
    #test_ortogonality(max_degree =20, left_bound = -1.0, right_bound = 3.0,num_segments_for_approximation = 100)#If running this line produces no error it means it is working
    #NOTE: If we take approximations of degree around 25 the error in norm becomes e-08 which is considered too large by our very tough test.
    #First EXAMPLE: Computation of a Tchebyshev series
    x = sp.symbols("x")
    h = sp.sqrt(4-x**2)/(2*sp.pi) #Semicircle law density
    left_bound = -2.0
    right_bound = 2.0
    max_degree = 20
    #Next we compute the corresponding Tchebyshev coefficients up to degree max_degree
    h_coeffs_vec = tchebyshev_coefficients_vector(
        x,
        h, 
        left_bound= left_bound, 
        right_bound = right_bound, 
        max_degree = max_degree, 
        num_segments_for_approximation =50)    
    #We need a dictionary of Scaled Tchebyshev polynomials to write the approximation
    Scaled_tch_List = scaled_tchebyshev_polynomials_vector(
        x, 
        max_degree = max_degree, 
        left_bound = left_bound,
        right_bound = right_bound)

    #And we plot the resulting approximation...
    approximation_degrees = [4,6,25]
    colors = ["g","m","r"]
    space_grid = np.linspace(left_bound, right_bound, 500)
    fig = plt.figure( figsize = (12,7) )
    ax = fig.add_subplot( 111 )
    original_function_values = [h.subs(x,v).evalf() for v in space_grid]
    ax.plot(space_grid, original_function_values, c="b", label="semicircle", linewidth=3.5)

    for index in range(len(approximation_degrees)):
        degree_limit = approximation_degrees[index]
        #We have already computed the approximation degrees up to 20, but we plot them gradually to see the improvement
        partial_coeffs_vec = h_coeffs_vec[0:degree_limit]
        chebyshev_approx_evaluator = evaluator_for_tchebyshev_series(
            coeffs_vector = partial_coeffs_vec,
            left_bound = left_bound,
            right_bound=right_bound
        )
        current_color = colors[index]
        chebyshev_approx_function_values = [chebyshev_approx_evaluator(v) for v in space_grid]
        ax.plot(space_grid, chebyshev_approx_function_values, "--", c=current_color, label=f"chebyshev_approx d<= {degree_limit}", linewidth=2.0) 


    ax.set(xlabel='Space (x)', ylabel='Value',
        title='Density')
    ax.grid()
    fig.legend()
    fig.show()

    #Second EXAMPLE: Computation of a Fourier series
    x = sp.symbols("x")
    h = 2.0*sp.sqrt(1-x**2)/(sp.pi) #Semicircle law density
    left_bound = -1.0
    right_bound = 1.0
    max_degree = 50
    #We compute the chebyshev coefficients
    h_fourier_coeffs_vec = Fourier_coefficients_vector(
        x,
        h, 
        max_degree = max_degree, 
        num_segments_for_approximation =100)    

    #And we plot the resulting approximation...
    approximation_degrees = [2,3,50]
    colors = ["g","m","r"]
    space_grid = np.linspace(-1.0, 1.0, 100)
    fig = plt.figure( figsize = (12,7) )
    ax = fig.add_subplot( 111 )
    original_function_values = [h.subs(x,v).evalf() for v in space_grid]
    ax.plot(space_grid, original_function_values, c="b", label="semicircle", linewidth=3.5)
    midpoint = int((len(h_fourier_coeffs_vec)-1)/2)

    for index in range(len(approximation_degrees)):
        degree_limit = approximation_degrees[index]
        #We have already computed the approximation degrees up to 20, but we plot them gradually to see the improvement
        partial_coeffs_vec = h_fourier_coeffs_vec[midpoint-degree_limit:midpoint+degree_limit+1]
        fourier_approx_evaluator = evaluator_for_fourier_series(coeffs_vector = partial_coeffs_vec)
        current_color = colors[index]
        fourier_approx_function_values = [fourier_approx_evaluator(v) for v in space_grid]
        ax.plot(space_grid, fourier_approx_function_values, "--", c=current_color, label=f"Fourier_approx d<= {degree_limit}", linewidth=2.0) 


    ax.set(xlabel='Space (x)', ylabel='Value',
        title='Density')
    ax.grid()
    fig.legend()
    fig.show()
    midpoint = int((len(h_fourier_coeffs_vec)-1)/2)
    h_fourier_coeffs_vec[midpoint]

    #THIRD Example: Computation of Fourier coefficients from the contour representation...
    t = sp.symbols("t")       
    path = 3.0*(sp.cos(t) + sp.I*sp.sin(t))
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
        num_newton_steps_per_point = 10
    )
    G_function_values = (path_sample_points-sqrt_function_values)*2.0
    #Computation of Fourier coefficients from the G(z) transform...
    max_degree = 5 #The numerics go to hell at degree 5...
    fourier_basis = Fourier_basis_vectors(z,max_degree)    
    fourier_basis_values_vector = []
    for k in range(len(fourier_basis)):
        sp_fourier_basic_func = fourier_basis[k]
        fourier_basis_values = np.array([sp_fourier_basic_func.subs(z,point).evalf() for point in path_sample_points])    
        fourier_basis_values_vector.append(fourier_basis_values)
    #Fourier coeffs...
    fourier_coeffs_from_contour_vec = []
    for k in range(len(fourier_basis)):
        function_values_vector = fourier_basis_values_vector[k] * G_function_values
        new_coefficient = (path_integral(
            path_sample_points, 
            function_values_vector, 
            is_path_a_contour = True) /(2*(2*sp.pi*sp.I))).evalf()
        fourier_coeffs_from_contour_vec.append(new_coefficient)

    fourier_coeffs_from_contour_vec = np.array(fourier_coeffs_from_contour_vec)

    #We compare with the other computation of Fourier coefficients...
    x = sp.symbols("x")
    h = 2.0*sp.sqrt(1-x**2)/(sp.pi) #Semicircle law density
    left_bound = -1.0
    right_bound = 1.0
    max_degree = max_degree
    #We compute the chebyshev coefficients
    h_fourier_coeffs_vec = np.array(Fourier_coefficients_vector(
        x,
        h, 
        max_degree = max_degree, 
        num_segments_for_approximation =100))    

    test = h_fourier_coeffs_vec - fourier_coeffs_from_contour_vec


    #And we plot the resulting approximation...
    approximation_degrees = [2,3,20]
    colors = ["g","m","r"]
    space_grid = np.linspace(-1.0, 1.0, 100)
    fig = plt.figure( figsize = (12,7) )
    ax = fig.add_subplot( 111 )
    original_function_values = [h.subs(x,v).evalf() for v in space_grid]
    ax.plot(space_grid, original_function_values, c="b", label="semicircle", linewidth=3.5)
    midpoint = int((len(h_fourier_coeffs_vec)-1)/2)

    for index in range(len(approximation_degrees)):
        degree_limit = approximation_degrees[index]
        #We have already computed the approximation degrees up to degree=max_degree, but we plot them gradually to see the improvement
        partial_coeffs_vec = fourier_coeffs_from_contour_vec[midpoint-degree_limit:midpoint+degree_limit+1]
        fourier_approx_evaluator = evaluator_for_fourier_series(coeffs_vector = partial_coeffs_vec)
        current_color = colors[index]
        fourier_approx_function_values = [fourier_approx_evaluator(v) for v in space_grid]
        ax.plot(space_grid, fourier_approx_function_values, "--", c=current_color, label=f"Fourier_approx d<= {degree_limit}", linewidth=2.0) 


    ax.set(xlabel='Space (x)', ylabel='Value',
        title='Density via contour')
    ax.grid()
    fig.legend()
    fig.show()



    #COMMENT: The following is a very interesting example: The convergence on the interval for the Tchebyshev weight is awfully 
    # slow...we need 100000 samples to get a sense of the correct value of the weight function alone. This may be an obstacle for the proposed approach
    # the reason is that we are using equsipaced points which are really bad estimators.
    # Does the same thing happen with complex integration? 
    left_bound = -1
    right_bound = 1
    step_size = (right_bound-left_bound)/10000
    points = np.arange(left_bound+step_size, right_bound,step_size)
    values = np.array([weight.subs(x,point) for point in points])
    num_evals = len(values)-1
    para_values = np.array([(values[k]+values[k+1])/2 for k in range(num_evals)])
    np.sum(para_values*step_size)# NO bug, turns out the problem is extremely slow convergence...

    