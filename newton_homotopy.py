import math
import sympy as sp
import numpy as np

def newton_steps(lh_function, rh_value, init_guess, num_newton_steps = 1):
    """We wish to solve an equation of the form f(z)=w_0
    where w_0 is a given value, f is a given function (initially rational in sympy)
    and init_guess is an initial value for the equation using Newton's method.
    """
    #TODO: Add Kantorovich guarantee for convergence
    exp_symbols = list(lh_function.free_symbols)
    symbol = exp_symbols[0]
    lh_function_prime = sp.diff(lh_function, symbol)
    values_list = []
    current_value = init_guess
    for k in range(num_newton_steps):
        z = symbol
        den = lh_function_prime.subs(z,current_value).evalf()
        num = lh_function.subs(z,current_value).evalf() - rh_value
        next_value = current_value - num/den 
        next_value = next_value.evalf()
        values_list.append(next_value) 
        current_value = next_value
    return values_list, next_value


def homotopy_path_lift(path, t_init,t_final, num_computed_points, covering_map, initial_lifted_point, num_newton_steps_per_t = 10):
    """Given a path p(t), an interval [t_init,t_final], a covering space map w=f(z)
    and an initial point z_0 in the fiber above p(t_init) the homotopy lifting property
    guarantees that there is a unique path sigma(t) satisfying f(sigma(t)) = p(t) for all t.
    This function computes the values of this unique path at num_computed_points equispaced 
    in the interval [t_init, f_final]
    """
    measured_times= [t_init]
    lifted_positions = [initial_lifted_point]
    delta = (t_final-t_init)/num_computed_points
    current_time = t_init
    current_position_z = initial_lifted_point
    path_symbols = list(path.free_symbols)
    t_symbol = path_symbols[0]
    path_prim = sp.diff(path,t_symbol)#symb compute the derivative of the path
    covering_symbols = list(covering_map.free_symbols)
    z_symbol = covering_symbols[0]
    covering_prim = sp.diff(covering_map,z_symbol)

    for k in range(num_computed_points):    
        path_prime = path_prim.subs(t_symbol, current_time).evalf()
        den = covering_prim.subs(z_symbol, current_position_z).evalf()        
        der = path_prime/den #TODO: Check non-vanishing -- guarantees that one is away from branch points
        next_position_guess_z = current_position_z + delta*der
        next_position_guess_z = next_position_guess_z.evalf()
        next_time = current_time + delta    
        #Now we improve our initial next_position_guess with Newton:
        rh_value = path.subs(t_symbol, next_time)
        improvements, next_position_z = newton_steps(covering_map, rh_value, next_position_guess_z, num_newton_steps=num_newton_steps_per_t)         
        #TODO: Check Kantorovich here!! If it fails redo with less delta
        measured_times.append(next_time)
        lifted_positions.append(next_position_z)
        #get ready for next step
        current_time = next_time
        current_position_z = next_position_z

    return measured_times, lifted_positions


def compute_companion_matrix(p):
    """Computes the companion matrix of a sympy polynomial
    p(z) given the polynomial and the variable
    """
    p = sp.poly(p)
    p=p.monic()
    d = p.degree()
    coeffs_vector = p.all_coeffs()
    companion_matrix = np.zeros([d,d],float)
    for k in range(d-1):
        companion_matrix[k+1,k] = 1.0
    for k in range(d):
        companion_matrix[k,d-1] =(-1)*coeffs_vector[d-k] 
    return companion_matrix

def compute_roots(p,num_newton_steps = 10):
    """Computes an estimate of the roots of a polynomial p by computing an initial guess from 
    companion matrices and successively improving those initial guesses with Newton.
    """
    p = sp.poly(p)
    M = compute_companion_matrix(p)
    initial_guesses = list(np.linalg.eigvals(M))
    final_estimates = []
    rh_value = 0.0 
    for initial_guess in initial_guesses:
        estimates, final_estimate = newton_steps(p, rh_value, initial_guess, num_newton_steps=num_newton_steps)
        final_estimates.append(final_estimate)
    return final_estimates

def compute_ramification_and_branch_points(covering_map, num_newton_steps):
    """Given a RATIONAL covering_map we compute its ramification 
    and branch points to high accuracy combining companion matrices and Newton's method"""
    ramification_points = []
    branch_points = []
    covering_map_symbols = list(covering_map.free_symbols)
    z = covering_map_symbols[0]
    covering_map_prim = sp.diff(covering_map,z) 
    rat = sp.ratsimp(covering_map_prim)   
    ramification_poly = sp.fraction(rat)[0]
    ramification_poly = sp.poly(ramification_poly)#we make it monic
    companion_matrix = compute_companion_matrix(ramification_poly)
    initial_guesses = np.linalg.eigvals(companion_matrix)    
    rh_value = 0.0 
    #Since evaluating the derivative covering_map_prim, or its derivative is not bad
    #It is perhaps better to search for a zero of the derivative of the original covering map instead of
    #only a zero of its numerator: #TODO: Understand this
    for initial_guess in initial_guesses:
        estimates, final_estimate = newton_steps(covering_map_prim, rh_value, initial_guess, num_newton_steps=num_newton_steps)
        ramification_points.append(final_estimate)
        branch_points.append(covering_map.subs(z,final_estimate).evalf())
    return ramification_points, branch_points



if __name__=="__main__":
    N = 25
    z = sp.symbols("z")
    p = sp.prod([z-t for t in range(N)])
    M = compute_companion_matrix(p)
    roots = list(np.linalg.eigvals(M))
    #NOTE: When N grows, like 40, the initial guesses coming from companion matrices suck
    #they even become complex numbers with imaginary parts that are NOT numerically zero.
    #the proverbial ill-conditioning of the monomial basis.
    #We can try correcting them using Newton...
    newton_roots = compute_roots(p,num_newton_steps=10)
    # Interestingly, when the degree is largish, around 25 Newton's method does not 
    # improve the quality of the solution noticeably!! This is kind of interesting since Newton 
    # relies only in evaluating the polynomial (and does not rely on the monomial basis). 
    # Off course, even a polynomial can be written in ways that make it easier or harder to evaluate.
    # Perhaps keeping the poles and not extracting the numerator is better for evaluation? 
    # Because the derivative of m(z) is an easy to evaluate sum of ratios.

