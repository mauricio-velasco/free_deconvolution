import math
import sympy as sp

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
