import sympy as sp
from newton_homotopy import newton_steps, homotopy_path_lift
import cmath

#Example 0: Constructing and manipulating rational functions in sympy
z = sp.symbols("z") #To define rational functions we need a symbolic variable
f = (1/2)*1/(z-1) + (1/2)*1/(z-2)
f.simplify()#symbolic manipulation of rational functions
f.subs(z,4+3*sp.I)#Substitution of values and  complex numbers
sp.diff(f,z)# Symbolic differentiation

#Example 1: Solving an equation with Newton's method
rh_value = f.subs(z,4+3*sp.I)
initial_guess = 3+2*sp.I
lh_function = f
lh_function_prime = sp.diff(f,z) 
iterates, final = newton_steps(lh_function, rh_value, initial_guess, num_newton_steps = 10)
print(iterates)
print(final) #Notice awesome agreement.

#Example 2: Lifting a path with homotopy + Newton method:
# We want to give a path p(t) in the base (i.e. in the w coordinate) and lift it 
# to the total space z of a covering space f(z)=w.
# The initial path is in the variable w and we wish to find the unique path in the z variable which starts at 
# a given initial position a satisfying f(a)=p(0). To find the path means 
# specifying it to machine precision at a set of times t_k.  
# To lift a path means to specify the lifted points
t = sp.symbols("z")
rh_value = f.subs(z,4+3*sp.I)
path = sp.cos(t)+sp.I*sp.sin(t)+rh_value-1 #complex path
t_init = 0.0
t_final = 2*cmath.pi
# The choice of constant in the path makes sure that the initial_lifted_point lies 
# in the fiber above path(t_init)
initial_lifted_point = 4+3*sp.I
covering_map = f
num_computed_points = 50
num_newton_steps_per_t = 30
measured_times, computed_lifts = homotopy_path_lift(
    path, 
    t_init,
    t_final, 
    num_computed_points, 
    covering_map, 
    initial_lifted_point, 
    num_newton_steps_per_t=num_newton_steps_per_t)
print(computed_lifts)#Notice awesome agreement of last lift with initial lifted point
