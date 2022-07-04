import numpy as np
import sympy as sp
from newton_homotopy import newton_steps, homotopy_path_lift, compute_ramification_and_branch_points
from discrete_measures import Positive_Points_Measure
import cmath

#Example 0: Constructing and manipulating rational functions in sympy
z = sp.symbols("z") #To define rational functions we need a symbolic variable
f = (1/2)*1/(z-1) + (1/2)*1/(z-2)
f.simplify()#symbolic manipulation of rational functions
f.subs(z,4+3*sp.I)#Substitution of values and  complex numbers
sp.diff(f,z)# Symbolic differentiation
ratio = sp.ratsimp(f)#Write f as a rational function
sp.fraction(ratio) #returns a vector with numerator and denominator

#Example 1: Solving an equation with Newton's method
rh_value = f.subs(z,4+3*sp.I)
initial_guess = 3+2*sp.I
lh_function = f
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
t = sp.symbols("t")
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

#Example 3: Lift a path under the more substantial covering map coming from a discrete 
# measure with N atoms (we are computing 50 points in this lifted path to high accuracy).
N = 50
atoms = np.random.rand(N)#random atoms in [0,1]
weights = np.random.rand(N)#random atoms in [0,1]
weights = weights/np.sum(weights)
DM = Positive_Points_Measure(atoms,weights)
f = DM.Gfunction()
#We will lift the same path as in example 2 but the covering space will be given by the G of a random measure
t = sp.symbols("t")
rh_value = f.subs(z,4+3*sp.I)
path = sp.cos(t)+sp.I*sp.sin(t)+rh_value-1 #complex path
t_init = 0.0
t_final = 2*cmath.pi

initial_lifted_point = 4+3*sp.I
covering_map = f
num_computed_points = 50
num_newton_steps_per_t = 10
measured_times, computed_lifts = homotopy_path_lift(
    path, 
    t_init,
    t_final, 
    num_computed_points, 
    covering_map, 
    initial_lifted_point, 
    num_newton_steps_per_t=num_newton_steps_per_t)
print(computed_lifts)#Notice awesome agreement of last lift with initial lifted point! 
#This probably means the original path does not enclose any branch points.
#it works pretty well even with 100 points.

# Example 4: Computation of the ramification points of the covering via a mixture of symbolic 
# computation, companion matrices and Newton's method. 
# **Questions: What conditions does a "branch polynomial" satisfy? it is obviously real, 
# of degree 2(d-1) and.... what about conditioning of roots from coeffs? should one use 
# other bases and interpolate instead? See comments and example at end of newton_homotopy.py

z = sp.symbols("z") #To define rational functions we need a symbolic variable
f = (1/2)*1/(z-1) + (1/2)*1/(z-2)
covering_map = f
num_newton_steps = 10
ramification_points, branch_points = compute_ramification_and_branch_points(covering_map, num_newton_steps)
ramification_points
branch_points

# Example 5: For a discrete measure mu we will compute the operator \ell_\mu in two ways:
# directly from the definition of the measure and via contour integrals of the function G(z)
# we are hoping to compare their accuracy (since the G function is, in general, 
# easily recoverable on a contour).

N = 5
atoms = np.random.rand(N)#random atoms in [0,1]
weights = np.random.rand(N)#random atoms in [0,1]
weights = weights/np.sum(weights)
DM = Positive_Points_Measure(atoms,weights)
Gfunc = DM.Gfunction() #This is the rational G function of the measure

#Next we create a set of points along a contour which encloses the interval [0,1]
t = sp.symbols("t")
path = 2*sp.cos(t)+2*sp.I*sp.sin(t)
M=200
points_sample_on_path = [path.subs(t,x*2*sp.pi/M).evalf() for x in range(M)]
#We wish to estimate the expected value of the function T
x = sp.symbols("x")
T = sp.I**x      
DM.expected_value_func(T)
#Numerical integration via path
