import sympy as sp
import numpy as np
from path_integrals import path_integral

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
    return ((path_integral(new_path_samples, function_values))/(2*sp.pi*sp.I)).evalf()

if __name__=="__main__":
    #Verify that sp.sqrt is the principal branch (i.e. with cut along negative real axis)
    a = sp.sqrt(sp.exp(sp.I *(sp.pi-0.0001)))
    b = sp.sqrt(sp.exp((-1)*sp.I *(sp.pi-0.0001)))
    a.evalf() # approx I
    b.evalf() # approx -I
    #Example 1: For the semicircle density in [-1,1] we have
    z = sp.symbols("z")
    G = 2*(z-sp.sqrt(z**2-1))  
    #We sample G along a circular contour and try to recover properties of G and the underlying density from the contour info alone
    radius = 3.0
    t = sp.symbols("t")
    p = radius *sp.cos(t)+ radius * sp.I*sp.sin(t)
    num_samples = 500
    path_samples = [p.subs(t, k*(2*sp.pi)/num_samples).evalf() for k in range(num_samples)]
    values_vector = [G.subs(z,p).evalf() for p in path_samples]
    #The pair (path_samples, values_vector) is the contour representation of the semicircle density
    new_point = 5.0 + 1*sp.I
    res = cauchy_estimate_for_point_outside(path_samples, values_vector, new_point)
    target = G.subs(z,new_point).evalf()
    res - target