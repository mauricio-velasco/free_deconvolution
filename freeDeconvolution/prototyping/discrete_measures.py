import math
import numpy as np
import sympy as sp

class Positive_Points_Measure:
    """A Point_Measure class is given by a set of real nonnegative atoms with their respective 
    weights, nonnegative and summing to one. 
    It can compute the rational functions G(z) and M(z) of the measure.
    """
    def __init__(self, atoms, weights) -> None:
        assert(len(atoms) == len(weights))
        self.num_atoms = len(atoms)
        self.atoms = atoms
        self.weights = weights
        for k in atoms:
            assert(k>=0)
        for w in weights:
            assert(w>=0)
        assert(math.isclose(np.sum(weights),1.0))

    def Gfunction(self):
        #returns the function G(z) defined as the expected value of 1/(z-t) d\mu(t)
        z = sp.symbols("z")       
        w = self.weights
        t = self.atoms
        summands = [w[k]/(z-t[k]) for k in range(self.num_atoms) ]
        return (sum(summands))

    def Mfunction(self):
        #returns the function zG(z)-1
        z = sp.symbols("z")       
        w = self.weights
        t = self.atoms
        summands = [w[k]*t[k]/(z-t[k]) for k in range(self.num_atoms) ]
        return (sum(summands))

    def expected_value_func(self,sp_target_function):
        sp_target_function_symbols = list(sp_target_function.free_symbols)
        v = sp_target_function_symbols[0]
        w = self.weights
        t = self.atoms
        summands = [w[k]*sp_target_function.subs(v,t[k]).evalf() for k in range(self.num_atoms)]
        return sum(summands)


if __name__ == "__main__":
    #Example usage
    N = 10
    atoms = np.random.rand(N)#random atoms in [0,1]
    weights = np.random.rand(N)#random atoms in [0,1]
    weights = weights/np.sum(weights)
    DM = Positive_Points_Measure(atoms,weights)
    g = DM.Gfunction()
    m = DM.Mfunction()
    z = sp.symbols("z")       
    h = z*g-1 - m
    h.simplify()#Note that this is, numerically, equal to zero because the numerator is
    mprim = sp.diff(m,z)
    ramif_poly = sp.fraction(mprim.ratsimp())[0]
    ramif_poly = sp.poly(ramif_poly)
    monic_ramif_poly = ramif_poly.monic()
    monic_ramif_poly.all_coeffs()
    monic_ramif_poly.degree()
