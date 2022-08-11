import numpy as np
from freeDeconvolution import utils

class DiscreteMeasure:

    def __init__(self, support, weights=None):
        self.support = support.copy()
        if weights is None:
            self.weights = None
        else:
            self.weights = weights.copy()
        #
        self.zeroes_first_kind  = support
        self.zeroes_second_kind = np.zeros( (len(support)-1,) )

    def compute_second_kind(self):
        for i in range( len(self.support) - 1):
            z = utils.dichotomy( self.support[i], self.support[i+1], self.G_empirical)
            z = utils.newton_raphson( self.G_empirical, self.G_prime_empirical, z )
            assert( self.support[i] < z )
            assert( z < self.support[i+1])
            if( (self.support[i] > z) or (self.support[i+1] < z) ):
                print("i: ", i)
                print(self.support[i], " ", self.support[i+1])
                print("z: ", z)
                print("")
            self.zeroes_second_kind[i] = z


    def G_empirical(self, z):
        array = z[...,None]-self.support[...,:]
        return np.sum( 1/array, axis=-1)/len(self.support)
        # if self.weights is None:
        #     array = z[...,None]-self.support[...,:]
        #     return np.sum( 1/array, axis=-1)/len(self.support)
        # else:
        #     array = z[...,None]-self.support[...,:]
        #     array = weights[None,]*array
        #     return np.sum( 1/array, axis=-1)

    def G_prime_empirical(self, z):
        array = z[...,None]-self.support[...,:]
        return np.sum( -1/(array*array), axis=-1)/len(self.support)

    def G_second_empirical(self, z):
        array = z[...,None]-self.support[...,:]
        return np.sum( 2/(array*array*array), axis=-1)/len(self.support)

    def M_theoretical(self, z):
        return z*self.G_theoretical(z)-1

    def M_empirical(self, z):
        return z*self.G_empirical(z)-1

    def M_prime_empirical(self, z):
        return z*self.G_prime_empirical(z) + self.G_empirical(z)

    def M_second_empirical(self, z):
        return z*self.G_second_empirical(z) + 2*self.G_prime_empirical(z)

    def Markov_Krein(self, z):
        array1 = z[...,None]-self.zeroes_first_kind[...,:]
        array2 = z[...,None]-self.zeroes_second_kind[...,:]
        value  = 1/z + np.sum( 1/array2, axis=-1) - np.sum( 1/array1, axis=-1)
        return value/(2*len(self.support))

    def Markov_Krein_prime(self, z):
        array1 = z[...,None]-self.zeroes_first_kind[...,:]
        array2 = z[...,None]-self.zeroes_second_kind[...,:]
        value = -1/(z*z) - np.sum( 1/(array2*array2), axis=-1) + np.sum( 1/(array1*array1), axis=-1)
        return value/(2*len(self.support))