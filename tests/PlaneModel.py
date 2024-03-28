import math
import matplotlib.pyplot as plt
import numpy as np

class PlaneModel():
    def __init__(self, M_critic=0.78, M=0.01):
        self._M_critic = M_critic
        self._M = M
        self.__Cx_min = 0.095 # drag coef.
        self.__Cz_max = 0.9 # lift coef.



    @property
    def M(self):
        return self._M
    
    @M.setter
    def M(self, M):
        self._M = M

    #aerodinamics with Mach
        
    def _Mach_Cz(self, Cz):
        Md = self._M_critic + (1 - self._M_critic) / 4
        if self._M <= self._M_critic:
            return Cz
        elif self._M <= Md:
            return Cz + (self._M - self._M_critic) / 10
        else:
            return Cz + 0.1 * (Md - self._M_critic) - 0.8 * (self._M - Md)
        
    
    def _Mach_Cx(self, Cx):
        if self._M <= self._M_critic:
            return Cx / np.sqrt(1 - self._M ** 2)
        else:
            return 7 * Cx * (self._M - self._M_critic) + Cx / np.sqrt(1 - self._M ** 2)
    
    #aerodinamics
    def Cz(self, alpha):

        alpha = alpha + np.radians(5)
        sign = np.sign(np.degrees(alpha))
        if abs(np.degrees(alpha)) < 15:
            Cz = sign * abs((np.degrees(alpha) / 15) * self.__Cz_max)
        elif abs(np.degrees(alpha)) < 20:
            Cz = sign * abs((1 - ((abs(np.degrees(alpha)) - 15) / 15)) * self.__Cz_max)
        else:
            Cz = 0
            
        return self._Mach_Cz(Cz)
    
    def Cx(self, alpha):

        alpha = alpha + np.radians(0)
        Cx = (np.degrees(alpha) * 0.02) ** 2 + self.__Cx_min
        return self._Mach_Cx(Cx)
        
    
        
if __name__ == '__main__':
    pm = PlaneModel()
    alphas = np.linspace(-5, 13, 60)
    Czs = [pm.Cz(alpha) for alpha in np.radians(alphas)]

    plt.plot(alphas, Czs)
    plt.savefig('tests/graphs/Cz_nomach')
        


    print(pm.M)