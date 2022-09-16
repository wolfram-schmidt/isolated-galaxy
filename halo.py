import numpy as np

from astropy.constants import M_sun
from scipy.constants import G, parsec
from scipy.optimize import root

# select cosmology
from astropy.cosmology import Planck18 as cosmo

kpc = 1e3*parsec


class Halo:

    def __init__(self):

        self.halo_profile = None
        self.halo_mass = 0
        self.halo_scale = 0
        self.halo_radius_vir = 0
        self.halo_norm_fct = 0
        self.halo_central_pot = 0
        
        self.particles = None


    # set halo parameters
    def set_halo(self, mass, r_core=None, c=10, nfw=True, springel=False):

        self.halo_norm_fct = np.log(1 + c) - c/(1 + c)
        
        if r_core == None:
            # virial radius is the radius at which the mean enclosed dark matter density
            # is 200 times the critical density
            mass_vir = mass*M_sun.value
            self.halo_radius_vir = (3*mass_vir / (800*np.pi * cosmo.critical_density(0).si.value))**(1/3)
            
            # determine halo scale from virial mass
            if nfw:
                self.halo_scale = self.halo_radius_vir / c
                self.halo_mass = mass_vir
            else:
                if springel:
                    # Springel et al., MNRAS 361 (2005), eq. (2)
                    self.halo_scale = (self.halo_radius_vir / c) * (2*self.halo_norm_fct)**(1/2)
                    self.halo_mass = mass_vir * (1 + self.halo_scale / self.halo_radius_vir)**2
                else:
                    # find halo scale and mass such that the inner density profile matches the profile of
                    # an NFW halo of equal virial mass and eq. (3) in Hernquist, ApJ 356 (1990), is satisfied
                    b = 0.5*c**3 / self.halo_norm_fct
                    sol = root(lambda x: 1 + 2*x + x**2 - b*x**3, 1/c)
                    self.halo_scale = sol.x[0] * self.halo_radius_vir
                    self.halo_mass = mass_vir * (1 + sol.x[0])**2
            
        else:
            # halo scale equals core radius
            self.halo_scale = r_core*kpc
            self.halo_radius_vir = c * self.halo_scale
            self.halo_mass = mass*M_sun.value
                        
        if nfw:
            self.halo_profile = "NFW"
            self.halo_central_pot = -G*self.halo_mass / (self.halo_norm_fct * self.halo_scale)
        else:
            self.halo_profile = "Hernquist"
            self.halo_central_pot = -G*self.halo_mass / self.halo_scale
    
            
    # halo density profile
    def halo_dens(self, r):
                
        if self.halo_profile == "NFW":
            rho_0 = self.halo_mass / (4*np.pi * self.halo_scale**3 * self.halo_norm_fct)
            #print(f"Density coefficient (NFW) = {rho_0:.3e} kg/m**3")
            return rho_0 / (r * (1 + r)**2)
        
        elif self.halo_profile == "Hernquist":
            # Hernquist, ApJ 356 (1990), eq. (2)
            rho_0 = self.halo_mass / (2*np.pi * self.halo_scale**3)
            #print(f"Density coefficient (Hernquist) = {rho_0:.3e} kg/m**3")
            return rho_0 / (r * (r + 1)**3)
        
        else:
            print("Error: unknown halo profile")
            return None

    # cumulative halo mass
    def halo_mass(self, r):
                
        if self.halo_profile == "NFW":
            # ICICLE doc, eq. (11) 
            x = r/self.halo_scale
            return self.halo_mass * (np.log(1 + x) - x/(1 + x)) / self.halo_norm_fct
        
        elif self.halo_profile == "Hernquist":
            # Hernquist, ApJ 356 (1990), eq. (2)
            return self.halo_mass * r**2 / (r + self.halo_scale**2)
        
        else:
            print("Error: unknown halo profile")
            return None
        
    
    # halo potential
    def halo_pot(self, r):
            
        if self.halo_profile == "NFW":
            # ICICLE doc, eq. (13) 
            return self.halo_central_pot * np.log(1 + r) / r
        
        elif self.halo_profile == "Hernquist":
            # Hernquist, ApJ 356 (1990), eq. (5)
            return self.halo_central_pot / (r + 1)
        
        else:
            print("Error: unknown halo profile")
            return None
        
    
# === END OF CLASS DEFINITION ===

_usage = """
Usage:
\tpython halo.py test nfw       (NFW profile, virial mass = 1e10 solar masses)
\tpython halo.py test hq        (Hernquist profile, virial mass = 1e10 solar masses)
\tpython halo.py test hq 1e12   (Hernquist profile, virial mass = 1e12 solar masses)"""

if __name__ == '__main__':
    import sys
        
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == 'help'):
        print(_usage)
        
    elif len(sys.argv) > 1:
        
        if sys.argv[1] == 'test':
            test = Halo()
        
            if len(sys.argv) > 3:
                mass = float(sys.argv[3])
            else:
                mass = 1e10
        
            if len(sys.argv) > 2:
                if sys.argv[2] == 'nfw':
                    test.set_halo(mass)
                elif len(sys.argv) > 2 and sys.argv[2] == 'hq':
                    test.set_halo(mass, nfw=False)
                else:
                    sys.exit(_usage)
                
                print(f"Virial radius = {test.halo_radius_vir/kpc:.2f} kpc")
                print(f"Halo scale = {test.halo_scale/kpc:.2f} kpc")

            else:
                sys.exit(_usage)

        else:
            sys.exit(_usage)
