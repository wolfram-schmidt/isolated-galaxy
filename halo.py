import numpy as np

from astropy.constants import M_sun
from scipy.constants import G, parsec
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import root, newton

# select cosmology
from astropy.cosmology import Planck18 as cosmo

kpc = 1e3*parsec


class Halo:

    def __init__(self):

        self.halo_profile = None
        self.halo_mass = 0
        self.halo_scale = 0
        self.halo_radius_vir = 0
        self.halo_radius_max = 0
        self.halo_norm_fct = 0
        self.halo_central_pot = 0
        
        self.particles = None
        self.particle_mass = 0

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
    def halo_cumulative_mass(self, r):
        
        if self.halo_profile == "NFW":
            # ICICLE doc, eq. (11) 
            return self.halo_mass * (np.log(1 + r) - r/(1 + r)) / self.halo_norm_fct
        
        elif self.halo_profile == "Hernquist":
            # Hernquist, ApJ 356 (1990), eq. (2)
            return self.halo_mass * (r / (1 + r))**2
        
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
        
    # integrand in Eddington's formula for NFW distribution function
    # ICICLE doc, eq. (17)
    def nfw_eddington(self, r, q_sqr):
    
        rho = 1/(r*(1 + r)**2)
        mass = (np.log(1 + r) - r/(1 + r))
    
        return rho*r**2 * (6*mass + r*(r + 1)*(3*r + 1)*rho) / \
               (2*(mass*(1 + r))**2 * (q_sqr - np.log(1 + r)/r)**(1/2))

        
    # distribution function
    def halo_distribution(self, energy):
        
        if not isinstance(energy, np.ndarray):
            print("Error: parameters of halo_distribution must be arrays")
            return None
            
        q_sqr = energy / (-self.halo_central_pot)
        v_g = (-self.halo_central_pot)**(1/2)

        if self.halo_profile == "NFW":
            f = np.zeros(energy.size)
            
            for i in range(energy.size):
                if q_sqr[i] >= 1e-6:
                    if q_sqr[i] > 1:
                        f[i] = np.nan
                    else:
                        x0 = (1 - q_sqr[i])/q_sqr[i] # position of maximum, function decreases for x > x0
                        # find radius where energy equals potential energy
                        x1 = newton(lambda x: np.log(1 + x) - q_sqr[i]*x, 1.01 * x0,
                                    lambda x: 1/(1 + x) - q_sqr[i]) 
                        #print(i, q_sqr[i], x0, x1, np.log(1 + x1) - q_sqr[i]*x1)
                        
                        integr = quad(self.nfw_eddington, x1, np.inf, args=(q_sqr[i]))
                        #print(i, q_sqr[i], x1, integr[0])
                        f[i] = integr[0]
           
            return f / (2**(1/2) * 4*np.pi*G * self.halo_scale**2 * v_g)
        
        elif self.halo_profile == "Hernquist":
            # ICICLE doc, eq. (36)
            q = q_sqr**(1/2)
            f = (3*np.arcsin(q) + q * (1 - q_sqr)**(1/2) * (1 - 2*q_sqr) * (8*q_sqr**2 - 8*q_sqr - 3)) * \
                (1 - q_sqr)**(-5/2)
            return self.halo_mass * f / (8 * 2**(1/2) * (np.pi * self.halo_scale * v_g)**3)
        
        else:
            print("Error: unknown halo profile")
            return None

    
    # auxiliary method to generate uniformely distributed points on unit sphere
    def isotropic(self, n):

        z = np.random.uniform(-1, 1, int(n))         
        phi = np.random.uniform(0, 2*np.pi, int(n))
        
        x = (1 - z*z)**(1/2) * np.cos(phi)
        y = (1 - z*z)**(1/2) * np.sin(phi)

        return x, y, z;

    
    # generate initial conditions for particles
    def generate_particles(self, n_part, r_max=None):

        n_part = int(n_part)

        if r_max == None:
            self.halo_radius_max = self.halo_radius_vir
        else:
            self.halo_radius_max = r_max*kpc
            
        try:
            if n_part > 0 and self.halo_radius_max > 0:
                self.particles = np.zeros((6, n_part))
            else:
                raise ValueError
        except ValueError:
            print("ValueError: number of particles and cutoff radius must be positive")           
            return None, None, None;

        # invert to radial distances relative to halo scale for
        # uniformly distributed sample of masses
        if self.halo_profile == "NFW":
            # mass within r_max
            m_max = self.halo_cumulative_mass(self.halo_radius_max / self.halo_scale) * \
                    self.halo_norm_fct / self.halo_mass
            #print(m_max)
            
            r_sample = np.ones(n_part)
            for i in range(n_part):
                m = np.random.uniform(0, m_max) 
                sol = root(lambda x: np.log(1 + x) - x/(1 + x) - m, r_sample[i])
                r_sample[i] = sol.x[0]
                #print(i, m, r_sample[i])
                
        elif self.halo_profile == "Hernquist":
            m_sample = np.random.random_sample(n_part) # normalized to mass within r_max
            r_sample = (self.halo_radius_max / (self.halo_scale + self.halo_radius_max)) * \
                       m_sample**(1/2)
            r_sample /= (1 - r_sample)
            
        else:
            print("Error: unknown halo profile")
            return None, None, None;

        self.particle_mass = self.halo_cumulative_mass(self.halo_radius_max / self.halo_scale) / n_part
        
        # set random particle positions in pc
        self.particles[0,:], self.particles[1,:], self.particles[2,:] = \
            (self.halo_scale * r_sample / parsec) * self.isotropic(n_part)

        # potential energy
        p_sample = self.halo_pot(r_sample)
        
        # compute distribution function
        energy = np.linspace(0, -p_sample.min(), 1000)
        f = self.halo_distribution(energy)

        # choose particle energy from distribution function
        e_sample = np.zeros(n_part)
        v_sample = np.zeros(n_part)
        for i in range(n_part):
            psi = -p_sample[i]
            # Boolean array to select possible total energies for given potential energy 
            physical = energy < psi
            # cummulative distribution, ICICLE doc, eq. (7)
            CMF = cumulative_trapezoid(f[physical] * (psi - energy[physical])**(1/2),
                                       energy[physical], initial=0)
            # find energy at which normalized CMF equals chosen random number
            e_sample[i] = np.interp(CMF[-1]*np.random.random_sample(), CMF, energy[physical])
            v_sample[i] = (2*(psi - e_sample[i]))**(1/2)
            
        # set random particle velocities in km/s
        self.particles[3,:], self.particles[4,:], self.particles[5,:] = \
            1e-3*v_sample * self.isotropic(n_part)
        
        return r_sample, p_sample, e_sample;


    # print particle data to file
    def save_particles(self, file):

        data_file = file + ".dat"

        n_part = self.particles[0].size
        
        if self.halo_profile != None:
            with open(file + "-header", 'w') as f:
                f.write("Density profile: " + self.halo_profile + "\n")
                f.write(f"Halo mass:     {self.halo_mass/M_sun.value:.3e} M_sun\n")
                f.write(f"Halo scale:    {self.halo_scale/kpc:.3f} kpc\n")
                f.write(f"Virial radius: {self.halo_radius_vir/kpc:.3f} kpc\n")
                f.write(f"Cutoff radius: {self.halo_radius_max/kpc:.3f} kpc\n")
                f.write("Total mass:    {:.10e} M_sun\n".format(n_part * self.particle_mass / M_sun.value))
                f.write("Particle mass: {:.10e} M_sun\n".format(self.particle_mass / M_sun.value))
                f.write("Particle ICs (x, y, z, vx, vy, vz, m) are stored in " + data_file)

        # initial conditions
        # add identical particle masses in column #7
        if self.particles.any() != None:
            np.savetxt(data_file,
                       np.append(self.particles.transpose(),
                                 (self.particle_mass / M_sun.value) * np.ones((n_part, 1)), axis=1),
                       fmt='%17.10e')

    
# === END OF CLASS DEFINITION ===

_usage = """
Usage:
\tpython halo.py test nfw       (NFW profile, virial mass = 1e10 solar masses)
\tpython halo.py test hq        (Hernquist profile, virial mass = 1e10 solar masses)
\tpython halo.py test hq 1e12   (Hernquist profile, virial mass = 1e12 solar masses)
\tpython halo.py ictest hq 1e12 (also print ICs for 100 random particles)"""

if __name__ == '__main__':
    import sys
        
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == 'help'):
        print(_usage)
        
    elif len(sys.argv) > 1:
        
        if sys.argv[1] == 'test' or sys.argv[1] == 'ictest':
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

            if sys.argv[1] == 'ictest':
                test.generate_particles(100)
                print("Particle ICs (x, y, z, vx, vy, vz):")
                
                for row in test.particles.transpose():
                    print("{:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f} ".format(*row))

        else:
            sys.exit(_usage)
