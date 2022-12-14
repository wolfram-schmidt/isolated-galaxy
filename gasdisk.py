from halo import np, Halo

#import numpy as np
import scipy.special as spec

from astropy.constants import M_sun
from scipy.constants import G, k, m_p, parsec
from scipy.integrate import quad, solve_ivp
from scipy.optimize import root

kpc = 1e3*parsec


class GasDisk(Halo):

    def __init__(self, mass, r_s, z_s, temp=1e4, mu=1, gamma=5/3, disk='double exponential'):

        super().__init__()
        
        try:
            if mass > 0 and r_s > 0 and z_s > 0: 
                self.scale_length = kpc*r_s
                self.scale_height = kpc*z_s
                self.central_density = mass*M_sun.value / \
                    (4*np.pi * self.scale_length**2 * self.scale_height)
            else:
                raise ValueError
        except ValueError:
            self.central_density = None
            self.scale_length = None
            self.scale_height = None
            print("Invalid argument: disk density and scales must be positive")

        self.disk = disk
        self.sound_speed_sqr = k*temp/(mu*m_p)
        
        # mesh is filled by compute() method
        self._mesh_r = None
        self._mesh_z = None        
        self._mesh_rho = None
        self._mesh_v_rot = None
    
    
    # halo potential as function of disk coordinates
    def halo_pot_cyl(self, r, z):
            
        # three-dimensional radius
        r3 = (r*r + z*z)**(1/2)
            
        if self.halo_profile == "NFW":
            # https://en.wikipedia.org/wiki/Navarro_Frenk_White_profile
            return self.halo_central_pot * np.log(1 + r3) / r3
        
        elif self.halo_profile == "Hernquist":
            # Hernquist, ApJ 356 (1990), eq. (5)
            return self.halo_central_pot / (r3 + 1)
        
        else:
            print("Error: unknown halo profile")
            return None
    
    
    # computes first and second derivates of the vertical potential difference of gas in NFW halo
    # Wang et al., MNRAS 407, eq. (19)
    def _gas_pot_nfw_derv(self, z_si, state, r, a):
      
        # three-dimensional radius
        z = z_si/self.halo_scale
        r3 = (r*r + z*z)**(1/2)
        
        halo_pot_diff = self.halo_central_pot * (np.log(1 + r3)/r3 - np.log(1 + r)/r)
            
        return np.array([state[1], 
                        a * np.exp(-(state[0] + halo_pot_diff) / self.sound_speed_sqr)])
    
 
    # computes first and second derivates of the vertical potential difference of gas in Hernquist halo
    # Wang et al., MNRAS 407, eq. (19)
    def _gas_pot_hernq_derv(self, z_si, state, r, a):
      
        # three-dimensional radius
        z = z_si/self.halo_scale
        r3 = (r*r + z*z)**(1/2)

        # analytically equal to 1/(r3 + 1) - 1/(r + 1), but numerically more precise
        halo_pot_diff = self.halo_central_pot * (r - r3)/((r3 + 1)*(r + 1))
            
        #print(f"{z:.2f}  {r:.2f}  {state[0]:.2e}  {state[1]:.2e}  {halo_pot_diff:.2e}")            
        return np.array([state[1], 
                        a * np.exp(-(state[0] + halo_pot_diff) / self.sound_speed_sqr)])

    
    # compute gas density and rotation curve
    def compute(self, r_max, z_max, dr=0.1, dz=0.1, n_r=None, n_z=None, 
                err_min=1e-3, r_min=1e-6, k_max=10, verb=False):
        
        if n_r == None:
            r = np.arange(0, kpc*r_max/self.scale_length, dr)
        else:
            try:
                if int(n_r) >= 0:
                    r = np.linspace(0, kpc*r_max/self.scale_length, int(n_r)+1)
                else:
                    raise ValueError
            except ValueError:
                print("ValueError: number of zones must be positive")           
                return         
            
        if n_z == None:
            z = np.arange(0, kpc*z_max/self.scale_height, dz)
        else:
            try:
                if int(n_z) >= 0:
                    z = np.linspace(0, kpc*z_max/self.scale_height, int(n_z)+1)
                else:
                    raise ValueError
            except ValueError:
                print("ValueError: number of zones must be positive")           
                return
            
        self._mesh_r, self._mesh_z = np.meshgrid(r, z)

        # AGORA isolated disk
        if self.disk == 'double exponential':
            
            self._mesh_rho = self.central_density * np.exp(-self._mesh_r)*np.exp(-self._mesh_z)
            
        # alogrithm described in Wang et al., MNRAS 407 (2010), section 2.2.3
        elif self.disk == 'equilibrium':
            
            if self.halo_profile == None:
                print("Error: halo profile not specified")
                return

            poisson_coeff = 4*np.pi*G
            r = np.where(r < r_min, r_min, r)
            
            # factors to rescale dimensionless coordinates from disk to halo
            rescale_r = self.scale_length / self.halo_scale
            rescale_z = self.scale_height / self.halo_scale

            # assume an exponential surface density 
            sigma_0 = 2*self.scale_height * self.central_density 
            sigma = sigma_0 * np.exp(-r)
            if (verb):
                print(f"Central surface density = {sigma_0:.3e} kg/m**2 = {sigma_0*kpc**2/M_sun.value:.3e} M_sun/kpc**2\n")
                     
            # initialize midplane density
            midplane_rho = 0.5*sigma / self.scale_height
                    
            # auxiliary arrays
            h = np.zeros(sigma.shape)
            gas_pot = np.zeros(self._mesh_r.shape)
                    
            k = 1
            err = 1
            while err > err_min:
                if verb:
                    print(f"{k:d}. iteration:")
                    
                for i in range(r.size):
                    if verb:
                        print(f"   solving ivp for r = {r[i]:.2f}, midplance density = {midplane_rho[i]:.3e} kg/m**3")
                        
                    # Wang et al., MNRAS 407 (2010), eq. (19)
                    if self.halo_profile == "NFW":
                        tmp = solve_ivp(self._gas_pot_nfw_derv, [0, self.scale_height*z[-1]], [0, 0], 
                                        args=(r[i]*rescale_r, poisson_coeff*midplane_rho[i]),
                                        dense_output=True)
                    elif self.halo_profile == "Hernquist":
                        tmp = solve_ivp(self._gas_pot_hernq_derv, [0, self.scale_height*z[-1]], [0, 0], 
                                        args=(r[i]*rescale_r, poisson_coeff*midplane_rho[i]),
                                        dense_output=True)
                    else:
                        print("Error: unknown halo profile")
                        return

                    # potential difference of gas is first component of solution
                    gas_pot[:, i] = tmp.sol(self.scale_height*z)[0]
                            
                    if verb:
                        print(f"   integrating potential from z = 0 to {z[-1]:.2f}")
                        
                    # Wang et al., MNRAS 407 (2010), eq. (25)
                    integr = quad(lambda zeta: np.exp(-(tmp.sol(self.scale_height*zeta)[0] + 
                                                        self.halo_pot_cyl(r[i]*rescale_r, zeta*rescale_z) - 
                                                        self.halo_pot_cyl(r[i]*rescale_r, 0)) / 
                                                      self.sound_speed_sqr),
                                  0, z[-1])
                    
                    # integral (first element returned by quad) defines r-dependent scale height
                    h[i] = self.scale_height*integr[0]
                    
                # Wang et al., MNRAS 407 (2010), eq. (25) 
                midplane_rho_new = 0.5*sigma/h
                
                err = np.mean(np.abs(midplane_rho_new - midplane_rho)/midplane_rho)
                if verb:
                    print(f"   mean relative error = {err:.3e}\n")
                        
                # iterate
                midplane_rho = midplane_rho_new
                k += 1
                if k > k_max: 
                    print("Error: accuracy not reached after maximum number of iterations")
                    return
                
            self._mesh_rho = np.zeros(self._mesh_r.shape)
                        
            print(f"Accuracy reached after {k:d} iterations:")
            print("      r       h")
            for i in range(r.size):
                print(f"   {r[i]:6.2f}  {h[i]/self.scale_height:8.4f}")
                
                # Wang et al., MNRAS 407 (2010), eq. (18)
                self._mesh_rho[:,i] = midplane_rho[i] * \
                    np.exp(-(gas_pot[:,i] + self.halo_pot_cyl(r[i]*rescale_r, z*rescale_z) - 
                                            self.halo_pot_cyl(r[i]*rescale_r, 0)) / 
                           self.sound_speed_sqr)
                
            # rotation curve: contribution of gas disc
            # Wang et al., MNRAS 407 (2010), eq. (29)
            y = 0.5*r
            v_gas_sqr = poisson_coeff * sigma_0 * self.scale_length * y**2 * \
                        (spec.iv(0, y) * spec.kn(0, y) - spec.iv(1, y) * spec.kn(1, y))
            
            # rotation curve: contribution of pressure gradient
            # Wang et al., MNRAS 407 (2010), eq. (30)
            # compute density gradient r d(log rho)/dr using finite differences
            log_rho = np.log(midplane_rho)
            fct = r[1:-1]/(r[2:] - r[:-2])
            
            v_press_sqr = np.zeros(r.size)
            v_press_sqr[1:-1] = fct*(log_rho[2:] - log_rho[:-2])
            v_press_sqr[-1] = r[-1]*(log_rho[-1] - log_rho[-2])/(r[-1] - r[-2])
            v_press_sqr *= self.sound_speed_sqr
            
            # rotation curve: contribution of dark matter halo
            v_halo_sqr = np.zeros(self._mesh_r.shape)
            
            # need three-dimensional radius in units of halo scale
            r3 = ((np.clip(self._mesh_r, r_min, r_max)*rescale_r)**2 + 
                  (self._mesh_z*rescale_z)**2)**(1/2)
            if self.halo_profile == "NFW":
                # Navarro et al., ApJ 462 (1996), eq. (5)
                v_halo_sqr = -self.halo_central_pot * (np.log(1 + r3) - r3/(1 + r3)) / r3
            if self.halo_profile == "Hernquist":
                # Hernquist, ApJ 356 (1990), eq. (16)
                v_halo_sqr = -self.halo_central_pot * r3/(1 + r3)**2
                    
            # total rotation curve
            self._mesh_v_rot = np.zeros(self._mesh_r.shape)

            for i in range(r.size):
                try:
                    v_rot_sqr = v_gas_sqr[i] + v_press_sqr[i] + v_halo_sqr[:,i]
                    if v_rot_sqr.min() > 0:
                        self._mesh_v_rot[:,i] = v_rot_sqr[:]**(1/2)
                    else:
                        raise ValueError
                except ValueError:
                    print("\nValueError: negative squared velocity - adjust parameters!")
                    break
                    
            print(f"\nMaximum rotation velocity = {1e-3*np.max(self._mesh_v_rot[0,:]):.1f} km/s")
            
        else:
            print("Error: unknown disk type")
    
    
    # print data to file
    def save(self, file, scaled=True):
                
        if scaled:
            
            # output relative to length scales and central density
            if self.disk == 'double exponential':
                data = np.array((self._mesh_r.flatten(), 
                                 self._mesh_z.flatten(), 
                                 self._mesh_rho.flatten() / self.central_density))
            else:               
                data = np.array((self._mesh_r.flatten(), 
                                 self._mesh_z.flatten(), 
                                 self._mesh_rho.flatten() / self.central_density,
                                 self._mesh_v_rot.flatten()))          
            
        else:
            
            # output in SI units
            if self.disk == 'double exponential':
                data = np.array((self._mesh_r.flatten() * self.scale_length/kpc, 
                                 self._mesh_z.flatten() * self.scale_height/kpc, 
                                 self._mesh_rho.flatten()))
            else:
                data = np.array((self._mesh_r.flatten() * self.scale_length/kpc, 
                                 self._mesh_z.flatten() * self.scale_height/kpc, 
                                 self._mesh_rho.flatten(),
                                 self._mesh_v_rot.flatten()))
            
        np.savetxt(file, data.transpose(), fmt='%.10e')
