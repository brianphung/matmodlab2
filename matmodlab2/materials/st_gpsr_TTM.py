from numpy import zeros, ix_, sqrt
import numpy as np

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import dyad, deviatoric_part, double_dot, magnitude, matrix_rep, array_rep

TOLER = 1e-8
ROOT3, ROOT2 = sqrt(3.0), sqrt(2.0)
ROOT23 = np.sqrt(2.0 / 3.0)
ONEHALF = 0.5

class ST_GPSR_TTM(Material):
    name = "st-gpsr-ttm"

    def __init__(self, **parameters):
        """Set up the Plastic material"""
        param_names = ["E", "Nu", "Y0", "Y1", "m", "B"]
        self.params = {}
        for (i, name) in enumerate(param_names):
            self.params[name] = parameters.pop(name, 0.0)
        if parameters:
            unused = ", ".join(parameters.keys())
            logger.warning("Unused parameters: {0}".format(unused))

        self.params["H"] = 1.0

        # Check inputs
        E = self.params["E"]
        Nu = self.params["Nu"]
        Y0 = self.params["Y0"]
        B = self.params["B"]
        H = self.params["H"]
        errors = 0
        if E <= 0.0:
            errors += 1
            logger.error("Young's modulus E must be positive")
        if Nu > 0.5:
            errors += 1
            logger.error("Poisson's ratio > .5")
        if Nu < -1.0:
            errors += 1
            logger.error("Poisson's ratio < -1.")
        if Nu < 0.0:
            logger.warning("#---- WARNING: negative Poisson's ratio")
        if Y0 < 0:
            errors += 1
            logger.error("Yield strength must be positive")
        if Y0 < 1e-12:
            # zero strength -> assume the user wants elasticity
            logger.warning("Zero strength detected, setting it to a larg number")
            self.params["Y0"] = 1e60
        if callable(B) == False:
            errors += 1
            logger.error("No mapping matrix as function.")
        if errors:
            raise ValueError("stopping due to previous errors")

        # At this point, the parameters have been checked.  Now request
        # allocation of solution dependent variables.  The only variable
        # is the equivalent plastic strain

        # Register State Variables
        # self.sdv_names = [
        #     "EP_X",
        #     "EP_Y",
        #     "EP_Z",
        #     "Y",
        #     "EQPS"
        # ]

        self.sdv_names = [
            "EP_XX",
            "EP_YY",
            "EP_ZZ",
            "EP_XY",
            "EP_YZ",
            "EP_XZ",
            "EQPS",
            "Y"
        ]
        self.num_sdv = len(self.sdv_names)

    def sdvini(self, statev):
        Y0 = self.params["Y0"]
        #return np.array([ 0.0, 0.0, 0.0, Y0, 0 ])
        return np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Y0 ])

    # def apply_some_mapping_to_stress_vector(self, A_tensor, S_vector):
    #     S_tensor = matrix_rep(S_vector,0)
    #     rot_tensor = np.tensordot(A_tensor, S_tensor)
    #     return array_rep(rot_tensor, (6,))
    
    # def dGdS(self, pk2_stress):
    #     sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6 = array_rep(pk2_stress, (6,))
    #     denom = sqrt(6*sigma_4**2 + 6*sigma_5**2 + 6*sigma_6**2 + (sigma_1 - sigma_2)**2 + (sigma_1 - sigma_3)**2 + (sigma_2 - sigma_3)**2)
    #     dGds1 = ROOT2*(2*sigma_1 - sigma_2 - sigma_3)/(2*denom)
    #     dGds2 = ROOT2*(-sigma_1 + 2*sigma_2 - sigma_3)/(2*denom)
    #     dGds3 = ROOT2*(-sigma_1 - sigma_2 + 2*sigma_3)/(2*denom)
    #     dGds4 = 3*ROOT2*sigma_4/denom
    #     dGds5 = 3*ROOT2*sigma_5/denom
    #     dGds6 = 3*ROOT2*sigma_6/denom
    #     dGdS_array = np.array([dGds1, dGds2, dGds3, dGds4, dGds5, dGds6])
    #     return matrix_rep(dGdS_array, 0)
    
    def dGdSMandell(self, mandell_stress_vector):
        sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6 = mandell_stress_vector
        sigma_4 = sigma_4/ROOT2
        sigma_5 = sigma_5/ROOT2
        sigma_6 = sigma_6/ROOT2
        denom = sqrt(6*sigma_4**2 + 6*sigma_5**2 + 6*sigma_6**2 + (sigma_1 - sigma_2)**2 + (sigma_1 - sigma_3)**2 + (sigma_2 - sigma_3)**2)
        dGds1 = ROOT2*(2*sigma_1 - sigma_2 - sigma_3)/(2*denom)
        dGds2 = ROOT2*(-sigma_1 + 2*sigma_2 - sigma_3)/(2*denom)
        dGds3 = ROOT2*(-sigma_1 - sigma_2 + 2*sigma_3)/(2*denom)
        dGds4 = 3*ROOT2*sigma_4/denom
        dGds5 = 3*ROOT2*sigma_5/denom
        dGds6 = 3*ROOT2*sigma_6/denom
        dGdS_array = np.array([dGds1, dGds2, dGds3, dGds4, dGds5, dGds6])
        dGdS_array[3:] = dGdS_array[3:]/ROOT2
        return dGdS_array
    
    # def dGdSigma(self, principal_stress_vector):
    #     sigma_1, sigma_2, sigma_3 = principal_stress_vector
    #     denom = (2*sqrt((sigma_1 - sigma_2)**2 + (sigma_1 - sigma_3)**2 + (sigma_2 - sigma_3)**2))
    #     dGdSigma1 = ROOT2*(2*sigma_1 - sigma_2 - sigma_3)/denom
    #     dGdSigma2 = ROOT2*(-sigma_1 + 2*sigma_2 - sigma_3)/denom
    #     dGdSigma3 = ROOT2*(-sigma_1 - sigma_2 + 2*sigma_3)/denom
    #     return np.array([dGdSigma1, dGdSigma2, dGdSigma3])
    
    # def vm_princ_stress(self, stress_vec):
    #     Y = self.params["Y0"]
    #     #P_iso = (1./Y**2)*np.array([[1, -ONEHALF, -ONEHALF], [ -ONEHALF, 1, -ONEHALF ], [ -ONEHALF, -ONEHALF, 1 ] ])
    #     #func = stress_vec @ P_iso @ stress_vec - 1
    #     sigma_1, sigma_2, sigma_3 = stress_vec
    #     vm = np.sqrt( (  (sigma_1 - sigma_2)**2 + (sigma_2 - sigma_3)**2 + (sigma_3 - sigma_1)**2  )/2 )
    #     func = vm - Y
    #     return func
    
    def vm_stress_mandell(self, mandel_stress_vec, eqps):
        Y = self.params["Y0"]
        H = self.params["H"]
        # MML stress comes in the following order: 11, 22, 33, 12, 23, 13
        sigma_11, sigma_22, sigma_33, sigma_12, sigma_23, sigma_13 = mandel_stress_vec
        sigma_12 = sigma_12/ROOT2
        sigma_23 = sigma_23/ROOT2
        sigma_13 = sigma_13/ROOT2
        internal = (sigma_11 - sigma_22)**2 + (sigma_22 - sigma_33)**2 + (sigma_33 - sigma_11)**2 + 6*(sigma_23**2 + sigma_13**2 + sigma_12**2)
        vm = np.sqrt(0.5*internal)
        Y_K = Y + H*eqps
        return vm - Y_K

    def eval(self, time, dtime, temp, dtemp, F0, F, stran_V, d_V, stress_V, X, **kwargs):
        # First, we'll convert everything into mandel notation
        # MML stress comes in the following order: 11, 22, 33, 12, 23, 13
        # MML strain comes in as engineering strain
        stress = np.array([ stress_V[i] for i in [0, 1, 2, 4, 5, 3] ])
        stress[3:] = ROOT2*stress[3:]
        strain = np.array([ stran_V[i] for i in [0, 1, 2, 4, 5, 3] ])
        strain[3:] = ROOT2/2.*strain[3:]
        delta_strain = np.array([ d_V[i] for i in [0, 1, 2, 4, 5, 3] ])
        #print('delta_strain before', delta_strain)
        delta_strain[3:] = ROOT2/2.*delta_strain[3:]
        #print('delta_strain after', delta_strain)
        e_p = np.array([X[ss] for ss in range(0,6)]) # e_p will also be in voigt notation
        e_p = ROOT2/2.*e_p[3:]
        # From here, we're working in a Mandell basis

        Y = self.params["Y0"]
        E = self.params["E"]
        Nu = self.params["Nu"]
        H = self.params["H"]

        # Get the bulk, shear, and Lame constants
        K = E / 3.0 / (1.0 - 2.0 * Nu)
        G = E / 2.0 / (1.0 + Nu)

        K3 = 3.0 * K
        G2 = 2.0 * G
        Lam = (K3 - G2) / 3.0
        
        # elastic stiffness, in Mandell
        C = np.zeros((6, 6))
        C[np.ix_(range(3), range(3))] = Lam
        C[range(3), range(3)] += G2
        C[range(3, 6), range(3, 6)] = 2*G
        C[3:6, 0:3] *= ROOT2
        C[0:3, 3:6] *= ROOT2



        #print(G)


        # Evaluate predicted stress in the spatial basis
        d_strain = delta_strain * dtime # Strain rate
        delta_T = np.dot(C, d_strain) #Mandell stress vector delta

        # Calculate a trial stress
        trial_T = stress + delta_T
        # Set the plastic strains
        e_p = np.array([0,0,0,0,0,0])

        # Transform the stress to a fictious isotropic space
        trial_eqps = X[6]
        A_in = self.params["B"](trial_eqps)
        if A_in.shape == (3,3):
            new_A = np.zeros((6,6))
            new_A[0:3, 0:3] = A_in
            new_A[3:, 3:] = np.eye(3)#*ROOT2 # Like the stuffness matrix, the shear comps are multiplied by ROOT2
            A_in = new_A
        
        trial_Sigma_f = np.dot(A_in, trial_T)
        

        #A_E = np.linalg.tensorinv(C) @ A_in @ C

        #print('&&&&&&&&&&&&&&&&&&&&&&&& BEGIN PLASTIC ITERATIONS &&&&&&&&&&&&&&&&&&&&&&&&')
        #print('PRE TRIAL STRESS: ', trial_Sigma_f)

        if self.vm_stress_mandell(trial_Sigma_f, trial_eqps) <= 0:
            pass
        else:
            for j in range(1000):
                #print(f'---------- BEGIN J {j} ----------')
                #print(f'Pre trial: {trial_Sigma_f}')
                trial_Sigma_f = trial_Sigma_f - np.dot(C, e_p) # We cut the strain because elastic isotropy and principal stresses
                #print(f'New trial: {trial_Sigma_f}')
                #print(f'from E_p:', e_p)

                # Calculate the yield function
                yield_F = self.vm_stress_mandell(trial_Sigma_f, trial_eqps)
                #print('Yield F', yield_F, 'Von Mises Stress:', yield_F + Y)
                flow_direction = self.dGdSMandell(trial_Sigma_f)
                #print('Flow dir: ', flow_direction)

                v_C_r = flow_direction @ C @ flow_direction
                #print('v_c_r: ', v_C_r)

                dGamma = yield_F / v_C_r
                #print('dGamma: ', dGamma)
                R_i = flow_direction 
                #R_i[3:] = R_i[3:]/4.
                delta_E_p = dGamma*R_i

                # delta eqps
                #print('H_R_i', R_i)
                #print('full', np.full((6), H))
                delta_eqps = dGamma * H
                #print(delta_eqps)
                trial_eqps += delta_eqps

                #print('Delta E_p: ', dGamma)
                e_p = e_p + delta_E_p
                #print('New E_p: ', e_p)
                if dGamma <= TOLER:
                    break
                if (yield_F) < 0:
                    raise RuntimeError('Shouldnt he here')
            else:
                raise RuntimeError("Newton iterations failed to converge")
            
        #print('---------------- END PLASTIC ITERATIONS ----------------')
    
        # transform the stress back to real space
        A_new = self.params["B"](trial_eqps)
        if A_new.shape == (3,3):
            new_A = np.zeros((6,6))
            new_A[0:3, 0:3] = A_new
            new_A[3:, 3:] = np.eye(3)#*ROOT2 # Like the stuffness matrix, the shear comps are multiplied by ROOT2
            A_new = new_A
        B_new = np.linalg.inv(A_new)
        final_mandel_stress = np.dot(B_new, trial_Sigma_f)
        # Transform mandel stress back to voigt
        stress = final_mandel_stress
        #print(stress)
        stress[3:] = stress[3:]/ROOT2
        #print(stress)
        stress = np.array([ stress[i] for i in [0, 1, 2, 5, 3, 4] ])

        # Update plastic strain
        e_p = np.array([ e_p[i] for i in [0, 1, 2, 5, 3, 4] ])
        X[:3] = e_p[:3]
        X[3:6] = e_p[3:6]/ROOT2*2


        X[6] = trial_eqps#+= ROOT2/3.*np.sqrt( (Epp[0] - Epp[1])**2 + (Epp[1] - Epp[2])**2 + (Epp[2] - Epp[0])**2 )

        return stress, X, None
    """
    def eval_VOIGHT(self, time, dtime, temp, dtemp, F0, F, stran, d, stress, X, **kwargs):

        Y = self.params["Y0"]
        E = self.params["E"]
        Nu = self.params["Nu"]

        # EPS
        e_p = np.array([X[ss] for ss in range(2,8)])
        #print(e_p)

        # Get the bulk, shear, and Lame constants
        K = E / 3.0 / (1.0 - 2.0 * Nu)
        G = E / 2.0 / (1.0 + Nu)

        K3 = 3.0 * K
        G2 = 2.0 * G
        # G3 = 3.0 * G
        Lam = (K3 - G2) / 3.0

        # elastic stiffness
        C = np.zeros((6, 6))
        C[np.ix_(range(3), range(3))] = Lam
        C[range(3), range(3)] += G2
        C[range(3, 6), range(3, 6)] = G



        # Define stress space transformation tensor
        B_in = self.params["B"]
        B_sigma = np.einsum('ik,jl->ijkl', np.eye(3), B_in)
        A_sigma = np.linalg.tensorinv(B_sigma)
        A_elastic = np.einsum('ik,jl->ijkl', np.eye(3), np.eye(3))

        Delta_E = d*dtime
        delta_stress = np.dot(C, Delta_E)
        stress_init = stress + delta_stress
        real_stress_tensor = matrix_rep(stress_init,0)

        fict_stress = np.tensordot(A_sigma, real_stress_tensor)
        fict_stress_array = array_rep(fict_stress, (6,))
    
        # Iterate isotropic constitutive law

        for j in range(20):

            if j == 0:
                array_S_trial = fict_stress_array.copy()
            
            temp = e_p.copy()
            temp[3:] *= 2
            Delta_Ep = matrix_rep(temp, 0)
            print(temp)

            # Compute stresses
            array_S_trial = array_S_trial - np.dot(C, temp)
            S_trial = matrix_rep(array_S_trial, 0)
            # Calculate the yield function
            eq_stress = self.eqv(array_S_trial)
            
            if eq_stress <= Y:
                X[8] = eq_stress
                # Elastic, transform the stress back and return
                real_S_trial = np.tensordot(B_sigma, S_trial)
                array_real_S_trial = array_rep(real_S_trial, (6,))
                # Set the plastic strain
                temp = array_rep(Delta_Ep, (6,))
                temp[3:] /= 2.
                X[2:8] = temp
                return array_real_S_trial, X, None
            
            yield_F = eq_stress - Y
            flow_direction = self.dGdS(S_trial)
            r_array = array_rep(flow_direction, (6,))
            v_C_r = np.dot( r_array, np.dot(C, r_array) )

            dGamma = yield_F / v_C_r
            R_ij = np.tensordot(A_sigma, np.tensordot(flow_direction, A_sigma))
            Delta_Ep = dGamma*R_ij
            temp = array_rep(Delta_Ep, (6,))
            temp[3:] /= 2
            e_p = temp
            
            # if (abs(dGamma) + 1.0) < TOLER + 1.0:
            #     break

        # else:
        #     raise RuntimeError("Newton iterations failed to converge")
        
        real_S_trial = np.tensordot(B_sigma, S_trial)
        array_real_S_trial = array_rep(real_S_trial, (6,))
        X[2:8] = e_p
        
        
        return array_real_S_trial, X, None

    def eqv(self, sig):
        # Returns sqrt(3 * rootj2) = sig_eqv = q
        s = sig - sig[:3].sum() / 3.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return 1.0 / ROOT23 * np.sqrt(np.dot(s[:3], s[:3]) + 2 * np.dot(s[3:], s[3:]))
    """
    """
    def eval_radian_return(self, time, dtime, temp, dtemp, F0, F, stran, d, stress, X, **kwargs):

        #  material properties
        Y = self.params["Y0"]
        E = self.params["E"]
        Nu = self.params["Nu"]

        B_in = self.params["B"]

        # Input parameter is yield in tension -> convert to yield in shear
        k = Y / ROOT3

        # Get the bulk, shear, and Lame constants
        K = E / 3.0 / (1.0 - 2.0 * Nu)
        G = E / 2.0 / (1.0 + Nu)

        K3 = 3.0 * K
        G2 = 2.0 * G
        # G3 = 3.0 * G
        Lam = (K3 - G2) / 3.0

        # elastic stiffness
        C = np.zeros((6, 6))
        C[np.ix_(range(3), range(3))] = Lam
        C[range(3), range(3)] += G2
        C[range(3, 6), range(3, 6)] = G

        # Define stress space transformation tensor
        B_sigma = np.einsum('ik,jl->ijkl', np.eye(3), B_in)
        A_sigma = np.linalg.tensorinv(B_sigma)

        # Trial stress
        de = d * dtime
        T_real = stress + np.dot(C, de)
        S_real = deviatoric_part(T_real)

        # convert stress into fictious stress space
        # We need to convert T into matrix from so we can apply the transformation
        T_real_mat = matrix_rep(T_real,0)
        # Transform the stress
        T_fict_mat = np.tensordot(A_sigma, T_real_mat)
        #print(B, '\n', A_sigma)
        # Transform back to vector rep
        T = array_rep(T_fict_mat, (6,)) # T is now a fictitious stress tensor
        print('Pre T real and T fictious vectors:\n', T_real, '\n', T, '\n')
        #raise Exception('Nope')
        # check yield
        S = deviatoric_part(T)
        RTJ2 = magnitude(S) / ROOT2
        f = RTJ2 - k

        if f <= TOLER:
            print('This step found elastic:')
            print('Deviatoric Stress', S)
            print('RTJ2', RTJ2, 'k', k, 'f', f)
            # Elastic loading, return what we have computed
            return T_real, X, C

        # Calculate the flow direction, projection direction
        # For the mapping, we need to ajudst the direction so we hit the correct location on the mapped yield surface

        M = self.apply_some_mapping_to_stress_vector(A_sigma, S_real / ROOT2 / RTJ2)
        N = self.apply_some_mapping_to_stress_vector(A_sigma, S_real / ROOT2 / RTJ2)
        A = 2 * G * M

        # Newton iterations to find Gamma
        Gamma = 0
        Ttrial = T.copy()
        for i in range(20):
            print(f'------------------------- Iter {i} -------------------------')
            print(f'Trail Stress:\n {Ttrial}')
            # Update all quantities
            dGamma = f * ROOT2 / double_dot(N, A)
            print(f'dGamma = {dGamma}')
            Gamma += dGamma

            T = Ttrial - Gamma * A
            S = deviatoric_part(T)
            RTJ2 = magnitude(S) / ROOT2
            f = RTJ2 - k
            print('Gamma: ', Gamma)
            print('Mag S', magnitude(S))
            print('RTJ2', RTJ2, 'k', k)
            print('After Gamma f', f)
            print('After Gamma Stress\n', T)
            print('A tensor*Gamma\n', Gamma*A)

            # Calculate the flow direction, projection direction
            T_real = self.apply_some_mapping_to_stress_vector(B_sigma, T)
            S_real = deviatoric_part(T_real)
            M = self.apply_some_mapping_to_stress_vector(A_sigma, S_real / ROOT2 / RTJ2)
            N = self.apply_some_mapping_to_stress_vector(A_sigma, S_real / ROOT2 / RTJ2)
            A = 2 * G * M
            Q = 2 * G * N

            if (abs(dGamma) + 1.0) < TOLER + 1.0:
                break

        else:
            raise RuntimeError("Newton iterations failed to converge")

        # Elastic strain rate and equivalent plastic strain
        # dT = T - stress
        # dep = Gamma * M
        # dee = de - dep
        deqp = ROOT2 / ROOT3 * Gamma

        # Transform stress back to the real world
        
        T_fict_mat = matrix_rep(T,0)
        T_real_mat = np.tensordot(B_sigma, T_fict_mat)
        T_real = array_rep(T_real_mat, (6,))

        print('Post T real and T fictious vectors:\n', T_real, '\n', T, '\n')

        S_real = deviatoric_part(T)
        RTJ2 = magnitude(S) / ROOT2

        # The D matrix will need to use the real stress tensor rather than the fictious one
        # Calculate the flow direction, projection direction
        M = S_real / ROOT2 / RTJ2
        N = S_real / ROOT2 / RTJ2
        A = 2 * G * M
        Q = 2 * G * N
        # Elastic stiffness
        D = C - 1 / double_dot(N, A) * dyad(Q, A)

        # Equivalent plastic strain
        X[0] += deqp

        return T_real, X, D
        """

    """ 
        # Fuller Method
        # Update all quantities
        dfdy = -1.0 / ROOT3
        dydG = ROOT2 / ROOT3 * Y1
        hy = ROOT2 / ROOT3 * Y1
        if Y1 > 1e-8 and eqps > 1e-8:
            hy *= m * ((self.Y(Y0, Y1, m, eqps) - Y0) / Y1) ** ((m - 1.0) / m)
            dydG *= m * eqps ** (m - 1.0)

        dGamma = f * ROOT2 / (double_dot(N, A) - dfdy * dydG)
        Gamma += dGamma

        fict_T = Ttrial - Gamma * A
        S = deviatoric_part(fict_T)
        RTJ2 = magnitude(S) / ROOT2
        #eqps += ROOT2 / ROOT3 * dGamma

        f = RTJ2 - self.Y(Y0, Y1, m, eqps) / ROOT3

        # Calculate the flow direction, projection direction
        M = S / ROOT2 / RTJ2
        N = S / ROOT2 / RTJ2
        A = 2 * G * M
        Q = 2 * G * N

        if abs(dGamma + 1.0) < TOLER + 1.0:
            break
        

        else:
            raise RuntimeError("Newton iterations failed to converge")

        # Elastic strain rate and equivalent plastic strain
        # dT = T - stress
        # dep = Gamma * M
        # dee = de - dep
        deqp = 0# ROOT2 / ROOT3 * Gamma

        # Elastic stiffness
        H = -2.0 * dfdy * hy / ROOT2
        D = C - 1 / (double_dot(N, A) + H) * dyad(Q, A)

        # Equivalent plastic strain
        X[0] += deqp
        # print X[0]
        # print eqps
        # assert abs(X[0] - eqps) + 1 < 1.1e-5, 'Bad plastic strain integration'

        # Convert fict_T back to 3x3 rep
        S_ten = matrix_rep(fict_T, 0)
        inverse_A = np.linalg.tensorinv(A_tensor)
        S_real_p = np.tensordot(inverse_A,S_ten)
        # Apply eigen rotations
        # Convert S_real back to vector
        #S_real = np.dot(np.transpose(Q_princ), np.dot(S_real_p, Q_princ))
        T_real = array_rep(S_real_p, (6,))

        return T_real, X, D
        
    
        def eval_principal(self, time, dtime, temp, dtemp, F0, F, stran_V, d_V, stress_V, X, **kwargs):
            
            E = self.params["E"]
            Nu = self.params["Nu"]

            # Get the bulk, shear, and Lame constants
            K = E / 3.0 / (1.0 - 2.0 * Nu)
            G = E / 2.0 / (1.0 + Nu)

            K3 = 3.0 * K
            G2 = 2.0 * G
            Lam = (K3 - G2) / 3.0
            
            # elastic stiffness, isotropic
            C = np.zeros((6, 6))
            C[np.ix_(range(3), range(3))] = Lam
            C[range(3), range(3)] += G2
            C[range(3, 6), range(3, 6)] = G

            # Evaluate predicted stress in the spatial basis
            d_strain = d_V * dtime
            delta_T = np.dot(C, d_strain)

            # Calculate a trial stress
            trial_T = stress_V + delta_T

            # Work with a matrix stress instead

            trial_S = matrix_rep(trial_T, 0)
            #print(f'PreStep Trial Stress: {trial_T}')
            # Work in principal stress space
            trial_Sigma_r, Q = np.linalg.eig(trial_S)

            # Transform stresses to ficitious isotropic space:
            A_in = self.params["B"]
            B_in = np.linalg.inv(A_in)

            A_E = A_in # This is incorrect when we consider elastic anisotropy

            trial_Sigma_f = np.dot(A_in, trial_Sigma_r)
            
            YF_0 = None

            # Grab the plastic principal strains
            e_p = np.array([0,0,0])

            D = C[0:3, 0:3]
            A_E = np.linalg.inv(D) @ A_in @ D
            #print('D and A_E', D, A_E)
            if self.vm_princ_stress(trial_Sigma_f) <= 0:
                #print(f'Elastic for step {time}')
                pass
            else:
                for j in range(1000):
                    # Compute stresses
                    trial_Sigma_f = trial_Sigma_f - np.dot(D, e_p) # We cut the strain because elastic isotropy and principal stresses

                    # Calculate the yield function
                    yield_F = self.vm_princ_stress(trial_Sigma_f)
                    if YF_0 is None:
                        YF_0 = abs(yield_F)
                    
                    #print(f'Yield F: {yield_F}, trial_stress: {trial_Sigma_f}')


                    #print(f'Plastic iter for step {time}, iter j {j}')
                    # Calculate the new plastic strain
                    flow_direction = self.dGdSigma(trial_Sigma_f)
                    #A_S_flow = A_in @ flow_direction
                    #A_e_flow = A_E @ flow_direction
                    #flow_trans = A_E @ flow_direction @ A_in
                    v_C_r = flow_direction @ D @ flow_direction
                    #D_f = A_E @ D @ A_in
                    #v_C_r = flow_trans @ D_f @ flow_trans
                    dGamma = yield_F / v_C_r
                    R_i = A_E @ B_in @ flow_direction #A_E @ (flow_direction @ A_in)
                    R_i = flow_direction
                    delta_E_p = dGamma * R_i
                    e_p = e_p + delta_E_p
                    #print('flow_dir', flow_direction, 'gamma', dGamma)
                    if dGamma <= TOLER:
                        break
                    if (yield_F) < 0:
                        raise RuntimeError('Shouldnt he here')
                else:
                    raise RuntimeError("Newton iterations failed to converge")
            
            # We've got transformed principal stress, let's transform it back
            trial_Sigma_r = np.dot(B_in, trial_Sigma_f)
            # Then, diagonalize the stresses and reverse the eigen problem
            trial_S = Q @ np.diag(trial_Sigma_r) @ np.transpose(Q)
            # Now, convert to vector
            trial_T = array_rep(trial_S, (6,))

            # Update plastic strain
            #print(e_p)
            X[:3] = e_p
            X[4] += ROOT2/3.*np.sqrt( (e_p[0] - e_p[1])**2 + (e_p[1] - e_p[2])**2 + (e_p[2] - e_p[0])**2 )


            return trial_T, X, None
        """