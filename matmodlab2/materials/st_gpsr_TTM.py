from numpy import zeros, ix_, sqrt
import numpy as np

import auto_diff

from ..core.logio import logger
from ..core.material import Material
from ..core.tensor import dyad, deviatoric_part, double_dot, magnitude, matrix_rep, array_rep

from scipy import optimize

TOLER = 1e-8
ROOT3, ROOT2 = sqrt(3.0), sqrt(2.0)
ROOT23 = np.sqrt(2.0 / 3.0)
ONEHALF = 0.5

class ST_GPSR_TTM(Material):
    name = "st-gpsr-ttm"

    def __init__(self, **parameters):
        """Set up the Plastic material"""
        param_names = ["E", "Nu", "Y0", "H", "A_mapping", "B_mapping"]
        self.params = {}
        for (i, name) in enumerate(param_names):
            self.params[name] = parameters.pop(name, 0.0)
        if parameters:
            unused = ", ".join(parameters.keys())
            logger.warning("Unused parameters: {0}".format(unused))

        self.P_vm = np.array([[1, -0.5, -0.5, 0, 0, 0],
                     [-0.5, 1, -0.5, 0, 0, 0],
                     [-0.5, -0.5, 1, 0, 0, 0],
                     [0, 0, 0, 3, 0, 0],
                     [0, 0, 0, 0, 3, 0],
                     [0, 0, 0, 0, 0, 3]])

        # Check inputs
        E = self.params["E"]
        Nu = self.params["Nu"]
        Y0 = self.params["Y0"]
        A = self.params["A_mapping"]

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
        if callable(A) == False:
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
            "FICT_EP_XX",  # 0
            "FICT_EP_YY",  # 1
            "FICT_EP_ZZ",  # 2
            "FICT_EP_XY",  # 3
            "FICT_EP_YZ",  # 4
            "FICT_EP_XZ",  # 5
            "FICT_EQPS",   # 6
            "REAL_EP_XX",  # 7
            "REAL_EP_YY",  # 8
            "REAL_EP_ZZ",  # 9
            "REAL_EP_XY",  # 10
            "REAL_EP_YZ",  # 11
            "REAL_EP_XZ",  # 12
            "REAL_EQPS",   # 13
            "Y",           # 14
            "S.VM",        # 15
            # SDVS ABOVE HERE ARE HARD CODED, do not modify SDVs above this line
            "FICT_SXX",    
            "FICT_SYY",
            "FICT_SZZ",
            "FICT_SXY",
            "FICT_SYZ",
            "FICT_SXZ", 
            "TRIAL_STRESS_POST_TRANS_XX",
            "TRIAL_STRESS_POST_TRANS_YY",
            "TRIAL_STRESS_POST_TRANS_ZZ",
            "TRIAL_STRESS_POST_TRANS_XY",
            "TRIAL_STRESS_POST_TRANS_YZ",
            "TRIAL_STRESS_POST_TRANS_XZ",
            "MML_STRESS_GUESS_XX",    
            "MML_STRESS_GUESS_YY", 
            "MML_STRESS_GUESS_ZZ", 
            "MML_STRESS_GUESS_XY",    
            "MML_STRESS_GUESS_YZ", 
            "MML_STRESS_GUESS_XZ", 
            "TRIAL_STRESS_PRE_TRANS_XX",   
            "TRIAL_STRESS_PRE_TRANS_YY",   
            "TRIAL_STRESS_PRE_TRANS_ZZ",   
            "TRIAL_STRESS_PRE_TRANS_XY",   
            "TRIAL_STRESS_PRE_TRANS_YZ",   
            "TRIAL_STRESS_PRE_TRANS_XZ",   
            "CONV_STRESS_ISO_XX", 
            "CONV_STRESS_ISO_YY", 
            "CONV_STRESS_ISO_ZZ", 
            "CONV_STRESS_ISO_XY", 
            "CONV_STRESS_ISO_YZ", 
            "CONV_STRESS_ISO_XZ", 
            "FICT_EQPS_INPUT", 
            "DELTA_EXX", 
            "DELTA_EYY", 
            "DELTA_EZZ", 
            "DELTA_EXY", 
            "DELTA_EYZ", 
            "DELTA_EXZ", 
            'DELTA_SXX',
            'DELTA_SYY',
            'DELTA_SZZ',
            'DELTA_SXY',
            'DELTA_SYZ',
            'DELTA_SXZ'
        ]
        self.SDV = {}
        for i, names in enumerate(self.sdv_names):
            self.SDV[names] = i
        self.num_sdv = len(self.sdv_names)
        #print(self.num_sdv)


        self.full_sdv_storage = []
        self.cutting_plane_history = []

    def sdvini(self, statev):
        #return np.array([ 0.0, 0.0, 0.0, Y0, 0 ])
        values = np.array([ 0.0 for i in range(self.num_sdv)])
        # Hard coded Y0
        values[14] = self.params["Y0"]
        values[7:13] = 1e-32
        return values

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
        raise NotImplementedError
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
    
    def ddGddSMandell(self, mandell_stress_vector):
        raise NotImplementedError
        sigma_1, sigma_2, sigma_3, sigma_4, sigma_5, sigma_6 = mandell_stress_vector
        sigma_4 = sigma_4/ROOT2
        sigma_5 = sigma_5/ROOT2
        sigma_6 = sigma_6/ROOT2
        axial_denom = (4*(sigma_1**2 - sigma_1*sigma_2 - sigma_1*sigma_3 + sigma_2**2 - sigma_2*sigma_3 + sigma_3**2 + 3*sigma_4**2 + 3*sigma_5**2 + 3*sigma_6**2)**(3/2))
        shear_denom = (6*sigma_4**2 + 6*sigma_5**2 + 6*sigma_6**2 + (sigma_1 - sigma_2)**2 + (sigma_1 - sigma_3)**2 + (sigma_2 - sigma_3)**2)**(3/2)
        dGds1 = 3*(sigma_2**2 - 2*sigma_2*sigma_3 + sigma_3**2 + 4*sigma_4**2 + 4*sigma_5**2 + 4*sigma_6**2)/axial_denom
        dGds2 = 3*(sigma_1**2 - 2*sigma_1*sigma_3 + sigma_3**2 + 4*sigma_4**2 + 4*sigma_5**2 + 4*sigma_6**2)/axial_denom
        dGds3 = 3*(sigma_1**2 - 2*sigma_1*sigma_2 + sigma_2**2 + 4*sigma_4**2 + 4*sigma_5**2 + 4*sigma_6**2)/axial_denom
        dGds4 = 3*sqrt(2)*(6*sigma_5**2 + 6*sigma_6**2 + (sigma_1 - sigma_2)**2 + (sigma_1 - sigma_3)**2 + (sigma_2 - sigma_3)**2)/shear_denom
        dGds5 = 3*sqrt(2)*(6*sigma_4**2 + 6*sigma_6**2 + (sigma_1 - sigma_2)**2 + (sigma_1 - sigma_3)**2 + (sigma_2 - sigma_3)**2)/shear_denom
        dGds6 = 3*sqrt(2)*(6*sigma_4**2 + 6*sigma_5**2 + (sigma_1 - sigma_2)**2 + (sigma_1 - sigma_3)**2 + (sigma_2 - sigma_3)**2)/shear_denom
        ddGddS_array = np.array([dGds1, dGds2, dGds3, dGds4, dGds5, dGds6])
        ddGddS_array[3:] = ddGddS_array[3:]/ROOT2
        return ddGddS_array
    
    def equivalent_stress(self, mandel_stress_vec):
        # MML stress comes in the following order: 11, 22, 33, 12, 23, 13
        raise NotImplementedError
        sigma_11, sigma_22, sigma_33, sigma_12, sigma_23, sigma_13 = mandel_stress_vec
        sigma_12 = sigma_12/ROOT2
        sigma_23 = sigma_23/ROOT2
        sigma_13 = sigma_13/ROOT2
        internal = (sigma_11 - sigma_22)**2 + (sigma_22 - sigma_33)**2 + (sigma_33 - sigma_11)**2 + 3*(sigma_23**2 + sigma_13**2 + sigma_12**2)
        vm = np.sqrt(0.5*internal)
        return vm
    
    def yield_function_mandell(self, mandel_stress_vec, e_p_strain_vec ):
        # if not np.isscalar(eqps):
        #     eqps = eqps[0][0]

        e_p_strain_vec = e_p_strain_vec.T
        eqps = ROOT2/ROOT3*np.sqrt(e_p_strain_vec @ np.transpose(e_p_strain_vec))
        #print('eqps in', eqps)
        A = self.params["A_mapping"](eqps) 
        #print(A)
        G = np.transpose(mandel_stress_vec) @ ( np.transpose(A) @ self.P_vm @ A ) @ mandel_stress_vec
        return G - 1

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
        e_p_real = np.array([X[ss] for ss in range(7,13)]) # e_p will also be in voigt notation
        e_p_real[3:] = ROOT2/2.*e_p_real[3:]

        E = self.params["E"]
        Nu = self.params["Nu"]

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

        # Calculate a trial stress
        delta_stress = np.dot(C, delta_strain*dtime)
        trial_T = stress + delta_stress

        delta_e_p = np.array([0,0,0,0,0,0], dtype=float)

        trial_real_eqps = X[13]

        trial_Sigma_f = trial_T

        if self.yield_function_mandell(trial_Sigma_f, e_p_real) <= 0:
            pass
        else:
            for j in range(1000):
                #print(f'iter {j}')
                #print(strain.shape, e_p_real.shape)
                trial_Sigma_f = trial_Sigma_f - np.dot(C, delta_e_p)

                # Calculate the yield function
                # yield_F = self.yield_function_mandell(trial_Sigma_f, trial_real_eqps)
                #print('Yield F', yield_F, 'Von Mises Stress:', yield_F + Y)
                x = np.array([trial_Sigma_f]).T
                y = np.array([e_p_real]).T
                #print('x,y autodiff', x, y)
                with auto_diff.AutoDiff(x, y) as (x,y):
                    f_eval = self.yield_function_mandell(x, y)
                    yield_F, (dGdSigma, H) = auto_diff.get_value_and_jacobians(f_eval)
                dGdSigma = np.transpose(dGdSigma[0])
                yield_F = yield_F[0][0]
                #print('dGdSigma', dGdSigma, dGdSigma.shape)
                H = np.transpose(H[0])
                #print('H', H)

                v_C_r = dGdSigma @ C @ dGdSigma + H @ dGdSigma
                #print('v_c_r: ', v_C_r)

                dGamma = yield_F / v_C_r
                #print('dGamma', dGamma)
                delta_e_p = dGamma* dGdSigma
                #print(delta_e_p)
                e_p_real = e_p_real + delta_e_p
                trial_real_eqps =  ROOT2/ROOT3*np.sqrt(e_p_real @ np.transpose(e_p_real))
                
                if abs(dGamma) <= TOLER:
                    break
                # if (yield_F) < 0:
                #     print(yield_F)
                #     raise RuntimeError('Shouldnt he here')
            else:
                raise RuntimeError("Newton iterations failed to converge")
        #print('new eqps', trial_real_eqps)
        #e_p = e_p + delta_e_p
        #trial_Sigma_f = trial_Sigma_f - np.dot(C, e_p) # We cut the strain because elastic isotropy and principal stresses
        X[self.SDV['CONV_STRESS_ISO_XX']] = trial_Sigma_f[0]
        X[self.SDV['CONV_STRESS_ISO_YY']] = trial_Sigma_f[1]
        X[self.SDV['CONV_STRESS_ISO_ZZ']] = trial_Sigma_f[2]
        X[self.SDV['CONV_STRESS_ISO_XY']] = trial_Sigma_f[5]
        X[self.SDV['CONV_STRESS_ISO_YZ']] = trial_Sigma_f[3]
        X[self.SDV['CONV_STRESS_ISO_XZ']] = trial_Sigma_f[4]
        """ Temporary hide """

        #trial_Sigma_f = np.dot(C, iso_strain - e_p_iso)
        X[self.SDV['FICT_SXX']] = trial_Sigma_f[0]
        X[self.SDV['FICT_SYY']] = trial_Sigma_f[1]
        X[self.SDV['FICT_SZZ']] = trial_Sigma_f[2]
        X[self.SDV['FICT_SXY']] = trial_Sigma_f[5]
        X[self.SDV['FICT_SYZ']] = trial_Sigma_f[3]
        X[self.SDV['FICT_SXZ']] = trial_Sigma_f[4]
        final_mandel_stress =  trial_Sigma_f

        real_eqps =  trial_real_eqps
        #X[15] = self.equivalent_stress(final_mandel_stress)
        # Transform mandel stress back to voigt
        stress = final_mandel_stress
        
        #print(stress)
        stress[3:] = stress[3:]/ROOT2
        #print(stress)
        stress = np.array([ stress[i] for i in [0, 1, 2, 5, 3, 4] ])
        
        # Update plastic strain
        e_p_real = np.array([ e_p_real[i] for i in [0, 1, 2, 5, 3, 4] ])
        X[7:10] = e_p_real[:3]
        X[10:13] = e_p_real[3:6]/ROOT2*2

        #X[6] = trial_iso_eqps #+= ROOT2/3.*np.sqrt( (Epp[0] - Epp[1])**2 + (Epp[1] - Epp[2])**2 + (Epp[2] - Epp[0])**2 )
        X[13] = real_eqps
        #X[14] = Y + trial_iso_eqps*H

        sdv_full_stack_stress = final_mandel_stress[[0, 1, 2, 5, 3, 4]]
        sdv_full = np.hstack([[time, dtime], sdv_full_stack_stress, X ])
        self.full_sdv_storage.append(sdv_full)

        #cutting_plane_history = np.array(cutting_plane_history)
        #self.cutting_plane_history.append(cutting_plane_history)


        #R_fict = self.dGdSMandell(trial_Sigma_f)
        
        #dFdalpha = H
        """
        consistent_tangent_stiffness_H = np.eye(6) + dGam_total * C *self.ddGddSMandell(trial_Sigma_f)

        term1 = consistent_tangent_stiffness_H @ R_fict
        term2 = R_fict @ consistent_tangent_stiffness_H
        ddsdde_numerator = np.einsum('i,j',term1, term2)
        
        term3 = - dFdalpha * H * np.linalg.norm(R_fict)
        term4 = R_fict @ consistent_tangent_stiffness_H @ R_fict
        ddsdde_demoninator = term3 + term4

        ddsdde = consistent_tangent_stiffness_H -  (  1./ddsdde_demoninator * ddsdde_numerator  )
        ddsdde = B_new @ ddsdde @ A_E_new
        ddsdde[3:6, 0:3] /= ROOT2
        ddsdde[0:3, 3:6] /= ROOT2
        """


        return stress, X, None #ddsdde
