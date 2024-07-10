import numpy as np
from numpy import array
import pandas as pd
from matmodlab2 import *
from sklearn.linear_model import LinearRegression
import copy
import matplotlib.pyplot as plt


# Planned parameters
E = 10e6
nu = .333
Y0 = 100
H = 10000
base_load = 0.01

# all_comps = (0.01, 0, 0)
# all_loads = 'ESS'
all_frames = 100


def align_pi_plane_with_axes_rot():
    """
    Returns a matrix that rotates the pi plane's normal to be the z axis
    i.e., a slice of pi plane becomes the xy plane after rotation
    """
    pi_vector = np.array([1, 1, 1]) / np.sqrt(3.)
    # wanted_vector = np.array([1, 0, 0])
    wanted_vector = np.array([0, 0, 1])
    wanted_vector = wanted_vector / np.linalg.norm(wanted_vector)
    added = (pi_vector + wanted_vector).reshape([-1, 1])
    # from Rodrigues' rotation formula, more info here: https://math.stackexchange.com/a/2672702
    rot_mat = 2 * (added @ added.T) / (added.T @ added) - np.eye(3)
    return rot_mat


def align_axes_with_pi_plane_rot():
    """
    Returns a matrix that undoes the align_pi_plane_with_axes_rot rotation
    """
    return np.linalg.inv(align_pi_plane_with_axes_rot())

number_of_eqps_values = 7

eqps_values = pd.DataFrame(np.array([ i*0.002 for i in range(number_of_eqps_values) ]).reshape(-1,1), columns=['EP_Equiv'])


eqps_values_np = np.array([ i*0.002 for i in range(number_of_eqps_values) ])
def run_generic_mps(mps1, loads='ESS', components=(0.02, 0, 0), frames=50):
    mps1.run_step(loads,  components, frames=frames)
    return mps1
def get_load_from_comp(comps):
    load_string = ''
    for value in comps:
        if abs(value) < 1e-5:
            load_string += 'S'
        else:
            load_string += 'E'
    return load_string 

# Run the included von mises model
pVM = {'E': E, 'Nu': nu, 'Y0': Y0, 'Y1': H, 'm': 1.0}

loading_conditions = []
# Rotate strain components along X-Y, 12 degree increment

angles = np.linspace(0,2.*np.pi, 20)
strain_pairs = [ (0,1), (0,2), (1,2) ]

for strain_pair in strain_pairs:
    for angle in angles:
        loading_mags = [ 0, 0, 0, 0, 0, 0]
        loading_mags[strain_pair[0]] = base_load*np.sin(angle)
        loading_mags[strain_pair[1]] = base_load*np.cos(angle)
        loading_mags = tuple(loading_mags)

        loading_conditions.append([get_load_from_comp(loading_mags), loading_mags, strain_pair])



explicit_formatted_data = []
implicit_formatted_data = [ [] for i in range(number_of_eqps_values) ]
for III, item in enumerate(loading_conditions):
    load = item[0]
    comp = item[1]

    strain_pair = item[2]

    mpsVM = MaterialPointSimulator('VM_Plastic')
    mpsVM.material = HardeningPlasticMaterial(**pVM)
    mpsVM = run_generic_mps(mpsVM, frames=all_frames, components=comp, loads=load)
    # mpsVM.plot('EP_Equiv', 'S.XX')
    # plt.show()
    # regularization_index = None
    # if 0 not in strain_pair:
    #     regularization_index = 0
    # elif 1 not in strain_pair:
    #     regularization_index = 1
    # elif 2 not in strain_pair:
    #     regularization_index = 2
    # else:
    #     raise Exception('What?')

    stress_history = mpsVM.df[['S.XX', 'S.YY', 'S.ZZ', 'EP_Equiv']]
    print(stress_history)
    stress_history = stress_history[stress_history['EP_Equiv'] > 0.0001]

    regressorSXX = LinearRegression()
    regressorSXX.fit(stress_history[['EP_Equiv']], stress_history[['S.XX']])

    regressorSYY = LinearRegression()
    regressorSYY.fit(stress_history[['EP_Equiv']], stress_history[['S.YY']])

    regressorSZZ = LinearRegression()
    regressorSZZ.fit(stress_history[['EP_Equiv']], stress_history[['S.ZZ']])

    SXX = regressorSXX.predict(eqps_values)
    SYY = regressorSYY.predict(eqps_values)
    SZZ = regressorSZZ.predict(eqps_values)

    stress_all = np.zeros((number_of_eqps_values, 3))
    stress_all[:, 0] = SXX[:,0]
    stress_all[:, 1] = SYY[:,0]
    stress_all[:, 2] = SZZ[:,0]
    print(stress_all)
    stress_all = stress_all @ align_axes_with_pi_plane_rot()

    # Lazily force S33 to 0
    stress_all[:,2] = 0
    # # Calculate this "all inclusive" "Lode" angle
    # lodes = np.arctan2(stress_all[:,1],stress_all[:,0])
    # lodes[lodes < 0] += 2*np.pi
    # sortdices = np.argsort(lodes)
    # stress_all[:,0] = stress_all[sortdices,0]
    # stress_all[:,1] = stress_all[sortdices,1]
    # stress_all[:,2] = stress_all[sortdices,2]

    # # print(stress_all)
    # # exit()

    # Unrotate
    stress_all = stress_all @ align_pi_plane_with_axes_rot()

    SXX = stress_all[:,0]
    SYY = stress_all[:,1]
    SZZ = stress_all[:,2]
    
    for i in range(number_of_eqps_values):
        implicit_formatted_data[i].append([SXX[i], SYY[i], SZZ[i], eqps_values_np[i]])
        explicit_formatted_data.append([SXX[i], SYY[i], SZZ[i], eqps_values_np[i]])
    if III != (number_of_eqps_values - 1):
        explicit_formatted_data.append([np.nan,np.nan,np.nan,np.nan])
implicit_formatted_data = np.array(implicit_formatted_data)
# We gotta sort all them points...
sort_set = implicit_formatted_data[i]
pi_sort_set = sort_set[:,0:3] @ align_axes_with_pi_plane_rot()
# Calculate this "all inclusive" "Lode" angle
lodes = np.arctan2(pi_sort_set[:,1],pi_sort_set[:,0])
lodes[lodes < 0] += 2*np.pi
sortdices = np.argsort(lodes)

# Woof, now we gotta sort all sub points in implicit and explicit:
for i in range(number_of_eqps_values):
    implicit_formatted_data[i] = implicit_formatted_data[i][sortdices]
# # Now we sort the outer layer of explicit data
# explicit_formatted_data = np.array(explicit_formatted_data)[sortdices,:]
# # One dimensionalize explicit data
# explicit_formatted_data_copy = copy.copy(explicit_formatted_data)
# explicit_formatted_data = []
# for i in range(number_of_eqps_values):
#     for list_of_values in explicit_formatted_data[i]:
#         explicit_formatted_data.append(list_of_values)
#     if i != number_of_eqps_values - 1:
#         explicit_formatted_data.append([np.nan,np.nan,np.nan,np.nan])

implicit_formatted_data_copy = copy.copy(implicit_formatted_data)
implicit_formatted_data = []
for i, e in enumerate(implicit_formatted_data_copy):
    for E in e:
        implicit_formatted_data.append(E)
    if i != (number_of_eqps_values-1):
        implicit_formatted_data.append([np.nan,np.nan,np.nan,np.nan])
#print(implicit_formatted_data)



n = 1
np.savetxt(f"D:/Work/software/st-bingo/research/data/processed_data/vm_{n}_transpose_bingo_format.txt", implicit_formatted_data)
np.savetxt(f"D:/Work/software/st-bingo/research/data/processed_data/vm_{n}_bingo_format.txt", explicit_formatted_data)
