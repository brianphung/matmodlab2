from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def np_interpolate(xval, df, xcol, ycol):
    # compute xval as the linear interpolation of xval where df is a dataframe and
    #  df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted.
    return np.interp([xval], df[xcol], df[ycol])[0]

def plot_Y_components(set_of_results, components, eqps_to_plot, labels, legend=False):
    eqps_panda = pd.DataFrame(eqps_to_plot, columns=['REAL_EQPS'])
    f, ax = plt.subplots(figsize=(4, 4), dpi=300)
    dataset = [ [] for _ in eqps_to_plot  ]
    for mps in set_of_results:
        stress_history = mps.df[['S.XX', 'S.YY', 'S.ZZ', 'REAL_EQPS']]

        #mps.plot('E.XX', 'S.XX')

        stress_history = stress_history[stress_history['REAL_EQPS'] > 1e-6]

        # regressorSXX = LinearRegression()
        # regressorSXX.fit(stress_history[['REAL_EQPS']], stress_history[['S.XX']])

        # regressorSYY = LinearRegression()
        # regressorSYY.fit(stress_history[['REAL_EQPS']], stress_history[['S.YY']])

        # regressorSZZ = LinearRegression()
        # regressorSZZ.fit(stress_history[['REAL_EQPS']], stress_history[['S.ZZ']])

        # SXX = regressorSXX.predict(eqps_panda)
        # SYY = regressorSYY.predict(eqps_panda)
        # SZZ = regressorSZZ.predict(eqps_panda)
        SXX = np_interpolate(eqps_to_plot, stress_history, 'REAL_EQPS', 'S.XX')
        SYY = np_interpolate(eqps_to_plot, stress_history, 'REAL_EQPS', 'S.YY')
        SZZ = np_interpolate(eqps_to_plot, stress_history, 'REAL_EQPS', 'S.ZZ')
        #print(eqps_to_plot, SXX, SYY, SZZ)

        for _, eqps in enumerate(eqps_to_plot):
            #print(_)
            dataset[_].append([SXX[_], SYY[_], SZZ[_]])
    dataset = [ np.array(ds) for ds in dataset ]
    for _, eqps in enumerate(eqps_to_plot):
        ax.scatter(dataset[_][:,components[0]], dataset[_][:,components[1]] , marker='.', label=f'EQPS {eqps:.3f}')

    ax.set_xlabel(f'{labels[0]}')
    ax.set_ylabel(f'{labels[1]}')
    ax.set_aspect('equal', adjustable='box')
    if legend == True:
        ax.legend()

    return ax