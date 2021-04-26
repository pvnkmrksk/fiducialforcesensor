"""
Make graphs for uniaxial (F=kx, where k.shape is (1,1))

date: 16 Feb 2019
author: nouyang

"""

from scipy.stats.stats import pearsonr
from sklearn import metrics
from sklearn import linear_model
from scipy.stats import linregress
from scipy.interpolate import interp1d
import copy
import math
import sys
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


sns.set(rc={"figure.figsize": (16, 10)},
        font_scale=1.8)
plt.style.use('bmh')
# plt.style.use('fivethirtyeight')
# sns.set() #use seaborn style for matplotlib
# ===============================================
#    DECLARE FILES ####
# ===============================================
path = "data"

DATA_FOLDER = os.getcwd()

datafilename = '2019-09-13 12:02:07'  


def extractDF(filename):
    opto_files = [
        file
        for file in glob.glob(
            os.path.join(DATA_FOLDER, path, filename + "_optoforceData.csv")
        )
    ]
    tag_files = [
        file
        for file in glob.glob(
            os.path.join(DATA_FOLDER, path, filename + "_arucotagData.csv")
        )
    ]
    print("\n--------------------------\n")
    print(filename)
    # print(opto_files)
    # print(tag_files)
    opto_fname = opto_files[0]
    tag_fname = tag_files[0]

    # ===============================================
    #    Import Data ####
    # ===============================================
    opto_cols = ["Description", "Timestamp",
                 "x", "y", "z", "theta", "phi", "gamma"]

    # following are zero and averaged (in the original collecting)
    tag_cols = ["Description", "Timestamp", "x1", "y1", "z1", "x2", "y2", "z2",
                "theta1", "phi1", "gamma1", "theta2", "phi2", "gamma2",
                "_x", "_y", "_z", "_theta", "_phi", "_gamma", "filter weight"]


    opto_df = pd.read_csv(
        opto_fname,
        header=None,
        sep=";",
        names=opto_cols,
        skip_blank_lines=True,
        usecols=range(8),
    )
    print('head opto', opto_df['z'].head)

    opto_df = opto_df.drop(["Description"], 1)

    tag_df = pd.read_csv(
        tag_fname,
        header=None,
        sep=";",
        names=tag_cols,
        skip_blank_lines=True,
        usecols=range(20),  #skip the weights column
    )

    tag_df = tag_df.drop(["Description"], 1)

    # ===============================================
    #   Conversion to real units ####
    # ===============================================

    # --- Tag
    # -- MULTIPLY - convert tag from meters to mm
    for var in ["x1", "y1", "z1"] + ["_x", "_y", "_z"] + ["x2", "y2", "z2"]:
        tag_df[var] *= 1000

    # convert optoforce counts to newtons (provided by optoforce datasheet)
    # print('\nx axis opto before conversion', opto_df['x'].head)
    opto_df["x"] /= 96.11
    # print('\nx axis opto after conversion', opto_df['x'].head)
    print(opto_df['x'].dtype)
    opto_df["y"] /= 95.64
    print('\nz axis opto before conversion', opto_df['z'].head)
    opto_df["z"] /= 20.92
    print('\nz axis opto after conversion', opto_df['z'].head)

    # --- Optoforce
    # -- convert counts to newton-meters (provided by optoforce datasheet)
    # -- Note: We don't account for if conversion may have changed since factory calibration
    opto_df["theta"] /= 5425.36
    opto_df["phi"] /= 5524.60
    opto_df["gamma"] /= 8103.25

    # -- MULTIPLY - convert optoforce, | N to mN | N*m to mN*m |
    print('before zero', opto_df.head(2))
    avg_opto_zeroing = np.average(opto_df.head(10), axis=0)
    print('zero', avg_opto_zeroing)
    opto_df -= avg_opto_zeroing
    print('after zero', opto_df.head(2))

    for a in ["x", "y", "z", "theta", "phi", "gamma"]:
        opto_df[a] *= 1000 # N to mN
    print('z axis opto N to mN', opto_df['z'].head)


    # -- print range of forces measured

    print("\n--------- Force ranges measured -----------------\n")
    for a in ["x", "y", "z", "theta", "phi", "gamma"]:
        print("%s axis force. Min: %0.3f, Max: %0.3f mN or mN*m" %
              (a, opto_df[a].min(), opto_df[a].max()))

    # print(opto_df.head())
    # print(opto_df.describe())

    # ===============================================
    #    FIX THE DATA (swap axis etc.) ####
    # ===============================================
    # NOTE: truncate beginning and end samples, noise bad for lin fit

    tag = tag_df.copy()[20:]
    opto = opto_df.copy()
    # [20:] tag = tag_df.copy()

    # Swap axis as needed to align tag axes with opto axis
    # NOTE: no longer needed (fixed in hardware)

    # Negate axes as needed
    negateList = ["y1", "z1",  "y2", "z2",
                  "theta1", "theta2", "gamma1", "gamma2",
                  "_y", "_z", "_theta", "_gamma"]


    for n in negateList:
        tag[n] *= -1

    return tag, opto


# ===============================================
#    Linear regress to match apriltag to optoforce & plot ####
# ===============================================


avgd_prefix = "_"
# tagnum_suffix = '2'
tagnum_suffix = ''


def gen_interp(tag_df, opto_df):
    list_axis = ['x', 'y', 'z', 'theta', 'phi', 'gamma']
    interp_tag_data = []
    interp_opto_data = []
    delays = []
    for AXIZ in list_axis:
        print("\n>--- delay AXIZ: ", AXIZ)
        a_x = tag_df[avgd_prefix + AXIZ + tagnum_suffix ]  # single tag
        # a_x = tag_df[avgd_prefix + AXIZ]  # single tag
        a_t = tag_df.Timestamp

        o_x = opto_df[AXIZ]
        o_t = opto_df.Timestamp

        # interp_tagfxn = interp1d(a_t - lag, a_x, kind='linear', fill_value='extrapolate')
        num_samples = len(o_t) + len(a_t)
        begin = min(o_t.min(), a_t.min())
        end = max(o_t.max(), a_t.max())
        t_regular = np.linspace(begin, end, num_samples * 2)

        # a_x = a_x[20:]
        f_o = interp1d(o_t, o_x, kind="linear", fill_value="extrapolate")
        f_a = interp1d(
            a_t, a_x, kind="linear", fill_value="extrapolate"
        )  # definitely not cubic!
        interp_o = f_o(t_regular)
        interp_a = f_a(t_regular)

        # normalize for cross-correlation
        norm_interp_a = interp_a - interp_a.mean()
        norm_interp_a /= norm_interp_a.std()
        norm_interp_o = interp_o - interp_o.mean()
        norm_interp_o /= norm_interp_o.std()

        corr = np.correlate(norm_interp_o, norm_interp_a, mode="full")
        # source https://dsp.stackexchange.com/questions/9058/measuring-time-delay-of-audio-signals
        delay = int(len(corr) / 2) - np.argmax(corr)
        delays.append(delay)
        # TODO: why are all the delays different? 
        # delay = 11  # NOTE hardcod3

        print("\n>--- delay # datapoints: ", delay, "; in real time: ",
              delay * (t_regular[1] - t_regular[0]))
        timefix_a_x = np.roll(interp_a, shift=-delay)

        # gunk = 250
        # gunky = -250
        gunk = 200
        gunky = -500
        # remove some gunkiness, I think it's introduced by the "extrapolate" option
        # on the interpolation
        t_regular = t_regular[gunk:gunky]
        y = interp_o.reshape(-1, 1)[gunk:gunky]
        X = np.array(timefix_a_x).reshape(-1, 1)[gunk:gunky]
        # interp_opto_data[AXIZ] = y
        # interp_tag_data[AXIZ] = X
        # print('xamples of y', y[100:110])
        # print('samples of X', X[100:110])
        interp_opto_data.append(y)
        interp_tag_data.append(X)
        # TODO: plot at this intermediate stage, to diagnose why 
        # later on the "gunky" beginning and ending values after linreg
        # turn out so gunky

        # print('interp o', interp_opto_data.shape)
        # interp_opto_data = np.vstack((interp_opto_data, y))
        # print('interp o', interp_opto_data.shape)
        # print('y', y.shape)
        # interp_tag_data = np.vstack((interp_tag_data, X))
    interp_opto_data = np.array(interp_opto_data)[:, :, 0]
    interp_tag_data = np.array(interp_tag_data)[:, :, 0]

    return t_regular, interp_tag_data, interp_opto_data, delays


# delay = '12=0.045'  # NOTE HARDCODE
# delay = 'normal'

afile = datafilename
tag_df, opto_df = extractDF(afile)  # all 6 axis
t_regular, tag_data, opto_data, delays = gen_interp(tag_df, opto_df)
print('delays', delays)

print(tag_data.shape)
print(opto_data.shape)
# for i, key in enumerate(tag_data):
# print('i', i, 'key', key)
# print('shape', tag_data[key].shape)
tag_data = tag_data.T
opto_data = opto_data.T
regr = linear_model.LinearRegression().fit(tag_data, opto_data)
regr = linear_model.Ridge().fit(tag_data, opto_data)
y_est = regr.predict(tag_data)
# tag_data2 = np.array([tag_df['_x'], tag_df['_y'], tag_df['_z'],
                      # tag_df['_theta'], tag_df['_phi'], tag_df['_gamma']])
# y_est = regr.predict(tag_data2.T)

print('y_est shape', y_est.shape)
print(y_est.shape[0])

print('regression coefficionts', regr.coef_)
print('!-- shape!--', regr.coef_.shape)
# print(regr.intercept_)
# print(regr.score(y_est, opto_data))
# print(y_est - opto_data)
# print(np.sum(y_est - opto_data))

# ------------ PLOT overlaid
f, axess = plt.subplots(nrows=2, ncols=3, sharex=False)
axess = np.array(axess).flatten()
ax1, ax2, ax3, ax4, ax5, ax6, = axess

axes_reordered = [ax1, ax4, ax2, ax5, ax3, ax6, ]

# -- Plot row of x,y,z tag, then row of x,y,z, opto, 12 plots
for i in range(y_est.shape[1]):
    print(y_est.shape)
    print(opto_data.shape)
    figA = axess[i]
    a_x = y_est[:, i]
    o_x = opto_data[:, i]
    figA.plot(t_regular, o_x, 'k.', ms=1.5,)
    # label=('Measured opto data: ' + str(i)))

    figA.plot(t_regular, a_x, 'r.', ms=1,)
    # label=('Measured opto data: ' + str(i)))

    # figA.plot(t_regular, tag_data[:, i], 'r.', ms=1.5,)
    # label=('Measured tag data: ' + str(i)))
    # figA.legend.

# ax1 = fig_axs[0]
# ax4 = fig_axs[3]
# ax2 = fig_axs[1]
# ax5 = fig_axs[4]
ax1.set_title('X Force')
ax2.set_title('Y Force')
ax3.set_title('Z Force')

ax4.set_title('X Moment')
ax5.set_title('Y Moment')
ax6.set_title('Z Moment')


ax1.set_ylabel("Force (mN)")
ax4.set_ylabel("Torque (mN*m)")
ax5.set_xlabel("Time (secs)")

plt.tight_layout()

# plt.suptitle("Timesteps data, y_est vs y")
plt.savefig('./plots/' + datafilename + '-' + 'delay-' + str(delays[0]) +
            '-' + avgd_prefix + 'axis' + tagnum_suffix + '.png')
# plt.show()

# ------------ YES: PLOT LINEAR FIT ---------------------------
f, fig_axs = plt.subplots(nrows=2, ncols=3, sharex=False)
fig_axs = np.array(fig_axs).T.flatten(order='F')

# Plot of Uniaxial, 6 single axis
for i in range(y_est.shape[1]):
    fig_ax = fig_axs[i]
    a_x = y_est[:, i]
    o_x = opto_data[:, i]
    # print('shapes', a_x.shape, o_x.shape)
    fig_ax.plot(o_x, o_x, 'k-')  # label='Optoforce data '+key)
    print(a_x.shape)
    print(o_x.shape)
    # regr.score(a_x.reshape(-1, 1), o_x.reshape(-1, 1))
    r_score = pearsonr(o_x, a_x)
    print(r_score)
    fig_ax.plot(o_x, a_x, 'r.', markersize=1.5,
                # label=('Linear fit estimate\nR: %.2f, axis %d' % (r_score[0]**2, i)))
                label=('$R^2$ = %.3f' % (r_score[0]**2)))
    # print('i: ', i)
    # print("coef: %f   intercept: %f   r-score: %f"
    # % (regr.coef_, regr.intercept_, r_score))
    fig_ax.legend(markerscale=0, markerfirst=False)

ax1, ax2, ax3, ax4, ax5, ax6, = fig_axs
# ax1 = fig_axs[0]
# ax4 = fig_axs[3]
# ax2 = fig_axs[1]
# ax5 = fig_axs[4]
ax1.set_title('X Force')
ax2.set_title('Y Force')
ax3.set_title('Z Force')

ax4.set_title('X Moment')
ax5.set_title('Y Moment')
ax6.set_title('Z Moment')

ax1.set_ylabel("Prototype Sensor (mN)")
ax4.set_ylabel("Prototype Sensor (mN*m)")
ax2.set_xlabel("Commercial Sensor (mN)")
ax5.set_xlabel("Commercial Sensor (mN*m)")
# plt.suptitle("!--- Linear fits (uniaxial) ---!\n Perfect fit is black line\nRed dots are \
# tag measurements\n(1st order linear fit with affine term)")
# plt.gca().set_aspect('equal', adjustable='datalim')
plt.tight_layout()
# plt.axis('scaled')

plt.savefig('./plots/LINFIT-' + datafilename + '-' + 'delay-'\
            + str(delays[0]) +
            '-' + avgd_prefix + 'axis' + tagnum_suffix + '.png')
plt.show()
