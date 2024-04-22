from calculations import AeroForce, Rotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

test = AeroForce(dir_polar="data/NACA_643_618")
test._prepare("BL")

df_sep = pd.read_csv("data/NACA_643_618/Beddoes_Leishman_calculations/sep_points.dat")
df_polar = pd.read_csv("data/NACA_643_618/polars.dat", delim_whitespace=True)
alpha_0 = np.deg2rad(-3.8380952380952382)
rot = Rotations()

C_t = 2*np.pi*np.sin(np.deg2rad(df_sep["alpha"])-alpha_0)**2*np.sqrt(np.abs(df_sep["f_t"]))*np.sign(df_sep["f_t"])
C_n = 2*np.pi*np.sin(np.deg2rad(df_sep["alpha"])-alpha_0)*\
    ((1+np.sqrt(np.abs(df_sep["f_n"]))*np.sign(df_sep["f_n"]))/2)**2

rotated = rot.rotate_2D(np.c_[C_t, C_n], np.deg2rad(df_sep["alpha"]).to_numpy())
C_d = -rotated[:, 0]
C_l = rotated[:, 1]

fig, ax = plt.subplots(2)
ax[0].plot(df_polar["alpha"], df_polar["Cl"], "x")
ax[0].plot(df_sep["alpha"], C_l)

ax[1].plot(df_polar["alpha"], df_polar["Cd"], "x")
ax[1].plot(df_sep["alpha"], C_d)
plt.show()


# from scipy.interpolate import griddata
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from calculations import Rotations

# rot = Rotations()

# df_polar = pd.read_csv("data/NACA_643_618/polars.dat", delim_whitespace=True)
# coefs = np.c_[df_polar["Cd"].to_numpy(), df_polar["Cl"].to_numpy()]
# alpha = np.deg2rad(df_polar["alpha"].to_numpy())
# C_n = np.cos(alpha)*df_polar["Cl"].to_numpy()+np.sin(alpha)*df_polar["Cd"].to_numpy()
# plt.plot(C_n, df_polar["Cm"])
# plt.show()

from calculations import ThreeDOFsAirfoil, AeroForce
import numpy as np
import matplotlib.pyplot as plt

res = 400
t = np.linspace(0, 40, res)
k = 0.1
alpha = np.deg2rad(15+10*np.sin(2*k*t))


test = ThreeDOFsAirfoil("data/NACA_643_618", t, None, None, 1, None, None, None, None, None, None)
test.pos = np.c_[np.zeros((res, 2)), -alpha]
test.vel = (test.pos[1:, :]-test.pos[:-1, :])/t[1]
test.inflow = np.c_[np.ones_like(t), np.zeros_like(t)]
test.dt = t[1:]-t[:-1]
test.density = 1
# test.set_aero_calc("BL", A1=0.3, A2=0.7, b1=0.14, b2=0.53)
test.set_aero_calc("BL", A1=0.165, A2=0.335, b1=0.045, b2=0.3)

aero = AeroForce("data/NACA_643_618")
aero._prepare("BL")
aero._alpha_0 = np.deg2rad(-3.7)

for i in range(res-1):
    aero._BL(test, i, A1=0.3, A2=0.7, b1=0.14, b2=0.53)

fig, ax = plt.subplots()
ax.plot(np.rad2deg(alpha[1:-1]), test.CnC[1:-1], label="CnC")
ax.plot(np.rad2deg(alpha[1:-1]), test.CnI[1:-1], label="CnI")
ax.plot(np.rad2deg(alpha[1:-1]), test.CnPot[1:-1], label="CnPot")
ax.legend()
ax.set_ylim((-1, 3))
ax.grid()
plt.show()









