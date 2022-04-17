import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import spinchain_dynamics as spd

#%%

N = 3
J = 1
h = 5
gammat = 5
esplit = 1
w = 1
Beta = 10
dtnum = 2000
tintnum = 1000

params = N, J, h, gammat, esplit, w, Beta, dtnum, tintnum
#%%

rhot = spd.lindsolve(params)

#%%

magn_0z = spd.magnetization(N = N, which = 0, direction = 'z', rhot = rhot)
magn_0x = spd.magnetization(N = N, which = 0, direction = 'x', rhot = rhot)
magn_0y = spd.magnetization(N = N, which = 0, direction = 'y', rhot = rhot)

magn_1z = spd.magnetization(N = N, which = 1, direction = 'z', rhot = rhot)
magn_1x = spd.magnetization(N = N, which = 1, direction = 'x', rhot = rhot)
magn_1y = spd.magnetization(N = N, which = 1, direction = 'y', rhot = rhot)

magn_2z = spd.magnetization(N = N, which = 2, direction = 'z', rhot = rhot)
magn_2x = spd.magnetization(N = N, which = 2, direction = 'x', rhot = rhot)
magn_2y = spd.magnetization(N = N, which = 2, direction = 'y', rhot = rhot)

#%%
frm = 0
to = 2000
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(magn_1x[frm:to], magn_1y[frm:to], magn_1z[frm:to])

#%%
frm = 2000
to = 4000
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot3D(magn_2x[frm:to], magn_2y[frm:to], magn_2z[frm:to])
