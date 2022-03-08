import matplotlib.pyplot as plt
import kinetic_MC_v2 as kmc
import numpy as np

N = 32
particle_number = int(N*N/4)
Betas = np.linspace(0.8,1.1,10)

it_num = 100000

final_corrs = kmc.transition_point_finder(Betas = Betas, N = N, particle_number = particle_number, it_num = it_num)

fig = plt.figure()

plt.scatter(Betas,final_corrs)
plt.ylabel(r"Final correlation  [1/particle]", fontsize = 14)
plt.xlabel(r"$ \beta $", fontsize = 14)

np.savetxt(fname = "/home/zoltan/Documents/BME_MSC[2021-2023]/ComputerSimulationsInPhysics/Week3/Homework/kinetic_MC_data_v3", X = np.array(final_corrs))