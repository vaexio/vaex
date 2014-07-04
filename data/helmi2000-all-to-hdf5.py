# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import h5py

from optparse import OptionParser
parser = OptionParser()

parser.add_option("--seed",
                 help="seed for RNG", default=0, type=int)
(options, args) = parser.parse_args()

dir1 = '/net/maury/data/users/ahelmi/DATA/gaia_jbh/IntegralsOfMotion/'

Nhalos = 33
Nparticles = 100000

hdf5filename = args[0]

h5file = h5py.File(hdf5filename, "w", driver="core")
datagroup = h5file.create_group("data")
group_x = datagroup.create_dataset("x", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_y = datagroup.create_dataset("y", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_z = datagroup.create_dataset("z", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_vx = datagroup.create_dataset("vx", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_vy = datagroup.create_dataset("vy", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_vz = datagroup.create_dataset("vz", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_E = datagroup.create_dataset("E", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_Lz = datagroup.create_dataset("Lz", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_L = datagroup.create_dataset("L", shape=(Nparticles * Nhalos,), dtype=np.float64)
group_FeH = datagroup.create_dataset("FeH", shape=(Nparticles * Nhalos,), dtype=np.float64)

x = np.zeros(Nparticles, dtype=np.float64)
y = np.zeros(Nparticles, dtype=np.float64)
z = np.zeros(Nparticles, dtype=np.float64)
vx = np.zeros(Nparticles, dtype=np.float64)
vy = np.zeros(Nparticles, dtype=np.float64)
vz = np.zeros(Nparticles, dtype=np.float64)

print "setting seed to: %d" % options.seed
np.random.seed(options.seed)

for halo_index in range(Nhalos):
	filename_EL = os.path.join(dir1, "en_angmom_f_000.%02d" %  halo_index)
	data = file(filename_EL, "rb").read()
	E_sth_Lz = np.fromstring(data[4:-4], dtype=np.float32)
	print len(E_sth_Lz)
	E_sth_Lz = E_sth_Lz.reshape((Nparticles+1, 3))
	E_sth_Lz = E_sth_Lz[1:,] # first particle is central
	group_Lz[halo_index*Nparticles:(halo_index+1)*Nparticles] = E_sth_Lz[:,2]
	group_L[halo_index*Nparticles:(halo_index+1)*Nparticles] = E_sth_Lz[:,1]
	group_E[halo_index*Nparticles:(halo_index+1)*Nparticles] = E_sth_Lz[:,0]
	
	filename_pos = "/net/maury/data/users/ahelmi/DATA/gaia/halo/simulations/fn/pos_b%02d_relf_nn.dat" % halo_index
	lines = file(filename_pos).readlines()
	assert len(lines) == Nparticles+1
	lines = lines[1:] # skip first line
	particle_index = 0
	for line in lines:
		pos = map(float, line.split())
		assert len(pos) == 3
		x[particle_index] = pos[0]
		y[particle_index] = pos[1]
		z[particle_index] = pos[2]
		particle_index += 1
	group_x[halo_index*Nparticles:(halo_index+1)*Nparticles] = x
	group_y[halo_index*Nparticles:(halo_index+1)*Nparticles] = y
	group_z[halo_index*Nparticles:(halo_index+1)*Nparticles] = z
		
	
	filename_vel = "/net/maury/data/users/ahelmi/DATA/gaia/halo/simulations/fn/vel_b%02d_relf_nn.dat" % halo_index
	lines = file(filename_vel).readlines()
	assert len(lines) == Nparticles+1
	lines = lines[1:] # skip first line
	particle_index = 0
	for line in lines:
		vel = map(float, line.split())
		assert len(vel) == 3
		vx[particle_index] = vel[0]
		vy[particle_index] = vel[1]
		vz[particle_index] = vel[2]
		particle_index += 1
	group_vx[halo_index*Nparticles:(halo_index+1)*Nparticles] = vx
	group_vy[halo_index*Nparticles:(halo_index+1)*Nparticles] = vy
	group_vz[halo_index*Nparticles:(halo_index+1)*Nparticles] = vz
	
	group_vz[halo_index*Nparticles:(halo_index+1)*Nparticles] = vz
	
	FeH_mean = np.random.normal(-1.7, 0.45)
	FeH_sigma = 0.13 + 0.05 * (np.random.random() - 0.5)*2
	FeH = np.random.normal(FeH_mean, FeH_sigma, Nparticles)
	print "FeH: %4.2f+/-%4.2f" % (FeH_mean, FeH_sigma)
	group_FeH[halo_index*Nparticles:(halo_index+1)*Nparticles] = FeH
	
	