#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:07:48 2019

@author: KP
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.append("/Users/KP/Google/Python/Lib/QD_lib")
sys.path.append("D:/Google Drive/Python/Lib/QD_lib")
import parameter, matrix, geometry
from scipy.sparse.linalg import eigs, spsolve
from scipy.sparse import eye, csr_matrix, spdiags


# for  QD
#materials = ['ZnSe', 'ZnS', 'CdS']  
#radius = [4.6e-9, 2.6e-9, 2e-9]  # from outside to inside
dr = 2e-10
dz= 2e-10

""""""""""""""""   Nanorod Dimension """""""""""""""""""""""
" Material1 = [Material, radius, starting z, ending z]     "
" Materials = [Material1, Material2, Material3]            "  
" Mat3's ending z should be the highest                    "
" Material1's starting z should be 2e-9                    "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# for Nanorod
#material1 = ['ZnSe', 2e-9, 2e-9, 5e-9]
#material2 = ['CdS', 2e-9, 5e-9, 8e-9]
#materials = [material1, material2]

""""""""""""""""   Nanorod Dimension """""""""""""""""""""""
" rod = [Material, radius, starting z, ending z]           "
" dot = [Material, radius, center point (z)]               "
" Materials = [rod, dot]                                   "  
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# for dot in rod
rod = ['CdS', 2e-9, 2e-9, 10e-9] 
dot = ['ZnSe', 2e-9, 5e-9]
materials = [rod, dot]

def int_cyl(function, r, n, dr, dz): # return cylindrical integration of function**2
    fsquare = np.multiply(function, np.conjugate(function))
    temp = np.zeros_like(function)
    for i, rad in enumerate(r[1:]):
        temp[n*i:n*(i+1)]  = fsquare[n*i:n*(i+1)] * rad
    int_val = np.sum(temp) * dr * dz* 2 * np.pi
    return int_val

class Builder:
    def __init__(self):
        # Define Parameter
        self.hbar = 6.626e-34/2/np.pi     # [m2 kg/s]
        self.e = 1.6e-19                  # [c]
        self.eo = 8.85e-12;               # [F/m]
        self.emass = 9.1e-31;                 # [kg]
        self.delta = 1e-12   # to avoid divergence
        self.plank = 6.626e-34
        self.Ep = 23
        self.k_num = np.sqrt(self.Ep/2)

    def QD(self, materials, radius): 
        temp1 = 0
        temp2 = 10
        mater = []
        for key in materials:
            temp = parameter.material(key)
            mater.append(temp)
            if temp.cb > temp1:
                temp1 = temp.cb + 0
            if temp.vb < temp2:
                temp2 = temp.vb + 0
        self.Eg = temp2-temp1
        print ('Bulk Eg = {}'.format(round(self.Eg, 2)))
        
        ro = radius[0] + 2e-9
        self.r = np.arange(0, ro+dr, dr) 
        self.zo = radius[0]*2 + 4e-9
        self.z = np.arange(0, self.zo + dz, dz) 

        self.m = self.r.size
        self.n = self.z.size        
        print ('mesh size = {} x {} = {}'.format(self.m, self.n, self.m * self.n))
        
        self.er, self.cb, self.vb, self.me, self.mh = geometry.map2d(self.m, self.n, self.r, self.z, self.zo, mater, radius)
        self.cb_array = self.cb.reshape(self.m * self.n, 1) - self.cb.min() + self.Eg/2
        self.vb_array = self.vb.reshape(self.m * self.n, 1) - self.vb.min() + self.Eg/2
        self.er_array = self.er.reshape(self.m * self.n,1)
        self.me_array = self.me.reshape(self.m * self.n,1)
        self.mh_array = self.mh.reshape(self.m * self.n,1)
    
    def nanorod(self, materials):
        temp1 = 0
        temp2 = 10        
        for key in materials:
            temp = parameter.material(key[0])
            if temp.cb > temp1:
                temp1 = temp.cb + 0
            if temp.vb < temp2:
                temp2 = temp.vb + 0
        self.Eg = temp2-temp1
        print ('Bulk Eg = {}'.format(round(self.Eg, 2)))        
        
        ro = 0
        nmat = len(materials)
        for mat in materials:
            if mat[1] > ro:
                ro = mat[1]
        self.r = np.arange(0, ro + 2e-9 + dr, dr)
        self.zo = materials[-1][-1] + 2e-9
        self.z = np.arange(0, self.zo+dz, dz)
        self.m = self.r.size
        self.n = self.z.size        
        print ('mesh size = {} x {} = {}'.format(self.m, self.n, self.m * self.n))
        
        geo = np.zeros((self.m, self.n))
        self.er = np.ones((self.m, self.n)) * 2.5
        self.cb = np.zeros((self.m, self.n))
        self.vb = np.ones((self.m, self.n)) * 10
        self.me = np.ones((self.m, self.n))
        self.mh = np.ones((self.m, self.n))
             
        for i in range(nmat):            
            geo[:np.int(materials[i][1]/dr), np.int(materials[i][2]/dz):np.int(materials[i][3]/dz)] = 1 + i
        
        
        for i in range(len(self.r)):
            for j in range(len(self.z)):
                for k in range(nmat):
                    if geo[i,j] == 1 + k:
                        self.er[i,j] = parameter.material(materials[k][0]).er
                        self.cb[i,j] = 0 - parameter.material(materials[k][0]).cb
                        self.vb[i,j] = parameter.material(materials[k][0]).vb
                        self.me[i,j] = 1/parameter.material(materials[k][0]).me
                        self.mh[i,j] = 1/parameter.material(materials[k][0]).mh
        
        self.cb_array = self.cb.reshape(self.m * self.n, 1) - self.cb.min() + self.Eg/2
        self.vb_array = self.vb.reshape(self.m * self.n, 1) - self.vb.min() + self.Eg/2
        self.er_array = self.er.reshape(self.m * self.n,1)
        self.me_array = self.me.reshape(self.m * self.n,1)
        self.mh_array = self.mh.reshape(self.m * self.n,1)
        
    def dot_in_rod(self, materials):
        temp1 = 0
        temp2 = 10        
        for key in materials:
            temp = parameter.material(key[0])
            if temp.cb > temp1:
                temp1 = temp.cb + 0
            if temp.vb < temp2:
                temp2 = temp.vb + 0
        self.Eg = temp2-temp1
        print ('Bulk Eg = {}'.format(round(self.Eg, 2)))        
        
        ro = 0
        nmat = len(materials)
        for mat in materials:
            if mat[1] > ro:
                ro = mat[1]
        self.r = np.arange(0, ro + 2e-9 + dr, dr)
        self.zo = materials[0][-1] + 2e-9
        self.z = np.arange(0, self.zo+dz, dz)
        self.m = self.r.size
        self.n = self.z.size        
        print ('mesh size = {} x {} = {}'.format(self.m, self.n, self.m * self.n))
        
        geo = np.zeros((self.m, self.n))
        self.er = np.ones((self.m, self.n)) * 2.5
        self.cb = np.zeros((self.m, self.n))
        self.vb = np.ones((self.m, self.n)) * 10
        self.me = np.ones((self.m, self.n))
        self.mh = np.ones((self.m, self.n))
        
        geo[:np.int(materials[0][1]/dr), np.int(materials[0][2]/dz):np.int(materials[0][3]/dz)] = 1
        for i in range(self.m):
            for j in range(self.n):
                if self.r[i]**2 + (self.z[j] - materials[1][2])**2 <= materials[1][1]**2:
                    geo[i,j] = 2
                    
        for i in range(len(self.r)):
            for j in range(len(self.z)):
                for k in range(nmat):
                    if geo[i,j] == 1 + k:
                        self.er[i,j] = parameter.material(materials[k][0]).er
                        self.cb[i,j] = 0 - parameter.material(materials[k][0]).cb
                        self.vb[i,j] = parameter.material(materials[k][0]).vb
                        self.me[i,j] = 1/parameter.material(materials[k][0]).me
                        self.mh[i,j] = 1/parameter.material(materials[k][0]).mh
        
        self.cb_array = self.cb.reshape(self.m * self.n, 1) - self.cb.min() + self.Eg/2
        self.vb_array = self.vb.reshape(self.m * self.n, 1) - self.vb.min() + self.Eg/2
        self.er_array = self.er.reshape(self.m * self.n,1)
        self.me_array = self.me.reshape(self.m * self.n,1)
        self.mh_array = self.mh.reshape(self.m * self.n,1)    
    
    def plot_energy_band(self):
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize = (16,7))
        ax1 = plt.subplot(141)
        ax1.contourf(self.r, self.z, self.cb.transpose())
        plt.title('Conduction Band')
        ax1.set_xlabel('Radial axis')
        ax1.set_ylabel('Cylindrical axis')
        
        ax2 = plt.subplot(142)
        ax2.contourf(self.r, self.z, self.vb.transpose())
        plt.title('Valence Band')
        ax2.set_xlabel('Radial axis')
        ax2.set_ylabel('Cylindrical axis')
        
        ax3 = plt.subplot(143)
        ax3.plot(self.z, self.cb[0,:self.n] , 'b')
        ax3.plot(self.z, -self.vb[0,:self.n] - self.Eg, 'r')
        ax3.set_xlabel('Cylindrical axis')
        
        ax4 = plt.subplot(144)
        ax4.contourf(self.r, self.z, self.er.T, 10, vmin = 4, vmax = 20)
        plt.tight_layout()
    
    def matrix(self):
        self.ke = matrix.inhomo_laplacian(self.m, dr, self.n, dz, self.r, self.me_array) 
        self.kh = matrix.inhomo_laplacian(self.m, dr, self.n, dz, self.r, self.mh_array) 
        self.ke = self.ke / self.e * self.hbar ** 2 / 2 * (-1) / self.emass
        self.kh = self.kh / self.e * self.hbar ** 2 / 2 * (-1) /self.emass
        self.pe = matrix.potential(self.cb_array[self.n:,0]).todense()
        self.ph = matrix.potential(self.vb_array[self.n:,0]).todense()
        te = self.ke + self.pe
        th = (self.kh + self.ph) * -1
        
        off_1 = eye(self.n * (self.m-1), k = 1)
        off_2 = eye(self.n * (self.m-1), k = -1)
        off_3 = eye(self.n * (self.m-1), k = self.n)
        off_4 = eye(self.n * (self.m-1), k = -self.n)
        off = off_1/dz -off_2/dz +off_3/dr -off_4/dr
        off = off*(-1j) * self.hbar * self.k_num/2
        off = off.todense()
        
        self.kp = np.zeros((2 * self.n * (self.m-1), 2 * self.n * (self.m-1)), dtype=complex)
        self.kp[:self.n * (self.m-1), self.n * (self.m-1) : 2 * self.n * (self.m-1)] = off   # 2nd quadrant
        self.kp[self.n * (self.m-1) : 2 * self.n * (self.m-1), :self.n * (self.m-1)] = off   # 3rd quadrant
        self.kp[:self.n * (self.m-1), : self.n * (self.m-1)] = te    # 1st quadrant
        self.kp[self.n * (self.m-1) : 2 * self.n * (self.m-1), self.n * (self.m-1): 2 * self.n * (self.m-1)] = th # 4th quadrant
        self.sp_kp = csr_matrix(self.kp)
        self.mat_size = (self.m-1) * self.n * 2
        
    def e_field_matrix(self, field):  # EF : electric field [kV/cm]
        field *= 1e5   # [V/m]
        self.VE = np.linspace(0, field * self.z[-1], len(self.z))
        self.VE = np.tile(self.VE, (self.m - 1))
        self.VE = np.append(self.VE, self.VE)
        self.VE = spdiags(self.VE, [0], self.mat_size, self.mat_size, format = 'csr')

    def coulomb_matrix(self, solution, charges = 'exciton', image_charge = False):
        
        vcoul = matrix.inhomo_laplacian(self.m, dr, self.n, dz, self.r, self.er_array)
        vcoul_homo = matrix.inhomo_laplacian(self.m, dr, self.n, dz, self.r, np.ones((len(self.er_array),1))*9)
        vch = spsolve(vcoul, np.abs(solution.psi_h2) * self.e / self.eo)
        vce = spsolve(vcoul, np.abs(solution.psi_e2) * self.e / self.eo)
        vch_homo = spsolve(vcoul_homo, np.abs(solution.psi_h2) * self.e / self.eo)
        vce_homo = spsolve(vcoul_homo, np.abs(solution.psi_e2) * self.e / self.eo)
        
        if charges == 'electron':
            if image_charge == False:
                vc = np.concatenate([np.zeros(len(vch)), -vce])
            elif image_charge == True:
                vc = np.append(vce_homo - vce, -vce)
        elif charges == 'hole':
            if image_charge == False:
                vc = np.append(vch, np.zeros(len(vch)))
            elif image_charge == True:
                vc = np.append(vch, vch - vch_homo)
        elif charges == 'exciton':
            if image_charge == False:
                vc = np.append(vch, -vce)
            elif image_charge == True:
                vc = np.append(vch + vce_homo-vce, -vce + vch - vch_homo)
        self.vcm = spdiags(vc, [0], self.mat_size, self.mat_size, format = 'csr')

class Solver:      
    def solve(self, sys, e_field = False, coulomb_potential = False):
        total = sys.sp_kp.copy()
        if e_field == True:
            total = sys.sp_kp + sys.VE
            print ('Electric field added')
        if coulomb_potential == True:
            total = sys.sp_kp + sys.vcm
        
        self.eev, eef= eigs(total, 1, sigma = sys.Eg/2 - 0.1, which='LM')
        self.hev, hef= eigs(total, 1, sigma = -sys.Eg/2 + 0.1, which='LM')
        eef = eef[:(sys.m-1) * sys.n]
        hef = hef[(sys.m-1) * sys.n: (sys.m-1) * sys.n * 2]
        self.psi_e = eef / np.sqrt(int_cyl(eef, sys.r, sys.n, dr, dz)) 
        self.psi_h = hef / np.sqrt(int_cyl(hef, sys.r, sys.n, dr, dz))
        self.psi_e2 = np.multiply(self.psi_e , np.conjugate(self.psi_e))
        self.psi_h2 = np.multiply(self.psi_h , np.conjugate(self.psi_h))
        self.e2d = self.psi_e2.reshape((sys.m-1), sys.n)
        self.h2d = self.psi_h2.reshape((sys.m-1), sys.n)
        self.energy = np.abs(self.eev - self.hev)[0]
        print ('Energy : {} eV'.format(self.energy))

    def plot(self, sys):
        fig, ((ax1, ax3, ax4)) = plt.subplots(1, 3, figsize = (8,4))
        ax1.plot(sys.z * 1e9, np.abs(self.psi_e2[:sys.n]), color = 'blue', label = 'Electron Wavefunction')
        ax1.set_xlabel('Cylindrical axis')
        ax1.set_ylabel('Wavefunctions [a.u]')
        #ax1.axvline(1e9 * (sys.zo/2 + radius[0]), color = 'g', linestyle = '--')
        #ax1.axvline(1e9 * (sys.zo/2 - radius[0]), color = 'g', linestyle = '--')
        ax2 = ax1.twinx()
        ax2.plot(sys.z * 1e9, np.abs(self.psi_h2[:sys.n]), color = 'red', label = 'Hole Wavefunction')
        
        ax3 = plt.subplot(132)
        ax3.contourf(sys.r[1:] * 1e9, sys.z * 1e9, self.e2d.transpose())
        ax3.set_title('Electron wavefunction')
        ax3.set_ylabel('Cylinder axis [nm]')
        ax3.set_xlabel('Radial axis [nm]')
        
        ax4 = plt.subplot(133)
        ax4.contourf(sys.r[1:] * 1e9, sys.z*1e9, self.h2d.transpose())
        ax4.set_title('Hole wavefunction')
        ax4.set_ylabel('Cylinder axis [nm]')
        ax4.set_xlabel('Radial axis [nm]')
        plt.tight_layout()
    
    def self_consistency(self, sys, number,  e_field = False, charges = 'exciton', image_charge = False):
        self.solve(sys, e_field = e_field)   
        old_energy = self.energy
        
        for i in range(number):
            sys.coulomb_matrix(self, charges = charges, image_charge = image_charge)
            self.solve(sys, e_field = e_field, coulomb_potential = True)
            energy_difference = np.abs(old_energy - self.energy)
            if energy_difference < 0.001:
                break
            print ('{} : Energy difference : {}'.format(i, energy_difference))
            old_energy = self.energy
            
