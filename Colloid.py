#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from math import *
import time as tm
import numpy as np
import random
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from datetime import datetime

# Tabulation for all the cosinus of the interior angles for regular polygons
DictCosine = dict()

DictStructures = dict()  # All the obtained structures in the dynamics

# Update the dictionary of the cosinus of interior angles for regular polygons with n vertices


def Spread(x, a=0.25, b=0.1, c=0.75, d=0.9):
    if x<0 or x >1:
        print('Warning Spread')
    if x<=a:
        return x*b/a
    elif x <=c:
        return(d-b)*(x-a)/(c-a)+b
    else:
        return(1-d)*(x-c)/(1-c)+d
    
    
def PrintDictStructures(simplified=True):
    if simplified:
        spear = 0
        ring = 0
        complexe = 0
        disconnected = 0
        for keys in DictStructures:
            if DictStructures[keys]*keys[0] > 0.5 :
                spear += DictStructures[keys]
            if DictStructures[keys]*keys[1] > 0.5 :
                ring += DictStructures[keys]
            if DictStructures[keys]*keys[3] > 0.5 :
                complexe += DictStructures[keys]
            if keys[0]+keys[1]+keys[3] > 1.5 :
                disconnected += DictStructures[keys]
        print("DictStructures =", (spear,ring,complexe, disconnected))
    else:
        print(DictStructures)
        

def ReturnDictCosine(n):
    global DictCosine
    if not n in DictCosine:
        liste = []
        for i in range(n//2):
            liste.append(cos(2*(i+1)*pi/n)**2)
            liste.append(cos(2*(i+1)*pi/n)**2)
        if n % 2 == 0:
            liste.pop(n-1)
        DictCosine[n] = liste
    return DictCosine[n]

# Moyenne glissante


def MovingAverage(liste, param=1):
    liste_averaged = []
    for i in range(len(liste)-param+1):
        average = 0
        for j in range(param):
            average += liste[i+j]
        liste_averaged.append(average/param)

# Norm and square of the norm for a given vector m_0


def Norm(m_0):
    return sqrt(NormSquare(m_0))


def NormSquare(m_0):
    return np.dot(m_0, m_0)

# Rotations with respect to the 3 canonical axis


def RotationX(angle):
    return np.array([[1., 0., 0.], [0., cos(angle), -sin(angle)], [0., sin(angle), cos(angle)]])


def RotationY(angle):
    return np.array([[cos(angle), 0., sin(angle)], [0., 1., 0.], [-sin(angle), 0., cos(angle)]])


def RotationZ(angle):
    return np.array([[cos(angle), -sin(angle), 0.], [sin(angle), cos(angle), 0.], [0., 0., 1.]])

# Computing $H_0$ for the ellipsoid


def H_0_ellipsoid(m_0, u, gamma=np.array([0., 0., 0.])):
    return np.array([-gamma[0]*m_0[0] + u[0], -gamma[1]*m_0[1]+u[1], -gamma[2]*m_0[2]+u[2]])

# The vectorial product of R^3


def wedge(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

# The Landau Lifshitz Gilbert equation with magnetisation m and effective magnetic field H_eff


def LandauLifshitzGilbert(m, H_eff, gamma=1., damping=0.):
    return -gamma*np.cross(m, H_eff)-gamma*damping*np.cross(m, np.cross(m, H_eff))


# Class that contains the information relative to 1 nano-particle
class NanoParticule:
    def __init__(self):
        self.M = 1.  # Norm of the Super Spin
        # Super-spin of the nano-particle at current time and previous time
        self.m = np.array([[1., 0., 0.], [1., 0., 0.]])
        self.m_dot = np.array([0., 0., 0.])  # Velocity of the Super-spin

        # Position of the nano-particle at current time and previous time
        self.X = np.array([[0., 0., 0.], [0., 0., 0.]])

        # Magnetic field around the nanp-particle at current time and previous time
        self.B = np.array([[0., 0., 0.], [0., 0., 0.]])

        # velocity of the nano-particle at current time and previous time
        self.V = np.array([[0., 0., 0.], [0., 0., 0.]])
        # Angular velocity of the nano-particle at current time and previous time
        self.omega = np.array([[0., 0., 0.], [0., 0., 0.]])
        # Angular acceleration of the nano-particle
        self.omega_dot = np.array([0., 0., 0.])

        # The set of the neighbors of the nano-particle and the associated Laplace coefficient.
        self.Neighbors = set([])
        self.flag = False  # This flag is usefull for some recursive functions
        # The magnetic force exerced on the nano-particle.
        self.magnetic_force = np.array([[0., 0., 0.], [0., 0., 0.]])
        # The repulsive force between nano-particles.
        self.repulsive_force = np.array([[0., 0., 0.], [0., 0., 0.]])

        # The velocity of the fluid at the position of the nano-particle
        self.fluid_velocity = np.array([0., 0., 0.])
        # The vorticity of the fluid at the position of the nano-particle
        self.fluid_vorticity = np.array([0., 0., 0.])

    def UpdateNorm(self):  # Enforces the constraint |m| = M by updating M
        self.M = norm(self.m)

    # Update the position of the nano-particle
    def UpdatePosition(self, X_new=np.array([0., 0., 0.]), T=0):
        for i in range(3):
            self.X[(T+1) % 2][i] = X_new[i]

    # Update the position of the nano-particle
    def UpdateVelocity(self, V_new=np.array([0., 0., 0.])):
        # self.V = V_new
        for i in range(3):
            self.V[i] = V_new[i]

    # Update the external fluid_vorticity applied to the nano-particle
    def Update_fluid_vorticity(self, fluid_vorticity_new=np.array([0., 0., 0.])):
        for i in range(3):
            self.fluid_vorticity[i] = fluid_vorticity_new[i]

    # Update the external fluid_vorticity applied to the nano-particle
    def Update_fluid_velocity(self, fluid_velocity_new=np.array([0., 0., 0.])):
        for i in range(3):
            self.fluid_velocity[i] = fluid_velocity_new[i]

    # Update the anglular momentum of the nano-particle. If "use_delta" then the new omega is computed using omega dot
    def UpdateOmega(self, omega_dot_new=np.array([0., 0., 0.]), T=0, dt=1., use_delta=True, omega_new=np.array([0., 0., 0.])):
        for i in range(3):
            self.omega_dot[i] = omega_dot_new[i]
            if use_delta:
                self.omega[(T+1) % 2][i] = self.omega[T][i] + \
                    dt*omega_dot_new[i]
            else:
                self.omega[(T+1) % 2][i] = omega_new[i]

    # Update the magnetic momentum. If "use_delta", then the new momentum is computed using m dot new
    def UpdateMoment(self, m_dot_new=np.array([0., 0., 0.]), T=0, dt=1., use_delta=True, m_new=np.array([0., 0., 0.])):
        for i in range(3):
            self.m_dot[i] = m_dot_new[i]
            if use_delta:
                self.m[(T+1) % 2][i] = self.m[T][i]+dt*m_dot_new[i]
            else:
                self.m[(T+1) % 2][i] = m_new[i]

    # Enforces the constraint |m| = M by projecting m on the sphere of radius M
    def ProjectOnSphere(self, T=-1):
        if T == -1:
            norme = Norm(self.m[0])
            for i in range(3):
                self.m[0][i] = self.M*self.m[0][i]/norme
            norme = Norm(self.m[1])
            for i in range(3):
                self.m[1][i] = self.M*self.m[1][i]/norme
        else:
            norme = Norm(self.m[T])
            for i in range(3):
                self.m[T][i] = self.M*self.m[T][i]/norme

    # Update the magnetic field around the nano-particle
    def UpdateMagneticField(self, B_new=np.array([0., 0., 0.]), T_new=0):
        for i in range(3):
            self.B[T_new][i] = B_new[i]

    # Add a new contribution to the magnetic field
    def AddMagneticField(self, B_add=np.array([0., 0., 0.]), T_new=0):
        self.B[T_new] += B_add

    # Update the vector with the magnetic force at time T+dt
    def UpdateMagneticForce(self, magnetic_force_new=np.array([0., 0., 0.]), T_new=0):
        for i in range(3):
            self.magnetic_force[T_new][i] = magnetic_force_new[i]

    # Add a new contribution to the magnetic force at time T+dt
    def AddMagneticForce(self, magnetic_force_add=np.array([0., 0., 0.]), T_new=0):
        self.magnetic_force[T_new] += magnetic_force_add

    # Update the vector with the repulsive force at time T+dt
    def UpdateRepulsive(self, repulsive_force_new=np.array([0., 0., 0.]), T_new=0):
        for i in range(3):
            self.repulsive_force[T_new][i] = repulsive_force_new[i]

    # Add a new contribution to the repulsive force at time T+dt
    def AddRepulsive(self, repulsive_force_add=np.array([0., 0., 0.]), T_new=0):
        self.repulsive_force[T_new] += repulsive_force_add

    def Conservative(self, T=0):
        return self.magnetic_force[T]+self.repulsive_force[T]

    def AddNeighbor(self, i):  # Flags the nano-particle of index "i" as being a neighbor
        self.Neighbors.add(i)

    # Flags the nano-particle of index "i" as being not a neighbor
    def RemoveNeighbor(self, i):
        self.Neighbors.remove(i)

    def RemoveAllNeighborq(self):
        self.Neighbors = set([])


class Colloid:
    def __init__(self, NanoParticules=[]):
        self.NP = NanoParticules  # list of the nano-particles of the colloid
        self.Number = 0  # number of nano-particles in the colloid
        self.alpha = 1.  # collision exponent
        self.Radius = 1.  # radius of the nano-particles
        self.Repulse_constant = 1.  # strength of the repulsion between nano-particles
        self.magnetic_constant = 1.  # rescale the magnetic rotationnal effect
        self.magnetic_grad_constant = 1.  # rescale the magnetic gradient effect
        self.zeta_tr = 1.  # Viscosity coefficient in translation
        self.zeta_r = 1.  # Viscosity coefficient in rotation
        self.mass = 1.  # Mass of the nano-particles
        self.energy_magnetic = 0.  # Magnetic energy of the colloid
        self.energy_repulse = 0.  # Repulsive energy of the colloid
        self.energy_kinetic = 0.  # Kinetic energy of the colloid

    def AddNano(self, NewNano):  # Adds a new nano-particle to the colloid
        self.NP.append(NewNano)
        self.Number += 1

    # Declare the two nano-particles with respective numbers i and j as "neighbors"
    def AddNeighbor(self, i, j):
        if i < self.Number and j < self.Number and i >= 0 and j >= 0 and i != j:
            self.NP[i].AddNeighbor(j)
            self.NP[j].AddNeighbor(i)

    # Declare the two nano-particles with respective numbers i and j as "not neighbors"
    def RemoveNeighbor(self, i, j):
        if i < self.Number and j < self.Number and i >= 0 and j >= 0:
            self.NP[i].RemoveNeighbor(j)
            self.NP[i].RemoveNeighbor(i)

    # Mark as "Neighbors" all the Nano-particles which distance is below a threshold (this function is slow...)
    def UpdateNeighbors(self, dist):
        dist_square = dist*dist
        for i in range(self.Number):
            self.NP[i].Neighbors = set([])
        for i in range(self.Number):
            for j in range(i):
                if NormSquare(self.NP[i].X[0]-self.NP[j].X[0]) < dist_square:
                    self.AddNeighbor(i, j)

    def FlagAllParticles(self, boolean=True):
        for NP in self.NP:
            NP.flag = boolean

    # returns the list of all the connected components in the colloid
    def ConnectedComponents(self, Update_Neighbors=False, dist=1.):
        Components = []
        if Update_Neighbors:
            self.UpdateNeighbors(dist)
        self.FlagAllParticles(True)
        for i in range(self.Number):
            if self.NP[i].flag:
                ToExplore = [i]
                Explored = []
                while ToExplore != []:
                    j = ToExplore.pop(0)
                    if self.NP[j].flag:
                        Explored.append(j)
                        self.NP[j].flag = False
                        for k in self.NP[j].Neighbors:
                            ToExplore.append(k)
                Components.append(set(Explored))
        return Components

    # the T is an efficient way to store the values at time t and t-dt. If "T = 0" then the new value is stored at slot 0 and the new value at slot 1 (and vice versa if "T = 1").
    def VerletHeunStep(self, dt=1., T=0, H_ext_model=[''], Magnetism=True):
        dt_mass = dt/self.mass
        dt_mass_2 = dt_mass/2.
        dt_carre_mass = dt*dt/(2*self.mass)
        dt_2 = dt/2.
        # Coefficient for the computation of the influence of the viscosity
        coeff_viscosity_1 = self.zeta_tr*dt/(self.mass)

        dt_I = 5*dt/(2*self.mass*(self.Radius**2))
        dt_I_2 = dt_I/2

        T_new = (T+1) % 2

        # First step of Heun method (prediction):
        for NP in self.NP:
            # Prediction on the new angular velocity
            NP.UpdateOmega(np.cross(
                NP.m[T], NP.B[T])-self.zeta_r*(NP.omega[T]-NP.fluid_vorticity), T, dt_I)
            # Prediction on the new magnetic moment
            NP.UpdateMoment(np.cross(NP.omega[T], NP.m[T]), T, dt)

        # First step of the velocity verlet method (computation of the position):
        for NP in self.NP:
            NP.X[T_new] = NP.X[T] + dt*NP.V[T] + dt_carre_mass*NP.Conservative(T)

        # Computation of the prediction of conservative forces and fluid_vorticity:
        self.UpdateConservative(T_new, True, H_ext_model, Magnetism)

        # Second step of Heun method (correction):
        for NP in self.NP:
            # Corrective step : the average between prediction and evaluation at the predicted step
            NP.omega[T_new] = NP.omega[T]+dt_I_2*(NP.omega_dot+np.cross(NP.m[T_new], NP.B[T])-self.zeta_r*(NP.omega[T_new]-NP.fluid_vorticity))
            # Corrective step : the average between prediction and evaluation at the predicted step
            NP.m[T_new] = NP.m[T]+dt_2 *(NP.m_dot+np.cross(NP.omega[T_new], NP.m[T_new]))

        # Preservation of the magnetic moment
        for NP in self.NP:
            NP.ProjectOnSphere(T_new)

        # Computation of the conservative forces and fluid_vorticity after correction:
        self.UpdateConservative(T_new, False, H_ext_model, Magnetism)

        # Second step of the velocity verlet method (computation of the velocity):
        for NP in self.NP:
            NP.V[T_new] = (NP.V[T]+dt_mass_2*(NP.Conservative(T)+NP.Conservative(T_new))+coeff_viscosity_1*NP.fluid_velocity)/(1+coeff_viscosity_1)

        # Computation of the kinetic energy:
        if Compute_energy:
            self.UpdateKinetic(T_new)
            
        # To treat the case of planar dynamics:    
        if Planar_Dynamics:
            C.PlanarConfig()
            
            
    def AddExternalThermalEffect(self, dt=1., T=0, alea_param_tr=0, alea_param_r=0): #It is possible to model the thermal effects using and external white noise
        T_new = (T+1) % 2
        sq_dt = sqrt(dt)
        for NP in self.NP:
            NP.V[T_new] += sq_dt*alea_param_tr*np.random.randn(3)
            NP.omega[T_new] += sq_dt*alea_param_r*np.random.randn(3)
            
    def UpdateKinetic(self, T_new=0):
        self.energy_kinetic = 0.
        mass_2 = self.mass/2
        mass_5 = self.mass/5
        R_2 = pow(C.Radius, 2)
        for i in range(self.Number):
            self.energy_kinetic += mass_2*NormSquare(self.NP[i].V[T_new])+mass_5*NormSquare(self.NP[i].omega[T_new])*R_2
        return self.energy_kinetic    
            
    # Sets all the conservative quantities to $0$
    def UpdateConservative_init(self, T_new=0, update_repu=True):
        for NP in self.NP:
            # Add the new external magnetic field
            NP.UpdateMagneticField(np.array([0., 0., 0.]), T_new)
            # Add the new associated magnetic force
            NP.UpdateMagneticForce(np.array([0., 0., 0.]), T_new)
            if update_repu:
                # Initialize the new repulsive force
                NP.UpdateRepulsive(np.array([0., 0., 0.]), T_new)

        if Compute_energy and update_repu:
            self.energy_magnetic = 0.
            self.energy_repulse = 0.
    
    # Computation of the new conservative quantities for the self-interactions of the system      
    def UpdateConservative_self(self, T_new=0, update_repu=True, Magnetism=True):
        
        for i in range(self.Number):  # Self interaction of the system
            for j in range(i):
                vector_ij = self.NP[j].X[T_new]-self.NP[i].X[T_new]  # x_j-x_i
                norme_2_ij = NormSquare(vector_ij)  # |x_j-x_i|^2
                norme_ij = sqrt(norme_2_ij)
                norme_3_ij = norme_ij**3
                u_ij = vector_ij/norme_ij  # vecteur x_j-x_i normalisé

                if Magnetism:                   


                    # Magnetic field generated by the dipole i on the dipole j and conversely (with F_ij the associated force). We omitted temporously the factor mu_0/4*pi
                    H_ij = self.magnetic_constant*(3*np.dot(self.NP[i].m[T_new],u_ij)*u_ij-self.NP[i].m[T_new])/norme_3_ij  # Magnetic field generated by the dipole i on the dipole j and conversely (with F_ij the associated force). We omitted temporously the factor mu_0/4*pi
                    H_ji = self.magnetic_constant*(3*np.dot(self.NP[j].m[T_new],u_ij)*u_ij-self.NP[j].m[T_new])/norme_3_ij
                    F_ij = 3*self.magnetic_grad_constant*(5*np.dot(self.NP[i].m[T_new], u_ij)*np.dot(self.NP[j].m[T_new], u_ij)*u_ij-np.dot(self.NP[i].m[T_new], u_ij)*self.NP[j].m[T_new]-np.dot(self.NP[j].m[T_new], u_ij)*self.NP[i].m[T_new]-np.dot(self.NP[i].m[T_new], self.NP[j].m[T_new])*u_ij)/(norme_2_ij**2)
                    global t
    
                    # Add the magnetic influence of the particle i on the particle j and conversely
                    self.NP[j].AddMagneticField(H_ij, T_new)
                    self.NP[i].AddMagneticField(H_ji, T_new)
                    self.NP[i].AddMagneticForce(F_ij, T_new)
                    self.NP[j].AddMagneticForce(-F_ij, T_new)

                if update_repu:  # Add the repulsive influence of the particle i of the particle j.
                    repulse_force = (pow(self.Radius/norme_ij, self.alpha-1))*(self.alpha*self.Radius*self.Repulse_constant/norme_3_ij)*vector_ij
                    self.NP[i].AddRepulsive(-repulse_force, T_new)
                    self.NP[j].AddRepulsive(repulse_force, T_new)
                    if Compute_energy:
                        self.energy_magnetic += C.magnetic_grad_constant*((np.dot(self.NP[i].m[T_new], self.NP[j].m[T_new]))-3*np.dot(self.NP[i].m[T_new], u_ij)*np.dot(self.NP[j].m[T_new], u_ij))/norme_3_ij
                        self.energy_repulse += C.Repulse_constant*pow(self.Radius/norme_ij, self.alpha)
                        
                        
    # Update the influence of the external environement                   
    def UpdateConservative_ext(self, T_new=0, update_repu=True, H_ext_model=['']): 
        for i in range(self.Number):  # Interaction of the system with its environement
            # influence of the external field on nano-particle i
            Ext_i = self.external_influence(self.NP[i].X[T_new], t, H_ext_model)
            self.NP[i].fluid_vorticity = np.array([0., 0., 0.])
            self.NP[i].fluid_velocity = np.array([0., 0., 0.])
            # Add the magnetic influence of the particle i on the particle j and conversely
            self.NP[i].AddMagneticField(Ext_i[0], T_new)
            self.NP[i].AddMagneticForce(Ext_i[1], T_new)
            self.NP[i].Update_fluid_vorticity(Ext_i[2])
            self.NP[i].Update_fluid_velocity(Ext_i[3])

    # Update the magnetic field and the magnetic forces at time T+dt. If "repu==True" we also update the repulsive forces
    def UpdateConservative(self, T_new=0, update_repu=True, H_ext_model=[''], Magnetism=True):
        
        self.UpdateConservative_init(T_new, update_repu)
        self.UpdateConservative_self(T_new, update_repu, Magnetism)
        self.UpdateConservative_ext(T_new, update_repu, H_ext_model)


    # returns the influence of the external magnetic field (depending of the chosen model) and on the temperature.
    def external_influence(self, X=np.array([0., 0., 0.]), t=0., H_ext_model=['']):
        if len(H_ext_model) == 0:
            # The returned quantity is an array containing the value at point X of: 
            # [0] The external magnetic field, 
            # [1] The external force, 
            # [2] The external fluid_vorticity, 
            # [3] The external velocity, 
            # [4] Dipolar interaction pounderation (=1 usually, but=0 to remove the dipolar interaction) 
            return np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        elif H_ext_model[0] == 'alea':  # returns a random thermal forcing in the magnetic field
            return np.array([[0., 0., 0.], H_ext_model[1]*np.random.randn(3), H_ext_model[2]*np.random.randn(3), [0., 0., 0.]])
        elif H_ext_model[0] == 'uniform': # returns a constant uniform magnetic field
            return np.array([H_ext_model[1], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        elif H_ext_model[0] == 'slope': # returns a uniform magnetic field with increasing norm
            fator = max(t/((Frames-1)*dt*(NumberSubSteps*2+1)*(1-H_ext_model[2]))-1/(1-H_ext_model[2])+1,0.) #fator means "multiplicative factor"
            return np.array([[H_ext_model[1][0]*fator,H_ext_model[1][1]*fator,H_ext_model[1][2]*fator], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        # returns a sinusoidal travelling wave
        elif H_ext_model[0] == 'traveling':
            return np.array([H_ext_model[3]*sin(H_ext_model[1][1]*t-np.dot(H_ext_model[1][2], X)), H_ext_model[4]*cos(H_ext_model[1][1]*t-np.dot(H_ext_model[1][2], X)), [0., 0., 0.], [0., 0., 0.]])
        elif H_ext_model[0] == 'vortex':
            Nsq = X[0]**2+X[1]**2
            if Nsq >= H_ext_model[1]:
                return np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], H_ext_model[2]*np.array([-X[1], X[0], 0.])/Nsq])
            else:
                return np.array([[0., 0., 0.], [0., 0., 0.], H_ext_model[2]*np.array([0., 0., 1.]), H_ext_model[2]*np.array([-X[1], X[0], 0.])/H_ext_model[1]])
        else:
            return np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])


    def diameter_square(self):
        maximum = 0.
        for i in range(self.Number):
            for j in range(i):
                maximum = max(maximum, NormSquare(self.NP[i].X[0]-self.NP[j].X[0]))
        return maximum

    def smallest_dist_square(self):
        minimum = NormSquare(self.NP[0].X[0]-self.NP[1].X[0])
        for i in range(self.Number):
            for j in range(i):
                minimum = min(minimum, NormSquare(
                    self.NP[i].X[0]-self.NP[j].X[0]))
        return minimum

    # print the colloid inside a window
    def Show(self, arrows=True, nano=True, center=np.array([0., 0., 0.]), window=1., ball_size=1., arrow_size=1., title_plot='', save_title=''):
        ax = plt.figure().add_subplot(projection='3d')
        #ax.set(xlim = (0,5), ylim = (0,5), zlim = (0,5))
        # Make the grid
        x = []
        y = []
        z = []
        # Make the direction data for the arrows
        u = []
        v = []
        w = []
        # plotted particles
        neighb = []

        for i in range(self.Number):
            x.append(self.NP[i].X[0][0])
            y.append(self.NP[i].X[0][1])
            z.append(self.NP[i].X[0][2])
            u.append(self.NP[i].m[0][0]/self.NP[i].M)
            v.append(self.NP[i].m[0][1]/self.NP[i].M)
            w.append(self.NP[i].m[0][2]/self.NP[i].M)

        if arrows:
            ax.quiver(x, y, z, u, v, w, length=arrow_size, pivot='middle', color = 'darkred')

        if nano:
            ax.scatter(x, y, z, s=ball_size, color='blue')

        ax.set(xlim=(center[0]-window, center[0]+window), ylim=(center[1]-window,
               center[1]+window), zlim=(center[2]-window, center[2]+window), title=title_plot)
        if save_title != '':
            plt.savefig(save_title, format='png')
        plt.show()

    # Builds the movie of the dynamics of particles
    def Update(self, T, arrows=True, nano=True):
        #ax = plt.figure().add_subplot(projection = '3d', animated = True)
        # print(ax)
        #ax.set(xlim = (0,5), ylim = (0,5), zlim = (0,5))
        # Make the grid
        ax.cla()
        global NumberStep
        global NumberSubSteps
        NumberFrames = NumberStep//NumberSubSteps
        if T != 0 and NumberFrames >= 100 and T % (NumberFrames//100) == 0:
            print((100*T)//(NumberFrames), '%, ', sep='', end='')

        x = []
        y = []
        z = []
        # Make the direction data for the arrows
        u = []
        v = []
        w = []
        # plotted particles
        neighb = []

        for i in range(self.Number):
            x.append(self.NP[i].X[0][0])
            y.append(self.NP[i].X[0][1])
            z.append(self.NP[i].X[0][2])
            u.append(self.NP[i].m[0][0]/self.NP[i].M)
            v.append(self.NP[i].m[0][1]/self.NP[i].M)
            w.append(self.NP[i].m[0][2]/self.NP[i].M)

        if arrows:
            global arrowsize
            ax.quiver(x, y, z, u, v, w, length=arrowsize, pivot='middle', color = 'darkred')

        if nano:
            global ballsize
            ax.scatter(x, y, z, s=ballsize, color='blue')

        global center
        ax.set(xlim=(center[0]-window, center[0]+window), ylim=(center[1] -
               window, center[1]+window), zlim=(center[2]-window, center[2]+window))

        if T != 0:
            for i in range(NumberSubSteps*2+1):
                global alea_param_r
                global alea_param_tr
                self.VerletHeunStep(dt, (T+i) % 2, external_interaction)
                global t
                t += dt

    # Builds the movie of the dynamics of particles
    def Update_static(self, T, arrows=True, nano=True):
        ax.cla()
        x = []
        y = []
        z = []
        # Make the direction data for the arrows
        u = []
        v = []
        w = []
        # plotted particles
        neighb = []

        if T != 0 and T % 6 == 0:
            print((100*T)//(600), '%, ', sep='', end='')

        global center
        # if T<=100:
        for i in range(self.Number):
            X = np.array([0., 0., 0.])
            M = np.array([0., 0., 0.])
            if T <= 200:
                X = np.dot(RotationX(2*T*pi/200), self.NP[i].X[0]-center)
                M = np.dot(RotationX(2*T*pi/200),
                           self.NP[i].m[0]/self.NP[i].M)
            else:
                if T <= 400:
                    X = np.dot(RotationY(2*T*pi/200),
                               self.NP[i].X[0]-center)
                    M = np.dot(RotationY(2*T*pi/200),
                               self.NP[i].m[0]/self.NP[i].M)
                else:
                    X = np.dot(RotationZ(2*T*pi/200),
                               self.NP[i].X[0]-center)
                    M = np.dot(RotationZ(2*T*pi/200),
                               self.NP[i].m[0]/self.NP[i].M)
            x.append(X[0])
            y.append(X[1])
            z.append(X[2])
            u.append(M[0])
            v.append(M[1])
            w.append(M[2])
        if arrows:
            global arrowsize
            ax.quiver(x, y, z, u, v, w, length=arrowsize, pivot='middle', color = 'darkred')

        if nano:
            global ballsize
            ax.scatter(x, y, z, s=ballsize, color='blue')

        ax.set(xlim=(-window, window),
               ylim=(-window, window), zlim=(-window, window))

    # Creates a colloid with cubic structure (+ alea) and random magnetization
    def CubicCristal(self, K=1, M=1, N=1, N_total=1, dist=1., alea=0., m=1., sometimes_nothing=0.):
        self.NP = []
        self.Number = 0
        R = np.random.rand(K, M, N, 2)
        D = np.random.randn(K, M, N, 3)
        for i in range(K):
            for j in range(M):
                for k in range(N):
                    Nano = NanoParticule()
                    Nano.M = m
                    Nano.X[0] = dist * \
                        np.array([i, j, k])+dist*alea*D[i, j, k]
                    Nano.X[1] = dist * \
                        np.array([i, j, k])+dist*alea*D[i, j, k]
                    theta = acos(1-2*R[i, j, k, 0])
                    phi = 2*pi*R[i, j, k, 1]
                    Nano.m[0] = np.array(
                        [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])
                    Nano.m[1] = np.array(
                        [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)])
                    Nano.ProjectOnSphere()
                    if(np.random.rand()>sometimes_nothing):
                        self.AddNano(Nano)
                    if self.Number >= N_total:
                        return
                
    def PlanarConfig(self, a=np.array([0., 0., 1.])): #Projects the position  of the nano-particles on the plane orthogonal to the given vector
        norme_square_a = NormSquare(a)
        for NP in self.NP:
            NP.X[0]=NP.X[0]-(np.dot(NP.X[0],a)/norme_square_a)*a
            NP.X[1]=NP.X[1]-(np.dot(NP.X[1],a)/norme_square_a)*a
    
    # Creates a colloid with ring structure (+ alea)
    def NanoRing(self, N=2, Radius=1., alea_X=0., alea_M=0., m=1.):
        self.NP = []
        self.Number = 0
        R = np.random.rand(N, 2)
        D = np.random.randn(N, 3)
        RingRadius = Radius/(2*sin(pi/N))
        for i in range(N):
            Nano = NanoParticule()
            Nano.M = m
            angle = 2*i*pi/N
            Nano.X[0] = RingRadius * \
                np.array([cos(angle), sin(angle), 0.]) + \
                RingRadius*alea_X*D[i]
            Nano.X[1] = RingRadius * \
                np.array([cos(angle), sin(angle), 0.]) + \
                RingRadius*alea_X*D[i]
            theta = alea_M*acos(1-2*R[i, 0])+(1-alea_M)*pi/2
            phi = alea_M*2*pi*R[i, 1]+(1-alea_M)*angle
            if alea_M >= 1.000001:
                print("Warning in NanoRing : alea_M must be lower than 1")
            Nano.m[0] = np.array(
                [-sin(theta)*sin(phi), sin(theta)*cos(phi), cos(theta)])
            Nano.m[1] = np.array(
                [-sin(theta)*sin(phi), sin(theta)*cos(phi), cos(theta)])
            Nano.ProjectOnSphere()
            self.AddNano(Nano)

    # Creates a colloid with a spear structure (+alea)
    def NanoSpear(self, N=1, Rad=1., alea_X=0., alea_M=0., m=1.):
        self.NP = []
        self.Number = 0
        R = np.random.rand(N, 2)
        D = np.random.randn(N, 3)
        for i in range(N):
            Nano = NanoParticule()
            Nano.M = m
            Nano.X[0] = np.array([0., i*Rad, 0.])+Rad*alea_X*D[i]
            Nano.X[1] = np.array([0., i*Rad, 0.])+Rad*alea_X*D[i]
            theta = alea_M*acos(1-2*R[i, 0])+(1-alea_M)*pi/2
            phi = alea_M*2*pi*R[i, 1]
            if alea_M >= 1.000001:
                print("Warning in NanoSpear : alea_M must be lower than 1")
            Nano.m[0] = np.array(
                [-sin(theta)*sin(phi), sin(theta)*cos(phi), cos(theta)])
            Nano.m[1] = np.array(
                [-sin(theta)*sin(phi), sin(theta)*cos(phi), cos(theta)])
            Nano.ProjectOnSphere()
            self.AddNano(Nano)

    # Mesures the total magnetization vector of a given subset
    def MeasureMagnetization(self, subset=[]):
        if subset == []:
            subset = range(self.Number)
        measure = np.array([0., 0., 0.])
        n = len(subset)
        for i in subset:
            measure += self.NP[i].m[0]
        return measure

    def Centroid(self, subset=[]):  # Computes the centroid of a given subset
        if subset == []:
            subset = range(self.Number)
        Xbar = np.array([0., 0., 0.])
        for i in subset:
            Xbar += self.NP[i].X[0]
        return Xbar/len(subset)

    # Computes the mean distance to the centroid for a given subset
    def RadiusMean(self, subset=[], Is_Xbar_known=False, Xbar=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
        Rbar = 0.
        for i in subset:
            Rbar += Norm(self.NP[i].X[0]-Xbar)
        return Rbar/len(subset)

    # Computes the standard deviation of the distance to the centroid for a given subset
    def RadiusDev(self, subset=[], Is_Xbar_known=False, Is_Rbar_known=False, Xbar=np.array([0., 0., 0.]), Rbar=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
            Is_Xbar_known = True
        if not Is_Rbar_known:
            Rbar = self.RadiusMean(subset, Is_Xbar_known, Xbar)
        Rdev = 0.
        for i in subset:
            Rdev += (Norm(self.NP[i].X[0]-Xbar)-Rbar)**2
        return sqrt(Rdev/(len(subset)))

    # Computes the standard deviation of the cosinus of the angle with respect to the Ring structure
    def RingDevSquareXX(self, subset=[], Is_Xbar_known=False, Xbar=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
        CosAngles = []
        n = len(subset)
        i0 = subset[randrange(0, n)]
        norme2i0 = NormSquare(self.NP[i0].X[0]-Xbar)
        for i in subset:
            if i != i0:
                CosAngles.append(((np.dot(self.NP[i].X[0]-Xbar, self.NP[i0].X[0]-Xbar)))/sqrt(
                    NormSquare(self.NP[i].X[0]-Xbar)*norme2i0))
        CosAngles.sort(reverse=True)
        return NormeSquare(np.array(CosAngles-ReturnDictCosine(n)))/(n)

    def RingDevSquareMM(self):  # Identical to RingDevSquare but for magnetic spin
        CosAngles = []
        n = len(subset)
        i0 = subset[randrange(0, n)]
        for i in subset:
            if i != i0:
                CosAngles.append(
                    ((np.dot(self.NP[i].m[0], self.NP[i0].m[0])))/(self.NP[i].M**2))
        CosAngles.sort(reverse=True)
        return NormeSquare(np.array(CosAngles-ReturnDictCosine(n)))/(n-1)

    # Computes the deviation from alpha of the cosinus of angle between the spin and the position
    def RingDevSquareMX(self, subset=[], alpha=0., Is_Xbar_known=False, Xbar=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
        CosAngles = []
        n = len(subset)
        for i in subset:
            CosAngles.append(((np.dot(self.NP[i].X[0]-Xbar, self.NP[i].m[0])))/sqrt(
                NormSquare(self.NP[i].X[0]-Xbar)*self.NP[i].M))
        return NormeSquare(np.array(CosAngles)-alpha)/(n)

    # Computes main direction of the cloud of points
    def MainDirection(self, subset=[], Is_Xbar_known=False, Xbar=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
        MainDir = np.array([0., 0., 0.])
        RefVector = np.array([0., 0., 0.])
        seuil = 0.001*(C.Radius**2)
        for i in subset:
            if NormSquare(self.NP[i].X[0]-Xbar) >= seuil:
                RefVector = self.NP[i].X[0]-Xbar
                break
        for i in subset:
            if np.dot(self.NP[i].X[0]-Xbar, RefVector) >= 0:
                MainDir += (self.NP[i].X[0]-Xbar)
            else:
                MainDir += -(self.NP[i].X[0]-Xbar)
        return MainDir/Norm(MainDir)

    # Computes the direction that is the most orthogonal to the cloud of points
    def MainPlaneDirection(self, subset=[], Is_Xbar_known=False, Xbar=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
        MainDir = np.array([0., 0., 0.])
        RefVector = np.array([1., 0., 0.])
        seuil = 0.0001*(C.Radius**2)
        for i in subset:
            for j in subset:
                candidate = np.cross(
                    self.NP[i].X[0]-Xbar, self.NP[j].X[0]-Xbar)
                if NormSquare(candidate) >= seuil:
                    RefVector = candidate
                    break
            else:
                continue
            break

        for i in subset:
            for j in subset:
                if j < i:
                    truc = np.cross(self.NP[i].X[0]-Xbar,
                                 self.NP[j].X[0]-Xbar)
                    if np.dot(truc, RefVector) >= 0:
                        MainDir += truc
                    else:
                        MainDir += -truc
        nor = Norm(MainDir)
        if nor >= seuil:
            return MainDir/nor
        else:
            return np.array([1., 0., 0.])

    # computes the standard deviation with respect to the situation where all the particles are planar.
    def RingDevPlane(self, subset=[], Is_Xbar_known=False, Is_MainDir_known=False, Xbar=np.array([0., 0., 0.]), MainDir=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
            Is_Xbar_known = True
        if not Is_MainDir_known:
            MainDir = self.MainPlaneDirection(subset, Is_Xbar_known, Xbar)

        Dev = 0.
        total = 0
        seuil = 0.00001*(C.Radius**2)
        for i in subset:
            # One must remove the degenerate case where this norm is almost zero
            Nsq = NormSquare(self.NP[i].X[0]-Xbar)
            if Nsq >= seuil:  # One must remove the degenerate case where this norm is almost zero
                total += 1
                Dev += 1-(np.dot(MainDir, self.NP[i].X[0]-Xbar)**2)/Nsq
        return Dev/(total)

    # computes the standard deviation with respect to the situation where all the spins are orthogonal to the radial vector
    def RingDevMagnetic(self, subset=[], Is_Xbar_known=False, Xbar=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
            Is_Xbar_known = True

        Dev = 0.
        total = 0
        seuil = 0.00001*(C.Radius**2)

        for i in subset:
            # One must remove the degenerate case where this norm is almost zero
            Nsq = NormSquare(self.NP[i].X[0]-Xbar)
            Msq = (self.NP[i].M)**2
            if Nsq >= seuil:  # One must remove the degenerate case where this norm is almost zero
                total += 1
                Dev += 1 - \
                    (np.dot(self.NP[i].m[0],
                     self.NP[i].X[0]-Xbar)**2)/(Nsq*Msq)
        return Dev/(total)

    # Computes the standard deviation of the cosinus of the angle with respect to the Spear structure (similar to RingDevSquareXX for the Spear)
    def SpearDevSquareX(self, subset=[], Is_Xbar_known=False, Is_MainDir_known=False, Xbar=np.array([0., 0., 0.]), MainDir=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
            Is_Xbar_known = True
        if not Is_MainDir_known:
            MainDir = self.MainDirection(subset, Is_Xbar_known, Xbar)
        Dev = 0.
        total = 0
        seuil = 0.00001*(C.Radius**2)
        for i in subset:
            # One must remove the degenerate case where this norm is almost zero
            Nsq = NormSquare(self.NP[i].X[0]-Xbar)
            if Nsq >= seuil:  # One must remove the degenerate case where this norm is almost zero
                total += 1
                Dev += (np.dot(MainDir, self.NP[i].X[0]-Xbar)**2)/Nsq
        return Dev/(total)

    # Similar to SpearDevSquareX but for the spin
    def SpearDevSquareM(self, subset=[], Is_Xbar_known=False, Is_MainDir_known=False, Xbar=np.array([0., 0., 0.]), MainDir=np.array([0., 0., 0.])):
        if subset == []:
            subset = range(self.Number)
        if not Is_Xbar_known:
            Xbar = self.Centroid(subset)
            Is_Xbar_known = True
        if not Is_MainDir_known:
            MainDir = self.MainDirection(subset, Is_Xbar_known, Xbar)
        Dev = 0.
        for i in subset:
            Dev += (np.dot(MainDir, self.NP[i].m[0])
                    ** 2)/NormSquare(self.NP[i].m[0])
        return Dev/(len(subset))

    # Computes the magnetic energy of the colloid or of a subset of the colloid
    def MagneticEnergy(self, subset=[]):
        if subset == []:
            subset = range(self.Number)
        energy = 0.
        for i in range(self.Number):
            for j in range(self.Number):
                if i != j:
                    vector_ij = self.NP[j].X[0]-self.NP[i].X[0]
                    norm_ij_2 = NormSquare(vector_ij)
                    norm_ij_3 = pow(norm_ij_2, 3/2)
                    energy += self.magnetic_constant*(np.dot(self.NP[i].m[0], self.NP[j].m[0])-3*np.dot(
                        self.NP[i].m[0], vector_ij)*np.dot(vector_ij, self.NP[j].m[0])/norm_ij_2)/(norm_ij_3)
        return energy

    # Computes the repulsion energy of the colloid or of a subset of the colloid
    def RepulseEnergy(self, subset=[]):
        if subset == []:
            subset = range(self.Number)
        energy = 0.
        for i in range(self.Number):
            for j in range(self.Number):
                if i != j:
                    vector_ij = self.NP[j].X[0]-self.NP[i].X[0]
                    norm_ij_2 = NormSquare(vector_ij)
                    norm_ij = sqrt(norm_ij_2)
                    energy += C.Repulse_constant * \
                        pow(self.Radius/norm_ij, self.alpha)
        return energy

    # Returns True iff the nano-particles have the shape of a ring
    def IsRing(self, subset=[], threashold=0.8):
        return (C.RingDevPlane(subset)*max(0., 1-C.RadiusDev(subset))*C.RingDevMagnetic(subset) >= threashold)

    # Returns True iff the nano-particles have the shape of a spead
    def IsSpear(self, subset=[], threashold=0.8):
        return (C.SpearDevSquareX(subset)*C.SpearDevSquareM(subset) >= threashold)
    
    # Saves all data at the current time of the simulation
    def SaveAllData(self, title_p='', T=0, t=0):
        with open(title_p+'.dat', 'a') as file:
            file.write("\n")
            file.write("%f " % t)
            for i in range(len(self.NP)):
                file.write("%i " % i)
                file.write("%f %f %f " % (self.NP[i].m[T][0],self.NP[i].m[T][1],self.NP[i].m[T][2]))
                file.write("%f %f %f " % (self.NP[i].m_dot[0],self.NP[i].m_dot[1],self.NP[i].m_dot[2]))
                file.write("%f %f %f " % (self.NP[i].X[T][0],self.NP[i].X[T][1],self.NP[i].X[T][2]))
                file.write("%f %f %f " % (self.NP[i].B[T][0],self.NP[i].B[T][1],self.NP[i].B[T][2]))
                file.write("%f %f %f " % (self.NP[i].V[T][0],self.NP[i].V[T][1],self.NP[i].V[T][2]))
                file.write("%f %f %f " % (self.NP[i].omega[T][0],self.NP[i].omega[T][1],self.NP[i].omega[T][2]))
                file.write("%f %f %f " % (self.NP[i].omega_dot[0],self.NP[i].omega_dot[1],self.NP[i].omega_dot[2]))
            
            
                       
            
            
            
            
########################################################

# sys.exit()


k_B = 1.380649e-23  # Boltzmann constant
Temp = 500 # Temperature in Kelvin (It is possible to model the thermal effect either by a white noise in the acceleration of the particle or by a white noise in the magnetic field)
mu_0 = 1.2566370621219e-6  # Vacuum magnetic permeability
mu_B = 9.27400949e-24  # Bohr Magneton
eta = 3.e-3  # Viscosité cinématique du liquide


C = Colloid()  # definition of the colloid


# The real value of the physical quanties:
mu_s = 22000*mu_B  # Spin intensity of the nano-particles
C.alpha = 11.  # collision exponent
C.Radius = 6.e-9  # radius of the nano-particles
# strength of the repulsion between nano-particles
C.Repulse_constant = mu_0*mu_s*mu_s/((C.Radius**3))
# rescale the magnetic rotationnal effect
C.magnetic_constant = mu_0/(4*pi)
# rescale the magnetic gradient effect
C.magnetic_grad_constant = mu_0/(4*pi)
C.zeta_tr = 6*pi*eta*C.Radius  # Viscosity coefficient in translation
C.zeta_r = 8*pi*eta*(C.Radius**3)  # Viscosity coefficient in rotation
C.mass = 8.9e-22  # Mass of one nano-particle


# The modified value of the physical quanties with new units :
mu_B = 9.27400949e-6
mu_0 = mu_0*1e12
eta = 1.e-3
mu_s = 22000*mu_B  # Spin intensity of the nano-particles
C.alpha = 14.  # collision exponent
C.Radius = 12.  # radius of the nano-particles
# strength of the repulsion between nano-particles
C.Repulse_constant = mu_0*mu_s*mu_s/((16*C.Radius**3))
# rescale the magnetic rotationnal effect
C.magnetic_constant = mu_0/(4*pi)
# rescale the magnetic gradient effect
C.magnetic_grad_constant = mu_0/(4*pi)
C.zeta_tr = 6*pi*eta*C.Radius  # Viscosity coefficient in translation
C.zeta_r = 8*pi*eta*(C.Radius**2)  # Viscosity coefficient in rotation
C.mass = 0.89  # Mass of one nano-particle (e^-21 kg)
k_B = 1.380649e-2 # Boltzmann constant in the new units




#N_ttal = N*M*K  # Number of particles in the colloid
#N_total = 20

parser = argparse.ArgumentParser(
    prog='Temperature_colloid',
    description='What the program does',
    epilog='Text at the bottom of help')
parser.add_argument('N_total', type=int, nargs='?', default=20,
                    help='Number of particles')
args = parser.parse_args()
N_total = args.N_total

Make_movie_dynamic = False  # Makes a movie of the dynamics of nano-particles
Make_movie_static = False  # Makes a movie of the final result

Compute_energy = True  # Computes the energy of the system
Do_statitstics = False # Computes statistical indicators
Study_Structures = True

Planar_Dynamics = False # To impose the nano-particles to stay in the plane z=0

alea_param_tr = sqrt(2*k_B*Temp*C.zeta_tr)/C.mass  # Simulation of the temperature
alea_param_r = 5*sqrt(2*k_B*Temp*C.zeta_r)/(2*C.mass*(C.Radius**2))

Vortex_Radius_2 = 100.
Vortex_maximal_velocity = 2.

Save_100_Images = True # Saves 100 pictures of the dynamics

# ['vortex',Vortex_Radius_2,Vortex_maximal_velocity] # models the interaction with the environment



Statistic = 1  # Number of occurrencies to make statistics

Stat_Number_Spear = 0
Stat_Number_Ring = 0
Stat_Number_Disconnected = 0
Stat_kinetic = 0
Stat_magnetic = 0
Stat_repulse = 0
Stat_mechanic = 0

SpearDevSquare_in_time = []
RingDevSquare_in_time = []
Gaz_in_time =[]

Energy_kinetic_in_time = []
Energy_magnetic_in_time = []
Energy_repulse_in_time = []
Energy_potential_in_time = []
Energy_mechanic_in_time = []

# sys.exit()
t = 0.

time = []
time_struct = []

N = 5
M = 5
K = 5

N_total = 125
#N_total=20

# 2^3=8, 3^3=27, 4^3 = 64, 5^3=125, 6^3=216, 7^3= 343

dt = 2.e-1
dt_max = 3.e-1
param = 2.
dist = param*C.Radius

seed = int(str(datetime.now())[20:])   #   #   #   #   #   #   #   #   #   #   #   #
np.random.seed(seed) #   #   #   #   #   #   #   #   #   #   #   #   #
print("Radom seed = ",seed)  #   #   #   #   #   #   #   #   #   #   #   #

C.CubicCristal(N, M, K, N_total, dist, 0., mu_s, 0.)

#C.NanoRing(N_total, dist, 0., 0., mu_s)
#C.NanoSpear(N_total, dist/1.5,0., 0., mu_s)

window = sqrt(C.diameter_square())/4
window = 90

truc = 2.55

ballsize = (1000/window)**2
arrowsize = 10

NumberStep = int(10000)
NumberSubSteps = NumberStep//300+1
Frames = NumberStep//NumberSubSteps
# For movies : Number frames = Number Step // NumberSubSteps

T_total = dt*(NumberStep+1)


print('Initialization...', end='')

for l in range(int(60)): # alea on the initial datum
    C.VerletHeunStep(dt, l % 2, ['alea', alea_param_tr, alea_param_r], False)

for NP in C.NP:
    NP.m_dot = np.array([0., 0., 0.])
    NP.V = np.array([[0., 0., 0.], [0., 0., 0.]])
    NP.omega_dot = np.array([0., 0., 0.])
    NP.omega = np.array([[0., 0., 0.], [0., 0., 0.]])
    NP.B = np.array([[0., 0., 0.], [0., 0., 0.]])
    NP.magnetic_force = np.array([[0., 0., 0.], [0., 0., 0.]])
    NP.repulsive_force = np.array([[0., 0., 0.], [0., 0., 0.]])
    
print(' done.')
    
center = C.Centroid()
C.Show(True, True, center, window, ballsize, arrowsize, 'initial configuration')

#sys.exit()

Temp=0
B=250
StartSlope = 0.5
 # At which moment do we put the magnetic field (0.5=in the middle, 1=never)
external_interaction = ['alea', alea_param_tr, alea_param_r] 
external_interaction = ['uniform',[0.,0.,B]]
external_interaction = ['slope', [0.,0,B], StartSlope]
#external_interaction = ['vortex', 6.,50.]
#external_interaction = []


date = str(datetime.now())


title_p=''

if title_p =='':
    title_p = date[:10]+'_'+date[11:13]+'.'+date[14:16]+'.'+date[17:19]+' (N='+str(C.Number)+', T='+str(Temp)+', B='+str(B)+')'

if Save_100_Images or Make_movie_dynamic or Make_movie_dynamic:
    try:
        os.mkdir(title_p)
        print(f"Directory '{title_p}' created successfully.")
    except FileExistsError:
        print(f"Directory '{title_p}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{title_p}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    title_p = title_p+'\\'+title_p

print('-------------------------------------------------------')
print('Colloidal suspension ready for numerical computations :')
print('-------------------------------------------------------')
print('Number of nanoparticles : ', C.Number)
print('Radius : ', C.Radius)
print('Repulsion constant : ', C.Repulse_constant)
print('Magnetic constant : ', C.magnetic_constant)
print('Magnetic force constant : ', C.magnetic_grad_constant)
print('Zeta translation : ', C.zeta_tr)
print('Zeta rotation : ', C.zeta_r)
print('Time step dt : ', dt)
print('Number of steps : ', NumberStep)
print('Temperature : ', Temp)
print('External magnetic field : ', B)
print('-------------------------------------------------------')
print('Start of the simulation : '+date[:19])
print('-------------------------------------------------------')

save_all_data = False

if save_all_data:
    with open(title_p+'.dat', 'a') as file:
        file.write("# "+title_p[:35]+"\n")
        file.write("time AND for all particles : index, magnetization, magnetization velocity, position, velocity, angular velocity, angular acceleration")
        
    if Compute_energy and save_all_data:
            with open(title_p+'_energies.dat', 'a') as file:
                file.write("# "+title_p[:35]+"\n")
                file.write("time AND kinetic energy, magnetic potential energy, magnetic repulsion energy\n")
                
    if Study_Structures and save_all_data:
        with open(title_p+'_structures.dat', 'a') as file:
            file.write("# "+title_p[:35]+"\n")
            file.write("time AND indicator structure spear, indictor structure ring, number of isolated particles\n")

if Make_movie_dynamic:
    print("Making a movie of the dynamics")
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(projection='3d')

    ani = FuncAnimation(fig=fig, func=C.Update,
                        frames=Frames, interval=1.)

    ani.save(title_p+'.gif')
    print()
    
else:
    NumberSubSteps = 0
    Frames = NumberStep
    for i in range(Statistic):
        if Do_statitstics:
            dt = 0.01
            C.CubicCristal(N, M, K, N_total, dist, 0., mu_s)
            print(i+1, '/', Statistic, 'N=', C.Number,' : ')
        l = 0
        t = 0.

        Energy_kinetic_in_time = []
        Energy_magnetic_in_time = []
        Energy_repulse_in_time = []
        Energy_mechanic_in_time = []

        time = []

        #C.CubicCristal(N,M,K, N_total, dist, 0.5, mu_s)

        while l < NumberStep:

            #C.VerletHeunStep(dt, l%2, ['alea',0.,0.])
            
            if save_all_data: #if we wan to save everything
                C.SaveAllData(title_p, l%2, l*dt)
            
            if Save_100_Images and NumberStep >= 100 and l % (NumberStep//100) == 0:
                q=(100*l)//(NumberStep)
                C.Show(True, True, center, window, ballsize, arrowsize, '', title_p+str(q)+'.png')
                #C.Show(True, True, center, window, ballsize, arrowsize, '', '')
                print(q,'/100 ', sep='', end='')  
                if False : # This block is for stability tests
                    if q == 10:
                        dt= dt_max/10        
                    if q ==20 :
                        dt= dt_max/5                 
                    if q == 30:
                        dt=dt_max
                
            else:
                if Compute_energy and NumberStep >= 1000 and l % (NumberStep//1000) == 0:
                    C.UpdateKinetic()
                    Energy_kinetic_in_time.append(C.energy_kinetic)
                    Energy_magnetic_in_time.append(C.energy_magnetic)
                    Energy_repulse_in_time.append(C.energy_repulse)
                    Energy_potential_in_time.append(C.energy_magnetic+C.energy_repulse)
                    Energy_mechanic_in_time.append(C.energy_kinetic + C.energy_magnetic + C.energy_repulse)
                    # print(C.SpearDevSquareM(),C.SpearDevSquareM())
                    time.append(t)
                    
                    if save_all_data:
                        with open(title_p+'_energies.dat', 'a') as file:
                            file.write("%f %f %f %f \n" % (t, C.energy_kinetic, C.energy_magnetic, C.energy_repulse))
                    
                if Study_Structures and NumberStep >= 1000 and l % (NumberStep//1000) == 0:
                    Conne = C.ConnectedComponents(True, 2.5*C.Radius)
                    Indic_spear = 0
                    Indic_ring = 0
                    max_size = 1
                    gaz = 0
                    for subset in Conne:
                        if len(subset)>max_size :
                            max_size=len(subset)
                            #Indic_ring = C.RingDevPlane(subset)*max(0., 1-C.RadiusDev(subset))*C.RingDevMagnetic(subset)
                            Indic_ring = C.RingDevPlane(subset)*C.RingDevMagnetic(subset)*max(0., 1-C.RadiusDev(subset)/10)
                            Indic_spear = Spread(C.SpearDevSquareX(subset)*C.SpearDevSquareM(subset),0.35,0.25,0.6,0.85)
                        if len(subset)==1:
                            gaz+=1
                    SpearDevSquare_in_time.append(Indic_spear)
                    RingDevSquare_in_time.append(Indic_ring)
                    Gaz_in_time.append(gaz)
                    time_struct.append(t)
                    
                    if save_all_data:
                        with open(title_p+'_structures.dat', 'a') as file:
                            file.write("%f %f %f %f \n" % (t, Indic_spear, Indic_ring, gaz))
                
            t += dt
            l += 1
            C.VerletHeunStep(dt, l % 2, external_interaction)  
            

            if (not Do_statitstics) and (not Save_100_Images) and NumberStep >= 100 and l % (NumberStep//100) == 0:
                #C.Show(True, True, center, window, ballsize, arrowsize)
                print((100*l)//(NumberStep),'%, ', sep='', end='')
                if Compute_energy:
                    print("Energy = ",C.MagneticEnergy()+C.RepulseEnergy())
                #print(C.ConnectedComponents(True, 2.5*C.Radius))
                
                #C.UpdateKinetic()
                #print(C.energy_kinetic, C.MagneticEnergy(), C.RepulseEnergy())
                # print(C.MagneticEnergy())
                
  

            if Do_statitstics and (not Save_100_Images) and NumberStep >= 10 and l % (NumberStep//10) == 0:
                #C.Show(True, True, center, window, ballsize, arrowsize)
                print('-', end='')
                # print(C.MagneticEnergy())
                
            if Do_statitstics and (not Save_100_Images) and l==NumberStep:
                print("dt=",dt, " kinetic=", C.UpdateKinetic())
                if C.energy_kinetic >= 5.e-4:
                    l=0
                    dt=min(0.11,dt*2)
                    #C.Show(True, True, center, window, ballsize, arrowsize, '', '')

        if Do_statitstics and (not Save_100_Images):
            print('> ', end='')
            Conne = C.ConnectedComponents(True, 1.5*sqrt(C.smallest_dist_square()))
            Spear = 0
            Ring = 0
            Gaz = 0
            Complex = 0
            for Com in Conne:
                if len(Com) <= 1:
                    Gaz += 1
                elif C.IsRing(Com):
                    Ring += 1
                elif C.IsSpear(Com):
                    Spear += 1
                else:
                    Complex += 1
            if (Spear, Ring, Gaz, Complex) in DictStructures:
                DictStructures[(Spear, Ring, Gaz, Complex)] += 1
            else:
                DictStructures[(Spear, Ring, Gaz, Complex)] = 1
            #PrintDictStructures(False)
            print((Spear, Ring, Gaz, Complex))

            if Compute_energy:
                Stat_kinetic += sum(Energy_kinetic_in_time)/(Statistic*len(Energy_kinetic_in_time))
                Stat_magnetic += sum(Energy_magnetic_in_time)/(Statistic*len(Energy_magnetic_in_time))
                Stat_repulse += sum(Energy_repulse_in_time)/(Statistic*len(Energy_repulse_in_time))

        if not Do_statitstics:
            if Compute_energy:
                length=2*len(Energy_kinetic_in_time)//3
                Stat_kinetic += sum(Energy_kinetic_in_time[length:])/(len(Energy_kinetic_in_time[length:]))
                Stat_magnetic += sum(Energy_magnetic_in_time[length:])/(len(Energy_magnetic_in_time[length:]))
                Stat_repulse += sum(Energy_repulse_in_time[length:])/(len(Energy_repulse_in_time[length:]))
                Stat_mechanic += sum(Energy_mechanic_in_time[length:])/(len(Energy_mechanic_in_time[length:]))
            break
        
        else:
            truc=1
            #C.Show(True, True, center, window, ballsize, arrowsize)


print()
print('-------------------------------------------------------')
print('End of the simulation : '+str(datetime.now())[:19])
print('-------------------------------------------------------')
#print('Mean Kinetic Energy = ', Stat_kinetic)
#print('Mean Potential Energy = ', Stat_magnetic+Stat_repulse)

#print(C.ConnectedComponents(True,1.5*sqrt(C.smallest_dist_square())))

# sys.exit()

if Make_movie_static:
    print("Making a movie of the final result")
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(projection='3d')

    ani = FuncAnimation(fig=fig, func=C.Update_static,
                        frames=600, interval=1.)

    ani.save(title_p+'_Static.gif')
    print()


#title_p='N='+str(N_total)+', R='+str(C.Radius)+', Rep='+str(C.Repulse_constant)+', M='+str(C.magnetic_constant)+', Mg='+str(C.magnetic_grad_constant)
C.Show(True, True, center, window, ballsize, arrowsize)
print('N=', C.Number)
#print(DictStructures)

#print("Energy = ",C.MagneticEnergy()+C.RepulseEnergy())
PrintDictStructures(True)
#print("Connected components :", len(C.ConnectedComponents(True,1.5*sqrt(C.smallest_dist_square()))))

sys.exit()

# print(m_in_time)

Gaz_fig, Gaz_ax = plt.subplots()
Gaz_ax.plot(time_struct, Gaz_in_time)
titre = ''
Gaz_ax.set(xlabel='time ', ylabel='Indicator Gaz',
              title=titre, xlim=(0., t), ylim=(0.,1.*N_total))
Gaz_ax.grid()
Gaz_fig.savefig('T='+str(T)+'_G')
plt.show()

k_size = 7
Kernel = np.ones(k_size)/k_size
for k in range(k_size-1):
    time.pop()
    time_struct.pop()

Energy_fig, Energy_ax = plt.subplots()
Energy_ax.plot(time, np.convolve(Energy_kinetic_in_time, Kernel, mode='valid'), label = 'E_c')
Energy_ax.plot(time, np.convolve(Energy_potential_in_time, Kernel, mode='valid'), label = 'E_p')
Energy_ax.plot(time, np.convolve(Energy_mechanic_in_time, Kernel, mode='valid'), label='E_tot')
titre = ''
Energy_ax.set(xlabel='time ', ylabel='Energy',
              title=titre, xlim=(0., t))
Energy_ax.grid()
Energy_ax.legend(loc='upper left')
Energy_fig.savefig('T='+str(T)+'_E')
plt.show()


Structure_fig, Structure_ax = plt.subplots()
Structure_ax.plot(time_struct, np.convolve(SpearDevSquare_in_time, Kernel, mode='valid'), label = 'Aligned')
Structure_ax.plot(time_struct, np.convolve(RingDevSquare_in_time, Kernel, mode='valid'), label = 'Ring')
titre = ''
Structure_ax.set(xlabel='time ', ylabel='Indicator Structure',
              title=titre, xlim=(0., t), ylim=(0.,1.))
Structure_ax.grid()
Structure_ax.legend(loc='upper left')
Structure_fig.savefig('T='+str(T)+'_S')
plt.show()


sys.exit()

SpearDevSquare_fig, SpearDevSquare_ax = plt.subplots()
SpearDevSquare_ax.plot(time, SpearDevSquare_in_time)
titre = ''
SpearDevSquare_ax.set(xlabel='time ', ylabel='SpearDevSquare',
                      title=titre, xlim=(0., t), ylim=(0., 1.1))
SpearDevSquare_ax.grid()
# fig.savefig("test.png")
plt.show()


RingDevSquare_fig, RingDevSquare_ax = plt.subplots()
RingDevSquare_ax.plot(time, RingDevSquare_in_time)
titre = ''
RingDevSquare_ax.set(xlabel='time ', ylabel='RingDevSquare',
                     title=titre, xlim=(0., t), ylim=(0., 1.1))
RingDevSquare_ax.grid()
# fig.savefig("test.png")
plt.show()
