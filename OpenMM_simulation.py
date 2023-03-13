import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import Bio.PDB
from Bio.PDB import PDBParser
import simtk.unit as unit
import openmm as mm
import openmm.app as app
from openmm.app import *
from openmm import *
from simtk import unit
from sys import stdout
import nglview as nv

# Load the PDB file
pdb = PDBFile(r'path\2zwh_fixed_pH_7.pdb')

k = 1.987e-3  # Boltzmann constant in kcal/mol/K
T = 300.0  # temperature in K
dt = 0.002  # integration timestep in ps
nsteps = 100000  # number of MD steps to run

# Define the force field
forcefield = app.ForceField('amber99sbildn.xml', 'tip3p.xml')

# Create the simulation system
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.2*unit.nanometers, constraints=app.HBonds)
box_vectors = np.diag([10, 10, 50]) * unit.nanometer
system.setDefaultPeriodicBoxVectors(*box_vectors)

# Add a Langevin thermostat
integrator = LangevinIntegrator(T*unit.kelvin, 1.0/unit.picosecond, dt*unit.picoseconds)
integrator.setRandomNumberSeed(42)

# Create the simulation object
platform = mm.Platform.getPlatformByName('CPU')
simulation = Simulation(pdb.topology, system, integrator, platform)

# Set the initial positions of the atoms
positions = pdb.getPositions()
simulation.context.setPositions(positions)

# Minimize the energy
print('Minimizing energy...')
#simulation.minimizeEnergy()
simulation.minimizeEnergy(tolerance=1.0*unit.kilojoule_per_mole, maxIterations=100)  #change it to 100 from 1000


# Equilibrate the system
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(T*unit.kelvin)
simulation.step(100000)

#Create the DCD reporter to save the trajectory data
report_interval = 1
reporter = DCDReporter(r'path\trajectory.dcd', report_interval)
simulation.reporters.append(reporter)



# Run the production simulation and calculate the total potential energy at each frame
print('Running production simulation...')
potential_energy = []
for i in range(nsteps):
    simulation.step(1)
    state = simulation.context.getState(getEnergy=True)
    potential_energy.append(state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole))


# Save the force data to a csv file
np.savetxt('potential_energy.csv', potential_energy, delimiter=',')

# Plot the potential energy data
plt.plot(range(nsteps), potential_energy)
plt.xlabel('Time (ps)')
plt.ylabel('Potential energy (kcal/mol)')
plt.show()
