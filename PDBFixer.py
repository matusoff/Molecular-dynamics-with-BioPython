
from pdbfixer import PDBFixer
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import os

def fix_pdb(pdb_id):
    pdb = PDBFile(pdb_id)
    if len(pdb_id) != 4:
        print("Creating PDBFixer...")
        fixer = PDBFixer(pdb_id)
        print("Finding missing residues...")
        fixer.findMissingResidues()

        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                print("ok")
                del fixer.missingResidues[key]

        print("Finding nonstandard residues...")
        fixer.findNonstandardResidues()
        print("Replacing nonstandard residues...")
        fixer.replaceNonstandardResidues()
        print("Removing heterogens...")
        fixer.removeHeterogens(keepWater=True)

        print("Finding missing atoms...")
        fixer.findMissingAtoms()
        print("Adding missing atoms...")
        fixer.addMissingAtoms()
        print("Adding missing hydrogens...")
        fixer.addMissingHydrogens(7)
        print("Writing PDB file...")

        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            open(os.path.join(".", "%s_fixed_pH_%s.pdb" % (pdb_id.split('.')[0], 7)),
                 "w"),
            keepIds=True)
        return "%s_fixed_pH_%s.pdb" % (pdb_id.split('.')[0], 7)

pdb_id = r'path\2zwh.pdb'
fixed_pdb_file = fix_pdb(pdb_id)
pdb = PDBFile(fixed_pdb_file)
modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(modeller.topology)
