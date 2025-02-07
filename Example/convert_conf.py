#!/usr/bin/env python3

import numpy as np

try:
    f = open('opt_converged_000.xyz')
except:
    f = open('grown_string_000.xyz')
    
Coords = []
n_of_atoms = []
Energies = []
while True:
    line = f.readline()
    if line == '[Geometries] (XYZ)\n':
        tmp_coord = []
        n_of_atoms = f.readline()
        tmp_coord.append(n_of_atoms)
        empty_line = f.readline()
        tmp_coord.append(empty_line)
        for i in range(int(n_of_atoms)):
            coord_line = f.readline()
            tmp_coord.append(coord_line)
        Coords.append(tmp_coord)

    elif line == n_of_atoms:
        tmp_coord = []
        tmp_coord.append(n_of_atoms)
        empty_line = f.readline()
        tmp_coord.append(empty_line)
        for i in range(int(n_of_atoms)):
            coord_line = f.readline()
            tmp_coord.append(coord_line)
        Coords.append(tmp_coord)

    elif line == 'energy\n':
        for i in range(len(Coords)):
            energy = f.readline()
            Energies.append(float(energy))
    elif line == 'max-step\n':
        break

Energies = np.array(Energies)
TS_idx = np.argmax(Energies)
Reactant_idx = np.where(Energies == min(Energies[TS_idx:]))[0][0]
Product_idx = 0
Revised_energies = Energies - Energies[Reactant_idx]
TS_energy = Revised_energies[TS_idx]
Product_energy = Revised_energies[Product_idx]

f = open('conf_continue.xyz','w')
for i in range(len(Coords[Reactant_idx])):
    f.write(Coords[Reactant_idx][i])
for i in range(len(Coords[Product_idx])):
    f.write(Coords[Product_idx][i])
f.close()
