
import numpy as np
from McUtils import Numputils as nput

def fragment_submodes(mol, frag_pos):
    frag_mol = mol.take_submolecule(frag_pos)
    cart_pos = (np.arange(3)[np.newaxis, :] + frag_pos[:, np.newaxis ] *3).flatten()
    modes = mol.get_normal_modes()
    atom_disps = modes.matrix[cart_pos, :]
    f_base_sub = atom_disps @ np.diag(modes.freqs ** 2) @ atom_disps.T
    frag_mol.potential_derivatives = [
        0,
        f_base_sub
    ]
    base_modes = frag_mol.get_normal_modes(mass_weighted=True).matrix

    full_frag_modes = np.zeros((len(mol.atoms), 3, base_modes.shape[-1]))
    full_frag_modes[frag_pos, :, :] = base_modes.reshape(len(frag_pos), 3, -1)
    return full_frag_modes.reshape(-1, full_frag_modes.shape[-1])

def fragment_transrot_modes(mol, frag_pos):
    rem = np.delete(np.arange(len(mol.atoms)), frag_pos)
    tr_modes = mol.translation_rotation_modes[1].reshape(len(mol.atoms), 3, -1)
    frag = mol.take_submolecule(frag_pos)
    full_frag_modes = np.zeros(tr_modes.shape)
    submodes = frag.translation_rotation_modes[1].reshape(len(frag_pos), 3, -1)
    full_frag_modes[frag_pos, :, :] = submodes
    tr_modes = tr_modes.reshape(-1, tr_modes.shape[-1])
    full_frag_modes = full_frag_modes.reshape(-1, full_frag_modes.shape[-1])
    no_rot_modes = nput.project_out(full_frag_modes.T, tr_modes).T
    return np.concatenate([tr_modes, nput.vec_normalize(no_rot_modes, axis=0)], axis=1)

def fragment_localized_modes(mol, frag_pos):
    rel_modes = fragment_transrot_modes(mol, frag_pos)
    base_modes = fragment_submodes(mol, frag_pos)
    no_rot_modes = nput.project_out(base_modes.T, rel_modes).T
    loc_modes = nput.vec_normalize(no_rot_modes, axis=0)
    return np.concatenate([rel_modes, loc_modes], axis=1)