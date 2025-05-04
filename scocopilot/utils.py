#############################################################################################################################

##       #### ########  ########     ###    ########  #### ########  ######
##        ##  ##     ## ##     ##   ## ##   ##     ##  ##  ##       ##    ##
##        ##  ##     ## ##     ##  ##   ##  ##     ##  ##  ##       ##
##        ##  ########  ########  ##     ## ########   ##  ######    ######
##        ##  ##     ## ##   ##   ######### ##   ##    ##  ##             ##
##        ##  ##     ## ##    ##  ##     ## ##    ##   ##  ##       ##    ##
######## #### ########  ##     ## ##     ## ##     ## #### ########  ######

#############################################################################################################################

import ase
import math
import torch
import ase.io

import numpy           as np
import torch_geometric as tg

from pathlib          import Path as path

from ase.neighborlist import NeighborList
from ase.neighborlist import neighbor_list as neighbor
#
### Local libraries
#
from .io import fileio, stdout

#############################################################################################################################

########  ####  ######  ######## ####  #######  ##    ##    ###    ########  #### ########  ######  
##     ##  ##  ##    ##    ##     ##  ##     ## ###   ##   ## ##   ##     ##  ##  ##       ##    ## 
##     ##  ##  ##          ##     ##  ##     ## ####  ##  ##   ##  ##     ##  ##  ##       ##       
##     ##  ##  ##          ##     ##  ##     ## ## ## ## ##     ## ########   ##  ######    ######  
##     ##  ##  ##          ##     ##  ##     ## ##  #### ######### ##   ##    ##  ##             ## 
##     ##  ##  ##    ##    ##     ##  ##     ## ##   ### ##     ## ##    ##   ##  ##       ##    ## 
########  ####  ######     ##    ####  #######  ##    ## ##     ## ##     ## #### ########  ######

#############################################################################################################################
### Convert atomic number to element symbol

def z_to_symbol(elements):
        
    zval = {
        1 :"H" ,                                                                                                                                                                   2:"He",
        3 :"Li",   4:"Be",                                                                                                       5:"B" ,   6:"C" ,   7:"N" ,   8:"O" ,   9:"F" ,  10:"Ne",
        11:"Na",  12:"Mg",                                                                                                      13:"Al",  14:"Si",  15:"P" ,  16:"S" ,  17:"Cl",  18:"Ar",
        19:"K" ,  20:"Ca",  21:"Sc",  22:"Ti",  23:"V" ,  24:"Cr",  25:"Mn",  26:"Fe",  27:"Co",  28:"Ni",  29:"Cu",  30:"Zn",  31:"Ga",  32:"Ge",  33:"As",  34:"Se",  35:"Br",  36:"Kr",
        37:"Rb",  38:"Sr",  39:"Y" ,  40:"Zr",  41:"Nb",  42:"Mo",  43:"Tc",  44:"Ru",  45:"Rh",  46:"Pd",  47:"Ag",  48:"Cd",  49:"In",  50:"Sn",  51:"Sb",  52:"Te",  53:"I" ,  54:"Xe",
        55:"Cs",  56:"Ba",  57:"La",  72:"Hf",  73:"Ta",  74:"W" ,  75:"Re",  76:"Os",  77:"Ir",  78:"Pt",  79:"Au",  80:"Hg",  81:"Tl",  82:"Pb",  83:"Bi",  84:"Po",  85:"At",  86:"Rn",
        87:"Fr",  88:"Ra",  89:"Ac", 104:"Rf", 105:"Db", 106:"Sg", 107:"Bh", 108:"Hs", 109:"Mt", 110:"Ds", 111:"Rg", 112:"Cn", 113:"Nh", 114:"Fl", 115:"Mc", 116:"Lv", 117:"Ts", 118:"Og",
        
        58:"Ce",  59:"Pr",  60:"Nd",  61:"Pm",  62:"Sm",  63:"Eu",  64:"Gd",  65:"Tb",  66:"Dy",  67:"Ho",  68:"Er",  69:"Tm",  70:"Yb",  71:"Lu",
        90:"Th",  91:"Pa",  92:"U" ,  93:"Np",  94:"Pu",  95:"Am",  96:"Cm",  97:"Bk",  98:"Cf",  99:"Es", 100:"Fm", 101:"Md", 102:"No", 103:"Lr" 
        }
            
    vector   = [zval.get(i, f"{i}") for i in elements]

    to_numpy = np.array(vector)

    return to_numpy

#############################################################################################################################
### Calculate Pauling electronegativity difference between two indexes

def pauling(pairs=None, single=None, normalize=True):
    
    chi = {
        1 :2.20,                                                                                                                                                                   2:0.00,
        3 :0.98,   4:1.57,                                                                                                       5:2.04,   6:2.55,   7:3.04,   8:3.44,   9:3.98,  10:0.00,
        11:0.93,  12:1.31,                                                                                                      13:1.61,  14:1.90,  15:2.19,  16:2.58,  17:3.16,  18:0.00,
        19:0.82,  20:1.00,  21:1.36,  22:1.54,  23:1.63,  24:1.66,  25:1.55,  26:1.83,  27:1.88,  28:1.91,  29:1.90,  30:1.65,  31:1.81,  32:2.01,  33:2.18,  34:2.55,  35:2.96,  36:3.00,
        37:0.82,  38:0.95,  39:1.22,  40:1.33,  41:1.60,  42:2.16,  43:1.90,  44:2.20,  45:2.28,  46:2.20,  47:1.93,  48:1.69,  49:1.78,  50:1.96,  51:2.05,  52:2.10,  53:2.66,  54:2.60,
        55:0.79,  56:0.89,  57:1.10,  72:1.30,  73:1.50,  74:2.36,  75:1.90,  76:2.20,  77:2.20,  78:2.28,  79:2.54,  80:2.00,  81:1.62,  82:2.33,  83:2.02,  84:2.00,  85:2.20,  86:0.00,
        87:0.70,  88:0.89,  89:1.10, 104:0.00, 105:0.00, 106:0.00, 107:0.00, 108:0.00, 109:0.00, 110:0.00, 111:0.00, 112:0.00, 113:0.00, 114:0.00, 115:0.00, 116:0.00, 117:0.00, 118:0.00,
        
        58:1.12,  59:1.13,  60:1.14,  61:1.13,  62:1.17,  63:1.20,  64:1.20,  65:1.10,  66:1.22,  67:1.23,  68:1.24,  69:1.25,  70:1.10,  71:1.27,
        90:1.30,  91:1.50,  92:1.38,  93:1.36,  94:1.28,  95:1.30,  96:1.30,  97:1.30,  98:1.30,  99:1.30, 100:1.30, 101:1.30, 102:1.30, 103:1.30
        }
        
    scale = 1.0/3.98 if normalize else 1.0

    if pairs  is not None:diff = ( chi.get(pairs[0], 0.0) - chi.get(pairs[1], 0.0) )*scale
    if single is not None: diff = chi.get(i, 0.0)*scale

    return diff

#############################################################################################################################
##     ##  #######  ########  ##     ## ##       ########  ######
###   ### ##     ## ##     ## ##     ## ##       ##       ##    ##
#### #### ##     ## ##     ## ##     ## ##       ##       ##
## ### ## ##     ## ##     ## ##     ## ##       ######    ######
##     ## ##     ## ##     ## ##     ## ##       ##             ##
##     ## ##     ## ##     ## ##     ## ##       ##       ##    ##
##     ##  #######  ########   #######  ######## ########  ######

#############################################################################################################################
### Gather data from files

def read_xyz(species="", path=None, ext="xyz"):

    xyzfile = path/f"{species}.{ext}"

    if fileio(filename=xyzfile, option="read"):
        molecule = ase.io.read(xyzfile)

    return molecule

#############################################################################################################################
### Define bond order

def bonding_order(atoms=None, length=None, cutoff=None, metals=range(21,31,1)):
    
    infty = np.inf

    pairs = {
        6:{6:  [1.54, 1.33, 1.20, 1.41], # C-C
           7:  [1.47, 1.27, 1.15,infty], # C-N
           8:  [1.43, 1.23, 1.13,infty], # C-O
           15: [1.85, 1.67, 1.57,infty], # C-P
           16: [1.80, 1.58, 1.50,infty]  # C-S
          },
        7:{6:  [1.47, 1.27, 1.15,infty], # N-C
           7:  [1.47, 1.24, 1.10,infty], # N-N
           8:  [1.37, 1.19,infty,infty], # N-O
           16: [1.60,infty, 1.45,infty]  # N-S
          },
        8:{6:  [1.43, 1.23, 1.13,infty], # O-C
           7:  [1.37, 1.19,infty,infty], # O-N
           8:  [1.68, 1.20,infty,infty], # O-O
           15: [1.54, 1.47,infty,infty], # O-P
           16: [1.62, 1.44,infty,infty]  # O-S
          },
        15:{6: [1.85, 1.67, 1.57,infty], # P-C
            8: [1.54, 1.47,infty,infty], # P-O
            15:[2.14,infty, 1.90,infty], # P-P
            16:[infty,1.90,infty,infty]  # P-S
          },
        16:{6: [1.80, 1.58, 1.50,infty], # S-C
            7: [1.60,infty, 1.45,infty], # S-N
            8: [1.62, 1.44,infty,infty], # S-O
            15:[infty,1.90,infty,infty], # S-P
            16:[2.00, 1.87,infty,infty]  # S-S
          }
        }

    max_feat = 4
    
    data     = pairs.get(atoms[0],{0:0}).get(atoms[1],0)

    if isinstance(data, list):
        diff = [abs(i-length) for i in data]
        idx  = np.argmin(np.asarray(diff))
        
        idx += 1
        
        return idx

    idx = data
    
    if any(x in atoms for x in metals) and (length <= cutoff): idx = max_feat
    
    idx += 1
        
    return idx

#############################################################################################################################
### Local oxidation state [-1, 0 +1]

def oxidation_state(molecule=None, adjacency=None):

    rows, cols = np.where(adjacency==1)

    chi_diff   = torch.zeros(adjacency.shape, dtype=torch.float)

    atoms      = molecule.get_atomic_numbers()

    for row, col in zip(rows, cols):
        chi = pauling( pairs=[ atoms[row], atoms[col] ] )
        
        chi_diff[row,col] = 0.0 if chi==0.0 else math.copysign(1.0,  chi)
        chi_diff[col,row] = 0.0 if chi==0.0 else math.copysign(1.0, -chi)

    return chi_diff

#############################################################################################################################
### Get adjacency matrix

def adjacency_matrix(molecule=None, cutoff=None):

    nl        = NeighborList(cutoffs=cutoff, self_interaction=False, bothways=True)

    nl.update(molecule)

    adjacency = nl.get_connectivity_matrix(sparse=False)

    to_tensor = torch.tensor(adjacency, dtype=torch.float)

    return to_tensor

#############################################################################################################################
### Masking for coordination shell

def coordination_shell(molecule=None, adjacency=None, order=1):
    
    adjacency = adjacency.numpy()
    
    symbols   = np.array(molecule.get_chemical_symbols())
    
    if order is None: return np.ones(symbols.shape, dtype=bool)
    
    shells  = np.zeros(adjacency.shape, dtype=bool)
    
    for shell in range(order+1):
    
        mask = (symbols == "Cr") | (symbols == "Mn") | (symbols == "Fe") | (symbols == "Co") if shell == 0 else np.sum(adjacency[mask], axis=0).astype(bool)
        
        shells[mask] = adjacency[mask].astype(bool)
        
    mask = np.sum(shells, axis=1).astype(bool)
    
    return mask

#############################################################################################################################

########  ##     ## #### ##       ########      ######  ######## ########
##     ## ##     ##  ##  ##       ##     ##    ##    ## ##          ##
##     ## ##     ##  ##  ##       ##     ##    ##       ##          ##
########  ##     ##  ##  ##       ##     ##     ######  ######      ##
##     ## ##     ##  ##  ##       ##     ##          ## ##          ##
##     ## ##     ##  ##  ##       ##     ##    ##    ## ##          ##
########   #######  #### ######## ########      ######  ########    ##

#############################################################################################################################
### Build node features as the atomic number Z

def node_features(molecule=None):

    idx = molecule.get_atomic_numbers()
    
    to_tensor = torch.tensor(idx.tolist(), dtype=torch.float).unsqueeze(1)

    return to_tensor

#############################################################################################################################
### Build edge attributes

def edge_attributes(molecule=None, index=0, cutoff=None):

    rij = [molecule.get_distances(i[0], i[1], mic=True, vector=False)[0] for i in index.T]
    
    z   = molecule.get_atomic_numbers()
    
    idx = [ bonding_order(atoms=[ z[i[0]],z[i[1]] ], cutoff=cutoff[i[0]]+cutoff[i[1]],
                          length=j) for i, j in zip(index.T, rij) ]
    
    to_tensor = torch.tensor(idx, dtype=torch.float).unsqueeze(1)
    
    return to_tensor

#############################################################################################################################
### Build node attributes using local oxidation state (-1, 0, 1)

def node_attributes(molecule=None, adjacency=None, max_feat=16.0):
    
    chi_vec   = torch.arange(-max_feat, max_feat+1.0, step=1.0)
        
    chi_diff  = oxidation_state(molecule=molecule, adjacency=adjacency)
    
    charges   = torch.sum(chi_diff, dim=1).unsqueeze(1)
    
    charges   = torch.clamp(charges, min=-max_feat, max=max_feat)
        
    return charges

#############################################################################################################################
### Get keywords

def get_keys(node_feat=None, coord=None, edge_index=None, edge_attr=None, node_attr=None, adjacency=None, sco=None):

    keywords = {"node_feat":node_feat, "coord":coord, "edge_index":edge_index, "edge_attr":edge_attr, "node_attr":node_attr, "Esco":sco}
    
    data = tg.data.Data(**keywords)
    
    return data

#############################################################################################################################
### Build data set and save if requested

def graphdata(species=None, structures=None, ext="pth", sco=0.0, shell=None):
                
    molecule   = read_xyz(species=species, path=structures)
    
    adjacency  = adjacency_matrix(molecule=molecule, cutoff=ase.neighborlist.natural_cutoffs(molecule))
    
    mask       = coordination_shell(molecule=molecule, adjacency=adjacency, order=shell)
    
    ncutoff    = ase.neighborlist.natural_cutoffs(molecule[mask])
    ncutoff    = [1.25*i for i in ncutoff]

    ijshift    = neighbor("ijS", a=molecule[mask], cutoff=ncutoff, self_interaction=False)
    
    coord      = torch.tensor(molecule.get_positions()[mask], dtype=torch.float)
    
    edge_index = torch.stack([torch.LongTensor(ijshift[0]), torch.LongTensor(ijshift[1])], dim=0)
    
    node_feat  = node_features(molecule=molecule)[mask]
    
    edge_attr  = edge_attributes(molecule=molecule[mask], index=edge_index, cutoff=ncutoff)
    
    node_attr  = node_attributes(molecule=molecule, adjacency=adjacency)[mask]

    data       = get_keys(node_feat=node_feat, coord=coord, edge_index=edge_index, edge_attr=edge_attr, node_attr=node_attr, sco=sco)
    
    return data

#############################################################################################################################
