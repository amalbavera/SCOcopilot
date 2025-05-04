#############################################################################################################################

##       #### ########  ########     ###    ########  #### ########  ######
##        ##  ##     ## ##     ##   ## ##   ##     ##  ##  ##       ##    ##
##        ##  ##     ## ##     ##  ##   ##  ##     ##  ##  ##       ##
##        ##  ########  ########  ##     ## ########   ##  ######    ######
##        ##  ##     ## ##   ##   ######### ##   ##    ##  ##             ##
##        ##  ##     ## ##    ##  ##     ## ##    ##   ##  ##       ##    ##
######## #### ########  ##     ## ##     ## ##     ## #### ########  ######

#############################################################################################################################

import os
import torch
import matplotlib.pyplot as plt

from rdkit      import Chem
from rdkit.Chem import AllChem

from pathlib    import Path as path

import warnings
warnings.filterwarnings("ignore")

#
### Local libraries
#
from .nn    import EquivariantGraphNeuralNetwork
from .utils import graphdata

imhere = path.cwd()

#############################################################################################################################

##     ##  #######  ########  ##     ## ##       ########  ######
###   ### ##     ## ##     ## ##     ## ##       ##       ##    ##
#### #### ##     ## ##     ## ##     ## ##       ##       ##
## ### ## ##     ## ##     ## ##     ## ##       ######    ######
##     ## ##     ## ##     ## ##     ## ##       ##             ##
##     ## ##     ## ##     ## ##     ## ##       ##       ##    ##
##     ##  #######  ########   #######  ######## ########  ######

#############################################################################################################################
### Pass data through the neural network

def smiles_to_xyz(smiles="", filename="molecule.xyz", show=True, size=(2,2)):
    
    params = AllChem.ETKDGv3()
    params.useSmallRingTorsions = True
    params.randomSeed=0xf00d
    
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    
    if show:
        fig, ax = plt.subplots( figsize=(size) )
        
        opt   = Chem.Draw.MolDrawOptions()
    
        #opt.setBackgroundColour(color)
        opt.bondLineWidth = 7
        opt.fixedFontSize = 40
        opt.bondLineWidth = 5

        img = Chem.Draw.MolToImage(molecule,size=(512,512), fitImage=True, options=opt)

        ax.imshow(img)
        
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel(None)
        ax.set_ylabel(None)
        
        ax.spines[:].set_visible(False)
        
        plt.show()
        
    molecule = Chem.AddHs(molecule)

    AllChem.EmbedMolecule(molecule, params=params)
    
    AllChem.UFFOptimizeMolecule(molecule, maxIters=1_000)
        
    Chem.rdmolfiles.MolToXYZFile( molecule, str(filename) )
    
#############################################################################################################################
### Pass data through the neural network

def passdata(data=None, network=None, batch=None, size=1, transform=None):

    median = transform["mean"]
    stddev = transform["stddev"]

    with torch.no_grad():
        output = network(data.node_feat, data.coord, data.edge_index, data.edge_attr, data.node_attr, batch, size)
        
    return output*stddev + median

#############################################################################################################################
### Unit conversion

def convert_units(units="eV"):
    
    if "kcal" in units.lower():
        conversion = (23.060541945329334,"kcal/mol")

    elif "kj" in units.lower():
        conversion = (96.4853074992579,"kJ/mol")

    else:
        conversion = (1.0, "eV")
        
    return conversion

#############################################################################################################################
### Folding

def folding(models=None, data=None, transform=None):
    
    output = torch.zeros(len(models), dtype=torch.float)
    
    for idx, model in enumerate(models):
        x = passdata(data=data, network=model, batch=torch.zeros(len(data.coord), dtype=torch.int64), transform=transform)

        output[idx] = x.squeeze(1)
        
    q1 = torch.quantile(output, 0.25, axis=0)
    q2 = torch.quantile(output, 0.75, axis=0)
    
    quantiles = torch.where( ((output>=q1) & (output<=q2)), output, torch.nan )
    
    mean   = torch.nanmean(quantiles, dim=0)
    stddev = torch.std(quantiles[~torch.isnan(quantiles)], dim=0)
        
    return mean, stddev
    
#############################################################################################################################
### Spin-crossover analysis

def sco_analysis(sco, minsco=-0.2, maxsco=0.5, thresh=0.2):
    
    thresh += 1.0
        
    if minsco <= sco <= maxsco:
        statement = "\033[92mPROMISING\033[00m"

    elif thresh*minsco <= sco <= thresh*maxsco:
        statement = "\033[33mPLAUSIBLE\033[00m"

    else:
        statement = "\033[91mDUBITABLE\033[00m"
        
    return statement

#############################################################################################################################

 ######  ##          ###     ######   ######  ########  ######
##    ## ##         ## ##   ##    ## ##    ## ##       ##    ##
##       ##        ##   ##  ##       ##       ##       ##
##       ##       ##     ##  ######   ######  ######    ######
##       ##       #########       ##       ## ##             ##
##    ## ##       ##     ## ##    ## ##    ## ##       ##    ##
 ######  ######## ##     ##  ######   ######  ########  ######
        
#############################################################################################################################
### Query equivariant graph neural network

class query:
    
    __slots__ = ["models","transform"]
#
### Initialize
#
    def __init__(self, nodes=1, edge_nf=1, node_nf=1, hidden_nf=8, layers=1,
                 folds_dir=imhere/"scocopilot"/"models",
                 foldnames=[ f"model{i:03d}" for i in range(100) ],
                 ext="pth", device="cpu", transform={"mean":0.5385, "stddev":1.6786}):
        
        model_folds    = []
        
        for fold in foldnames:
            model = EquivariantGraphNeuralNetwork(nodes=nodes, edge_nf=edge_nf, node_nf=node_nf, layers=layers, hidden_nf=hidden_nf,
                                                  activation=torch.nn.Tanhshrink(), aggregation="sum")
            
            model.load_state_dict( torch.load(folds_dir/f"{fold}.{ext}", map_location=device) )
            model.to(device)
            
            model_folds.append(model)
            
        self.transform = transform
        self.models    = model_folds
#
### Pass SMILES or XYZ
#
    def __call__(self, smiles, filename=imhere/"molecule.xyz", units="eV", show=True, size=(4,4), shell=None, minimum=-0.2, maximum=0.5, verbose=True):
        
        conversion = convert_units(units=units)
        
        if os.path.exists(smiles):
            filename = path(smiles)
            
        else:
            smiles_to_xyz(smiles, filename=filename, show=show, size=size)
        
        kwargs = {"species":filename.name.replace(".xyz",""), "sco":0.0, "shell":shell, "structures":filename.parent}
        
        data   = graphdata(**kwargs)
        
        mean, stddev = folding(models=self.models, data=data, transform=self.transform)
            
        statement    = sco_analysis(mean, minsco=minimum, maxsco=maximum)
        
        if verbose:
            print(f"ΔE_HL ≈ {mean*conversion[0]:.2f} ± {stddev*conversion[0]:.2f} {conversion[1]}. The species seems {statement}")
            return
        
        return mean.item()*conversion[0], stddev.item()*conversion[0]

#############################################################################################################################
