# util.py
from rdkit import Chem
from rdkit.Chem import Draw
import rdkit.Chem.rdCoordGen as rdCoordGen
import ipywidgets as widgets
from IPython.display import display
from rdkit.Chem import AllChem

### Functions to create different molecules ###

def create_water():
    """Creates an H2O molecule with the oxygen highlighted."""
    water = Chem.RWMol()
    oxygen = water.AddAtom(Chem.Atom(8))  # Oxygen
    hydrogen1 = water.AddAtom(Chem.Atom(1))  # Hydrogen 1
    hydrogen2 = water.AddAtom(Chem.Atom(1))  # Hydrogen 2

    water.AddBond(oxygen, hydrogen1, Chem.BondType.SINGLE)
    water.AddBond(oxygen, hydrogen2, Chem.BondType.SINGLE)

    water = water.GetMol()
    Chem.SanitizeMol(water)
    rdCoordGen.AddCoords(water)

    return water, [oxygen]  # Highlight oxygen

def create_methane():
    """Creates CH4 (methane) with the carbon highlighted."""
    methane = Chem.RWMol()
    carbon = methane.AddAtom(Chem.Atom(6))  # Carbon
    h_atoms = [methane.AddAtom(Chem.Atom(1)) for _ in range(4)]  # 4 Hydrogens

    for h in h_atoms:
        methane.AddBond(carbon, h, Chem.BondType.SINGLE)

    methane = methane.GetMol()
    Chem.SanitizeMol(methane)
    rdCoordGen.AddCoords(methane)

    return methane, [carbon]  # Highlight carbon

def create_ammonia():
    """Creates NH3 (ammonia) with nitrogen highlighted."""
    ammonia = Chem.RWMol()
    nitrogen = ammonia.AddAtom(Chem.Atom(7))  # Nitrogen
    h_atoms = [ammonia.AddAtom(Chem.Atom(1)) for _ in range(3)]  # 3 Hydrogens

    for h in h_atoms:
        ammonia.AddBond(nitrogen, h, Chem.BondType.SINGLE)

    ammonia = ammonia.GetMol()
    Chem.SanitizeMol(ammonia)
    rdCoordGen.AddCoords(ammonia)

    return ammonia, [nitrogen]  # Highlight nitrogen

def create_carbon_dioxide():
    """Creates CO2 (carbon dioxide) with double bonds highlighted."""
    co2 = Chem.RWMol()
    oxygen1 = co2.AddAtom(Chem.Atom(8))
    carbon = co2.AddAtom(Chem.Atom(6))
    oxygen2 = co2.AddAtom(Chem.Atom(8))

    co2.AddBond(oxygen1, carbon, Chem.BondType.DOUBLE)
    co2.AddBond(carbon, oxygen2, Chem.BondType.DOUBLE)

    co2 = co2.GetMol()
    Chem.SanitizeMol(co2)
    rdCoordGen.AddCoords(co2)

    return co2, [(oxygen1, carbon), (carbon, oxygen2)]  # Highlight double bonds

def create_hydrogen_peroxide():
    """Creates H2O2 (hydrogen peroxide) with the oxygen-oxygen bond highlighted."""
    h2o2 = Chem.RWMol()
    h1 = h2o2.AddAtom(Chem.Atom(1))
    o1 = h2o2.AddAtom(Chem.Atom(8))
    o2 = h2o2.AddAtom(Chem.Atom(8))
    h2 = h2o2.AddAtom(Chem.Atom(1))

    h2o2.AddBond(h1, o1, Chem.BondType.SINGLE)
    h2o2.AddBond(o1, o2, Chem.BondType.SINGLE)
    h2o2.AddBond(o2, h2, Chem.BondType.SINGLE)

    h2o2 = h2o2.GetMol()
    Chem.SanitizeMol(h2o2)
    rdCoordGen.AddCoords(h2o2)

    return h2o2, [(o1, o2)]  # Highlight oxygen-oxygen bond

def visualize_molecule(molecule, highlight_atoms=None, highlight_bonds=None):
    """
    Generates and displays a molecule with optional atom/bond highlighting.
    
    Args:
        molecule (RDKit Mol): The molecule object.
        highlight_atoms (list): List of atom indices to highlight.
        highlight_bonds (list): List of bond tuples to highlight.
    """
    if highlight_bonds:
        highlight_bonds = [molecule.GetBondBetweenAtoms(*pair).GetIdx() for pair in highlight_bonds]

    img = Draw.MolToImage(
        molecule, 
        highlightAtoms=highlight_atoms if highlight_atoms else [], 
        highlightBonds=highlight_bonds if highlight_bonds else []
    )
    
    return img

# Dictionary of molecules
molecule_dict = {
    "Water (H2O)": create_water,
    "Methane (CH4)": create_methane,
    "Ammonia (NH3)": create_ammonia,
    "Carbon Dioxide (CO2)": create_carbon_dioxide,
    "Hydrogen Peroxide (H2O2)": create_hydrogen_peroxide
}

# Function to create and display an interactive widget
def display_molecule_selector():
    dropdown = widgets.Dropdown(
        options=list(molecule_dict.keys()),
        description="Molecule:"
    )

    output = widgets.Output()

    def on_change(change):
        with output:
            output.clear_output()
            mol_func = molecule_dict.get(dropdown.value)
            if mol_func:
                mol, highlights = mol_func()
                if isinstance(highlights[0], tuple):  # If bonds are being highlighted
                    img = visualize_molecule(mol, highlight_bonds=highlights)
                else:
                    img = visualize_molecule(mol, highlight_atoms=highlights)
                display(img)

    dropdown.observe(on_change, names="value")
    
    display(dropdown, output)
    on_change(None)  # Trigger initial visualization

def molecule_to_3d(molecule):
    mol = Chem.Mol(molecule)
    mol = AllChem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol