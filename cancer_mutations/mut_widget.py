import ipywidgets as widgets
from IPython.display import display, Markdown
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

Le_eda_df = pd.read_csv("eda_METABRIC.csv")

# Important mutation gene descriptions
mutation_descriptions = {
    "TP53": "Tumor suppressor gene. Protects the genome by triggering cell cycle arrest or apoptosis when DNA is damaged. Mutations in TP53 are common in many cancers.",
    "PIK3CA": "Encodes a subunit of PI3K, involved in cell growth and survival signaling. Mutations often lead to increased tumor cell proliferation.",
    "BRCA1": "Tumor suppressor involved in DNA double-strand break repair. Mutations increase breast and ovarian cancer risk significantly.",
    "BRCA2": "Like BRCA1, involved in DNA repair. BRCA2 mutations are strongly linked to hereditary breast and ovarian cancers.",
    "PTEN": "Tumor suppressor gene that regulates the PI3K/AKT pathway. Loss of PTEN promotes uncontrolled cell survival and tumor development.",
    "CDH1": "Encodes E-cadherin, critical for cell adhesion. Mutations can lead to loss of adhesion and increased cancer spread (metastasis).",
    "AKT1": "Oncogene involved in promoting cell growth and survival. Mutations in AKT1 are implicated in several types of cancers.",
    "RB1": "Retinoblastoma tumor suppressor protein. Regulates the cell cycle; loss of function leads to unchecked cell division.",
    "GATA3": "Transcription factor important for cell differentiation in breast tissue. Mutations are common in breast cancer.",
    "MAP2K4": "Involved in transmitting stress signals that regulate apoptosis and cell proliferation. Altered in some cancers."
}

mutation_options = [
    'tp53_mut', 'pik3ca_mut', 'brca1_mut',
    'pten_mut', 'cdh1_mut', 'akt1_mut', 'gata3_mut', 'map2k4_mut'
]

# Define your plotting function
def plot_mutation_by_subtype(mutation: str):
    grouped = Le_eda_df.groupby('pam50_+_claudin-low_subtype')[mutation].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=grouped.index, y=grouped.values)
    plt.title(f'{mutation.upper()} Mutation Frequency Across PAM50 Subtypes')
    plt.ylabel('Proportion Mutated')
    plt.xlabel('PAM50 Cancer Subtype')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.show()

# Define a function to create the widget and wire it up
def create_mutation_widget():
    mutation_dropdown = widgets.Dropdown(
        options=mutation_options,
        description='Select Mutation:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )

    output = widgets.Output()

    def on_mutation_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            with output:
                clear_output()
                plot_mutation_by_subtype(change['new'])

    mutation_dropdown.observe(on_mutation_change)

    display(mutation_dropdown, output)
