import ipywidgets as widgets
from IPython.display import display, Markdown

# Define column explanations
column_descriptions = {
    "patient_id": "A unique ID assigned to each patient. Used to track individuals without revealing personal information.",
    "age_at_diagnosis": "The age (in years) when the patient was first diagnosed with breast cancer. Useful for understanding how cancer risk varies with age.",
    "type_of_breast_surgery": "Indicates whether the patient had a mastectomy (removal of the whole breast) or breast-conserving surgery (removal of only the tumor area).",
    "cancer_type_detailed": "A specific classification of the breast cancer type, e.g., 'Invasive Ductal Carcinoma' or 'Mixed Ductal and Lobular Carcinoma.'",
    "pam50_+_claudin-low_subtype": "A molecular classification of breast cancer based on gene expression profiling. Subtypes include Luminal A, Luminal B, HER2-enriched, Basal-like, and Claudin-low.",
    "chemotherapy": "Indicates whether the patient received chemotherapy (0 = No, 1 = Yes).",
    "hormone_therapy": "Indicates whether the patient received hormone-blocking therapy, typically for hormone-receptor-positive cancers.",
    "radio_therapy": "Indicates whether the patient received radiation therapy after surgery.",
    "tumor_size": "Size of the primary tumor measured in millimeters (mm). Larger tumors often indicate more advanced disease.",
    "tumor_stage": "Clinical stage of the cancer, describing how much the cancer has spread (e.g., Stage I, II, III).",
    "overall_survival": "Whether the patient survived the follow-up period (0 = Alive, 1 = Deceased).",
    "overall_survival_months": "The number of months the patient survived after diagnosis. Useful for survival analysis."
}

# Create a dropdown widget
dropdown = widgets.Dropdown(
    options=column_descriptions.keys(),
    description='Column:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='60%')
)

# Create an output area
output = widgets.Output()

# Define what happens when user selects an option
def show_description(change):
    with output:
        output.clear_output()
        display(Markdown(f"**{change['new']}**: {column_descriptions[change['new']]}"))

# Define column explanations
column_descriptions = {
    "patient_id": "A unique ID assigned to each patient. Used to track individuals without revealing personal information.",
    "age_at_diagnosis": "The age (in years) when the patient was first diagnosed with breast cancer. Useful for understanding how cancer risk varies with age.",
    "type_of_breast_surgery": "Indicates whether the patient had a mastectomy (removal of the whole breast) or breast-conserving surgery (removal of only the tumor area).",
    "cancer_type_detailed": "A specific classification of the breast cancer type, e.g., 'Invasive Ductal Carcinoma' or 'Mixed Ductal and Lobular Carcinoma.'",
    "pam50_+_claudin-low_subtype": "A molecular classification of breast cancer based on gene expression profiling. Subtypes include Luminal A, Luminal B, HER2-enriched, Basal-like, and Claudin-low.",
    "chemotherapy": "Indicates whether the patient received chemotherapy (0 = No, 1 = Yes).",
    "hormone_therapy": "Indicates whether the patient received hormone-blocking therapy, typically for hormone-receptor-positive cancers.",
    "radio_therapy": "Indicates whether the patient received radiation therapy after surgery.",
    "tumor_size": "Size of the primary tumor measured in millimeters (mm). Larger tumors often indicate more advanced disease.",
    "tumor_stage": "Clinical stage of the cancer, describing how much the cancer has spread (e.g., Stage I, II, III).",
    "overall_survival": "Whether the patient survived the follow-up period (0 = Alive, 1 = Deceased).",
    "overall_survival_months": "The number of months the patient survived after diagnosis. Useful for survival analysis."
}

# Create a dropdown widget
dropdown = widgets.Dropdown(
    options=column_descriptions.keys(),
    description='Column:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='60%')
)

# Create an output area
output = widgets.Output()

# Define what happens when user selects an option
def show_description(change):
    with output:
        output.clear_output()
        display(Markdown(f"**{change['new']}**: {column_descriptions[change['new']]}"))

