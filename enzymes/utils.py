import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, Dropdown, IntSlider
from IPython.display import display, clear_output

def load_data():
    df = pd.read_csv('variantAnnotations/var_drug_ann.tsv', sep='\t')
    df = df.loc[df['Gene'] == 'CYP2D6']
    mini_view = df[['PMID', 'Gene', 'Variant/Haplotypes' ,'Drug(s)', 'Phenotype Category', 'Significance', 'Alleles', 'Metabolizer types', 'Population types' ,'Is/Is Not associated', 'Notes']]
    display(mini_view.tail())
    return df

def display_data_counts():
    df = pd.read_csv('variantAnnotations/var_drug_ann.tsv', sep='\t')
    df = df.loc[df['Gene'] == 'CYP2D6']
    var_hap_counts = df['Population types'].value_counts().head(12)
    alleles_counts = df['Metabolizer types'].value_counts().head(12)
    drugs_counts = df['Drug(s)'].value_counts().head(12)

    category_counts_list = [drugs_counts, var_hap_counts, alleles_counts]
    titles = ['Drug(s)', 'Population types', 'Metabolizer types']
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes = axes.flatten()

    def enum(category_counts_list, titles):
        for i, ax in enumerate(axes):
            category_counts_list[i].plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
            ax.set_title(titles[i])
            ax.set_ylabel('Count')
            ax.set_xlabel(titles[i])

    enum(category_counts_list, titles)
    plt.show()

def display_heatmap():
    phenotype = pd.read_csv('variantAnnotations/var_pheno_ann.tsv', sep='\t')
    phenotype = phenotype.loc[phenotype['Gene'] == 'CYP2D6']
    output = widgets.Output()
    
    def show_heatmap(dep, top_k):
        with output:
            clear_output(wait=True)
            
            indep = 'Phenotype Category'
            cross_tab = pd.crosstab(phenotype[indep], phenotype[dep])
            
            if cross_tab.empty:
                print("No data to display for this selection.")
                return
            
            top_indep = cross_tab.sum(axis=1).nlargest(top_k).index
            top_dep = cross_tab.sum(axis=0).nlargest(top_k).index
            filtered_cross_tab = cross_tab.loc[top_indep, top_dep]
            
            heatmap_data = filtered_cross_tab.reset_index().melt(
                id_vars=indep, var_name=dep, value_name='Count'
            )
            
            fig = px.density_heatmap(
                heatmap_data, x=indep, y=dep, z='Count', text_auto=True,
                title=f'Top {top_k} {indep} vs. {dep} for CYP2D6'
            )
            
            fig.show()
    
    indep_widget = widgets.Dropdown(
        options=['Metabolizer types', 'Population types'],
        value='Population types',
        description='Feature'
    )
    
    topk_widget = widgets.IntSlider(
        min=1, max=6, value=3, step=1, description='Top K'
    )
    
    ui = widgets.VBox([indep_widget, topk_widget])
    out = widgets.interactive_output(show_heatmap, {
        'dep': indep_widget,
        'top_k': topk_widget
    })
    display(ui, output) 

def display_population_types():
    phenotype = pd.read_csv('variantAnnotations/var_pheno_ann.tsv', sep='\t')
    phenotype = phenotype.loc[phenotype['Gene'] == 'CYP2D6']
    output = widgets.Output()#layout=widgets.Layout(height='600px'))
    def plot_population_data(column):
        with output:
            clear_output(wait=True)

            #if column not in phenotype.columns:
            #    print(f"Column '{column}' not found in data.")
            #    return
            col = phenotype[phenotype['Population types'] == column]
            counts = col["Phenotype Category"].value_counts().head(12)
            fig = counts.plot(kind='bar', color='skyblue', edgecolor='black', title=f'Phenotype Category {column}')
            plt.figure(figsize=(6, 4))
            plt.show()
    
    # Create dropdown widget for column selection
    column_widget = widgets.Dropdown(
        options=['in people with', 'in women', 'in women with',
       'in healthy individuals', 'in children with', 'in infants', 'in',
       'in children', 'in men with', 'in men'],
        description='Population Type:'
    )

    ui = widgets.VBox([column_widget])
    out = widgets.interactive_output(plot_population_data, {'column': column_widget})
    
    display(ui, output)

def run_SVM(X, y):
    categorical_cols = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', SVC(kernel='rbf', probability=True))], verbose=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))