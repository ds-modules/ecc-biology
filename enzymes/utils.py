import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact
import pandas as pd
import plotly.express as px
from ipywidgets import interact, Dropdown, IntSlider
from IPython.display import display, clear_output


def display_data_counts(df):
    var_hap_counts = df['Variant/Haplotypes'].value_counts().head(12)
    var_hap_counts.index = var_hap_counts.index.str.slice(0, 15) + '...'
    alleles_counts = df['Alleles'].value_counts().head(12)
    drugs_counts = df['Drug(s)'].value_counts().head(12)

    category_counts_list = [var_hap_counts, alleles_counts, drugs_counts]
    titles = ['Variant/Haplotypes', 'Alleles', 'Drug(s)']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes = axes.flatten()

    def enum(category_counts_list, titles):
        for i, ax in enumerate(axes):
            category_counts_list[i].plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
            ax.set_title(titles[i])
            ax.set_ylabel('Count')
            ax.set_xlabel(titles[i])

    enum(category_counts_list, titles)
    plt.tight_layout()
    plt.show()

def display_heatmap(phenotype,x, top_k):
    phenotype = phenotype.loc[phenotype['Gene'] == 'CYP2D6']
    indep = x
    dep = 'Phenotype Category'
    cross_tab = pd.crosstab(phenotype[indep], phenotype[dep])
    heatmap_data = cross_tab.reset_index().melt(id_vars=indep, var_name=dep, value_name='Count')

    top_indep = cross_tab.sum(axis=1).nlargest(top_k).index
    top_dep = cross_tab.sum(axis=0).nlargest(top_k).index
    filtered_cross_tab = cross_tab.loc[top_indep, top_dep]
    heatmap_data = filtered_cross_tab.reset_index().melt(id_vars=indep, var_name=dep, value_name='Count')
    fig = px.density_heatmap(heatmap_data, x=indep, y=dep, z='Count', text_auto=True,
                            title='Top 7 ' + x + ' vs. Phenotype Category Heatmap')

    display(fig)

def display_population_type_data(phenotype, column):
    healthy = phenotype[phenotype['Population types'] == 'in healthy individuals']
    healthy_alleles = healthy[column].value_counts().head(12)
    people = phenotype[phenotype['Population types'] == 'in people with']
    people_alleles = people[column].value_counts().head(12)
    children = phenotype[phenotype['Population types'] == 'in children']
    children_alleles = children[column].value_counts().head(12)
    women = phenotype[phenotype['Population types'] == 'in women']
    women_alleles = women[column].value_counts().head(12)

    category_counts_list = [healthy_alleles, people_alleles, children_alleles, women_alleles]
    titles = ['Healthy People', 'People with', 'Children with ','Women ']
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    axes = axes.flatten()

    def enum(category_counts_list, titles):
        for i, ax in enumerate(axes):
            category_counts_list[i].plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
            ax.set_title(titles[i])
            ax.set_ylabel('Count')
            ax.set_xlabel(titles[i])

    enum(category_counts_list, titles)

    plt.tight_layout()
    plt.show()


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