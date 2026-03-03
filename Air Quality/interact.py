import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import scipy.stats as stats
import folium
import ipywidgets as widgets
import branca
import otter

from IPython.display import display, clear_output
from branca.colormap import linear
from scipy.stats import pearsonr


def clean_enviro_data(file_path):
    """
    Cleans and merges environmental and demographic data from the given Excel file.
    
    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Merged and cleaned DataFrame for Los Angeles County.
    """
    enviro_df = pd.read_excel(file_path, engine="openpyxl")
    demo_df = pd.read_excel(file_path, sheet_name="Demographic Profile", header=1, engine="openpyxl")

    columns_to_keep = ["Census Tract", "Total Population", "California County", "Longitude", "Latitude",
                       "PM2.5", "Cleanup Sites", "Haz. Waste", "Pollution Burden", "Asthma", "Cardiovascular Disease", "Traffic", "Poverty", "Unemployment", "Education"]
    
    columns_to_keep2 = ["Census Tract", "Total Population", "California County",
                        "Hispanic (%)", "White (%)", "African American (%)", "Asian American (%)"]

    enviro_df_filtered = enviro_df[columns_to_keep]
    demo_df_filtered = demo_df[columns_to_keep2]

    # Save filtered datasets as CSV -- not necessary 
    
    # enviro_df_filtered.to_csv("enviro_df_filtered.csv", index=False)
    # demo_df_filtered.to_csv("demo_df_filtered.csv", index=False)

    enviro_data = enviro_df_filtered[enviro_df_filtered["California County"] == "Los Angeles"].dropna()
    demo_data = demo_df_filtered[demo_df_filtered["California County"] == "Los Angeles"].dropna()

    merged_df = enviro_data.merge(demo_data, on="Census Tract").rename(
        columns={"Total Population_x": "Population", "California County_x": "County"}
    )

    print("You just cleaned the data! Please do not run me multiple times.")
    return merged_df


def create_pollution_map(merged_df): #save_path="asthma_cardiovascular_pm25_map.html"):
    """
    Generates a Folium map showing Pollution Burden, Asthma, and Cardiovascular Disease.

    Parameters:
        merged_df (pd.DataFrame): The cleaned and merged DataFrame containing pollution & health data.
        save_path (str): File path to save the HTML map.
    
    Returns:
        None
    """

    columns_needed = ["Pollution Burden", "Latitude", "Longitude", "Asthma", "Cardiovascular Disease", "PM2.5"]
    df = merged_df.dropna(subset=columns_needed)

    m2 = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()], zoom_start=9, tiles="CartoDB Voyager")

    colormap = linear.Reds_09.scale(df["Pollution Burden"].min(), df["Pollution Burden"].max()).to_step(10)

    for _, row in df.iterrows():
        popup_info = f"""
        <b>Asthma Rate:</b> {row["Asthma"]:.2f}% <br>
        <b>Cardiovascular Disease Rate:</b> {row["Cardiovascular Disease"]:.2f}% <br>
        <b>Pollution Burden:</b> {row["Pollution Burden"]:.2f}% <br>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=max(row["Pollution Burden"] / 5, 3),
            color=colormap(row["Pollution Burden"]),
            fill=True,
            fill_color=colormap(row["Pollution Burden"]),
            fill_opacity=0.8,
            popup=folium.Popup(popup_info, max_width=300)
        ).add_to(m2)

    colormap.caption = "Pollution Burden (%)"
    colormap.add_to(m2)

    display(m2)
    #m2.save(save_path)
    #print(f"Map saved as: {save_path}")

    q_2()
    q_3()
    q_4()


def interactive_pollution_map(merged_df):
    """
    Creates an interactive Folium map with ipywidgets, allowing users to compare pollution factors with two racial groups.

    Parameters:
        merged_df (pd.DataFrame): The cleaned and merged DataFrame containing pollution & demographic data.

    Returns:
        None (Displays an interactive map inside a Jupyter Notebook)
    """

    numeric_columns = ["PM2.5", "Pollution Burden", "Cleanup Sites", "Haz. Waste", "Asthma", "Cardiovascular Disease"]
    race_columns = ["Hispanic (%)", "White (%)", "African American (%)", "Asian American (%)"]

    var1_dropdown = widgets.Dropdown(
        options=numeric_columns,
        value="Pollution Burden",
        description="Factor:",
        style={'description_width': 'initial'}
    )

    race1_dropdown = widgets.Dropdown(
        options=race_columns,
        value="Hispanic (%)",
        description="Group 1:",
        style={'description_width': 'initial'}
    )

    race2_dropdown = widgets.Dropdown(
        options=race_columns,
        value="White (%)",
        description="Group 2:",
        style={'description_width': 'initial'}
    )

    output = widgets.Output()

    def update_map(var1, race1, race2):
        with output:
            clear_output(wait=True)

            m = folium.Map(location=[merged_df["Latitude"].mean(), merged_df["Longitude"].mean()], zoom_start=9, tiles="CartoDB Voyager")

            colormap = linear.Purples_09.scale(merged_df[var1].min(), merged_df[var1].max())

            filtered_df = merged_df.dropna(subset=["Latitude", "Longitude", var1, race1, race2])

            for _, row in filtered_df.iterrows():
                race_difference = abs(row[race1] - row[race2])  
                radius = max(race_difference / 2, 3)

                popup_info = f"""
                <b>Census Tract:</b> {row["Census Tract"]} <br>
                <b>Population:</b> {row["Population"]} <br>
                <b>{var1}:</b> {row[var1]:.2f} <br>
                <b>{race1}:</b> {row[race1]:.2f}% <br>
                <b>{race2}:</b> {row[race2]:.2f}% <br>
                <b>Difference:</b> {race_difference:.2f}%
                """

                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=radius,
                    color=colormap(row[var1]),
                    fill=True,
                    fill_color=colormap(row[var1]),
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_info, max_width=300)
                ).add_to(m)

            colormap.caption = f"{var1} Levels"
            colormap.add_to(m)

            display(m)

    button = widgets.Button(description="Update Map", button_style="primary")

    def on_button_click(b):
        update_map(var1_dropdown.value, race1_dropdown.value, race2_dropdown.value)

    button.on_click(on_button_click)

    display(var1_dropdown, race1_dropdown, race2_dropdown, button, output)

def interactive_scatter_plot(merged_df):
    """
    Creates an interactive Seaborn scatter plot with ipywidgets, allowing users to compare pollution factors with two racial groups.

    Parameters:
        merged_df (pd.DataFrame): The cleaned and merged DataFrame containing pollution & demographic data.

    Returns:
        None (Displays an interactive scatter plot inside a Jupyter Notebook)
    """

    numeric_columns = ["Pollution Burden", "Cleanup Sites", "Haz. Waste", "Asthma", "Cardiovascular Disease"]
    race_columns = ["Hispanic (%)", "White (%)", "African American (%)", "Asian American (%)"]

    hazard_dropdown = widgets.Dropdown(
        options=numeric_columns,
        value="Pollution Burden",
        description="Hazard Factor:",
        style={'description_width': 'initial'}
    )

    race1_dropdown = widgets.Dropdown(
        options=race_columns,
        value="Hispanic (%)",
        description="Group 1:",
        style={'description_width': 'initial'}
    )

    race2_dropdown = widgets.Dropdown(
        options=race_columns,
        value="White (%)",
        description="Group 2:",
        style={'description_width': 'initial'}
    )

    output1 = widgets.Output()

    def update_plot(hazard, race1, race2):
        with output1:
            clear_output(wait=True)

            valid_df = merged_df.dropna(subset=[hazard, race1, race2])

            corr1, _ = pearsonr(valid_df[race1], valid_df[hazard])
            corr2, _ = pearsonr(valid_df[race2], valid_df[hazard])

            plt.figure(figsize=(12, 8))
            sns.set_style("whitegrid")

            sns.scatterplot(data=merged_df, x=race1, y=hazard, label=f"{race1}", s=80, color="purple", edgecolor="black", alpha=0.7)
            sns.scatterplot(data=merged_df, x=race2, y=hazard, label=f"{race2}", s=80, color="orange", edgecolor="black", alpha=0.7)

            sns.regplot(data=merged_df, x=race1, y=hazard, scatter=False, color="purple", line_kws={"linewidth": 2})
            sns.regplot(data=merged_df, x=race2, y=hazard, scatter=False, color="orange", line_kws={"linewidth": 2})

            plt.xlabel("Population Percentage", fontsize=12)
            plt.ylabel(hazard, fontsize=12)
            plt.title(f"Comparison of {hazard} Across {race1} & {race2}", fontsize=14, fontweight="bold")
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.5)

            plt.figtext(0.1, -0.12, f"Pearson Correlation:\n {race1} vs {hazard}: r = {corr1:.2f}\n {race2} vs {hazard}: r = {corr2:.2f}", 
                        fontsize=12, ha="left", bbox=dict(facecolor="white", alpha=0.8))

            plt.show()

    button = widgets.Button(description="Update Plot", button_style="primary")

    def on_button_click_2(b):
        update_plot(hazard_dropdown.value, race1_dropdown.value, race2_dropdown.value)

    button.on_click(on_button_click_2)

    display(hazard_dropdown, race1_dropdown, race2_dropdown, button, output1)


def showLAmap(save_path=None):
    """
    Creates a Folium map centered on Los Angeles County with a bounding box overlay.

    Parameters:
        save_path (str, optional): If provided, saves the map as an HTML file.

    Returns:
        folium.Map: The generated Folium map.
    """

    la_center = [34.0522, -118.2437]

    m = folium.Map(location=la_center, zoom_start=9, tiles="CartoDB Voyager")

    bounds = [[33.5, -119.0], [34.8, -117.5]]

    folium.Rectangle(
        bounds=bounds,
        color="red",
        weight=3,
        fill=True,
        fill_color="red",
        fill_opacity=0.2
    ).add_to(m)

    # Save the map if a path is provided
    if save_path:
        m.save(save_path)
        print(f"Map saved as: {save_path}")

    return m

def show_plot(aqi_states, x, y):

    sns.set_theme(style="whitegrid", palette="pastel")

    ax = sns.boxplot(
        x="state_name", 
        y="aqi", 
        data=aqi_states, 
        hue="state_name", 
        showfliers=False, 
        palette="Set3", 
        legend=False
    )

    ax.set_title("Air Quality Index (AQI) by State", fontsize=16)
    ax.set_xlabel("State", fontsize=14)
    ax.set_ylabel("AQI", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


### QUESTIONS ###

def display_text_question(question_text, q_id):
    """Displays an open-ended question with a text box for manual grading."""
    question_label = widgets.HTML(
        value=f"<div style='width:flex; white-space: normal; word-wrap: break-word; line-height: 1.50;'>{question_text}</div>"
    )
    text_area = widgets.Textarea(placeholder="Type your answer here",
                                layout=widgets.Layout(width='650px'))

    
    word_count_label = widgets.Label(value="Word count: 0")
    
    def update_word_count(change):
        count = len(change['new'].split())
        word_count_label.value = f"Word count: {count}"
    
    text_area.observe(update_word_count, names='value')
    
    button = widgets.Button(description="Submit Answer")
    button.style.button_color = 'lightblue'
    output = widgets.Output()

    def on_button_click(b):
        user_answers[q_id] = text_area.value  
        with output:
            output.clear_output()
            print("Answer submitted for manual grading.")

    button.on_click(on_button_click)
    
    display(widgets.VBox([question_label, text_area, word_count_label, button, output]))


def q_1():
    display_text_question(
        "The boxplot above shows higher AQI in California compared to the other states. Can you think of factors that might contribute to this?",
        "q1"
    )

def q_2():
    display_text_question(
        "Without running hypothesis testing, what do you immediately notice about the relationship between Pollution Burden and Health Outcomes? (Click on each bubble to display more information)",
        "q2"
    )

def q_3():
    display_text_question(
        "What do you notice about the relationship between Cardiovascular Disease Rate and Asthma Rate? ",
        "q3"
    )

def q_4():
    display_text_question(
        "Does the map shown in the interactive map below support the sentiment that minority groups are disproportionately affected by hazardous conditions? If not, what other factors might be influencing these patterns that we haven’t considered?",
        "q4"
    )

def q_5():
    display_text_question(
        "Use critical thinking skills to write no more than 120 words answering the questions above. There are no right answers to this question, so feel free to share your thoughts and opinions.",
        "q5"
    )

### OTTER GRADER ###

grader = otter.Notebook()

# Dictionary to store user answers for Otter Grader
user_answers = {}

def run_tests():
    """Runs Otter Grader tests for all questions, ensuring Q1 and Q5 are filled."""
    print("Running tests...\n")
    
    # Ensure Q1 through Q5 are answered before proceeding
    if not user_answers.get("q1", "").strip():
        print("❌ Q1: Please provide an answer in the text box before submitting.\n")
        return
    
    if not user_answers.get("q2", "").strip():
        print("❌ Q2: Please provide an answer in the text box before submitting.\n")
        return
    if not user_answers.get("q3", "").strip():
        print("❌ Q3: Please provide an answer in the text box before submitting.\n")
        return
    
    if not user_answers.get("q4", "").strip():
        print("❌ Q4: Please provide an answer in the text box before submitting.\n")
        return
    if not user_answers.get("q4", "").strip():
        print("❌ Q5: Please provide an answer in the text box before submitting.\n")
        return

    print("✅ Q1: Answer recorded for manual grading.")
    print("✅ Q2: Answer recorded for manual grading.")
    print("✅ Q3: Answer recorded for manual grading.")
    print("✅ Q4: Answer recorded for manual grading.")
    print("✅ Q5: Answer recorded for manual grading.\n")

    print("\nAll tests completed. Great work!")
