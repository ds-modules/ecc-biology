# Load data relative to this file so widgets work
# no matter where the notebook kernel starts.
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent


def _data_path(filename):
    return DATA_DIR / filename


# line graph widget
def infection_rates_per_county():
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import interactive
    
    mrsa_merged = pd.read_csv(_data_path('mrsa_merged.csv'))

    def line_county(county):
        fig, ax = plt.subplots(figsize=(10, 5))

        # group by year and sum only the numeric Infection_Count column
        by_year = (
            mrsa_merged.loc[mrsa_merged['County'] == county]
            .groupby('Year')['Infection_Count']
            .sum()
        )

        x = by_year.index.tolist()
        y = by_year.values.tolist()

        sns.lineplot(x=x, y=y, ax=ax)
        title = 'Infection Count in '+county+' County Per Year'
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Infection Count")
        plt.show()
        plt.close(fig)
        return 

    wid_1 = widgets.Dropdown(
            options = mrsa_merged['County'].unique().tolist(),
            description = 'County',
            disabled = False
    )

    widget = interactive(line_county, county=wid_1)
    return widget

    
# widget 2
def population_v_infection_by_county():
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
    
    mrsa_merged = pd.read_csv(_data_path('mrsa_merged.csv'))
    infec_pop_merge = pd.read_csv(_data_path('infec_pop_merge.csv'))
    
    def pop_v_infec_by_county(county):    
        df = infec_pop_merge.loc[infec_pop_merge['County'] == county]  
        p = sns.lmplot(x='Year',y='Infec_Div_Pop',data=df,ci=None,height=6,aspect=2)
        title = 'Infection Count Per 100,000 People in '+county+' County'
        p.ax.set_title(title)
        p.ax.set_xlabel("Year")
        p.ax.set_ylabel("Infection Rate")
        plt.setp(p.ax.lines,linewidth=2)

        ylims = (-.1,2)
        if (df['Infec_Div_Pop'].min()>=2) and (df['Infec_Div_Pop'].max()<=4):
            ylims = (1.9,4)

        p.ax.set_ylim(ylims[0],ylims[1])
        plt.show()
        plt.close(p.fig)

       # print('Correlation: ',df.corr()['Total_Population']['Infection_Count'])
        return 

    wid_2 = widgets.Dropdown(
            options = infec_pop_merge['County'].unique().tolist(),
            description = 'County',
            disabled = False
    )

    interact(pop_v_infec_by_county, county = wid_2);
    

# year widget
def population_vs_infection_by_year():
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
    
    mrsa_merged = pd.read_csv(_data_path('mrsa_merged.csv'))
    infec_pop_merge = pd.read_csv(_data_path('infec_pop_merge.csv'))
    
    def pop_v_infec_by_year(year):    
    
        df = infec_pop_merge.loc[infec_pop_merge['Year'] == year].copy()
        df = df.drop(df['Total_Population'].idxmax())
        df['pop_by_100k'] = df['Total_Population'] / 100000

        p = sns.lmplot(x='pop_by_100k',y='Infection_Count',data=df,ci=None,height=6,aspect=2)
        title = 'Infection Count Across Counties in Year '+ str(year)
        p.ax.set_title(title)
        p.ax.set_xlabel("Total Population Unit 100,000 People")
        p.ax.set_ylabel("Infection Count")
        plt.setp(p.ax.lines,linewidth=2)

        p.ax.set_ylim(-5, 83)

        # compute correlation using only numeric columns
        numeric_corr = df[['pop_by_100k', 'Infection_Count']].corr()
        print('Slope of Regression Line: ', numeric_corr.loc['pop_by_100k', 'Infection_Count'])
        plt.show()
        plt.close(p.fig)
        return 
    
    wid_year = widgets.Dropdown(
            options = infec_pop_merge['Year'].unique().tolist(),
            description = 'Year',
            disabled = False
    )

    interact(pop_v_infec_by_year, year = wid_year);


def census_race_totals_widget():
    """Race + sex totals vs year: county or statewide average; shows combined total and both parts.

    Matches the notebook pattern (e.g. Asian Male + Asian Female) with an option to average
    across all counties per year.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from ipywidgets import interactive

    plt.style.use("fivethirtyeight")

    raw = pd.read_csv(_data_path("census_bonus.csv"))
    grouped = raw.groupby(["Year", "County"], as_index=False).sum(numeric_only=True)

    # (label shown in dropdown, male column, female column) — CSV column names
    race_pairs = [
        ("Asian", "Asian Male", "Asian Female"),
        ("Black", "Black Male", "Black Female"),
        ("White", "White Male", "White Female"),
        ("American Indian", "American Indian Male", "American Indian Female"),
        ("Native Hawaiian", "Native Hawaiian Male", "Native Hawaiian Female"),
        ("2+ Race", "2+ Race Male", "2+ Race Female"),
        ("Hispanic (ethnicity)", "Hispanic Male", "Hispanic Female"),
        ("Not Hispanic (ethnicity)", "Not Hispanic Male", "Not Hispanic Female"),
    ]

    county_options = ["All counties (average)"] + sorted(grouped["County"].unique())

    def plot_race(county, race_label):
        pair = next(p for p in race_pairs if p[0] == race_label)
        _, male_c, female_c = pair

        work = grouped[["Year", "County", male_c, female_c]].copy()
        work["race_total"] = work[male_c] + work[female_c]

        fig, ax = plt.subplots(figsize=(10, 5))

        if county == "All counties (average)":
            agg = (
                work.groupby("Year", as_index=False)
                .agg(
                    avg_race_total=("race_total", "mean"),
                    avg_male=(male_c, "mean"),
                    avg_female=(female_c, "mean"),
                )
                .sort_values("Year")
            )
            ax.plot(agg["Year"], agg["avg_race_total"], label="Average race total (all counties)", linewidth=2.5)
            ax.plot(agg["Year"], agg["avg_male"], label=f"Average {male_c}", alpha=0.9)
            ax.plot(agg["Year"], agg["avg_female"], label=f"Average {female_c}", alpha=0.9)
            title = f"{race_label}: average per year (mean across counties)"
        else:
            one = work.loc[work["County"] == county].sort_values("Year")
            ax.plot(one["Year"], one["race_total"], label="Race total (male + female)", linewidth=2.5)
            ax.plot(one["Year"], one[male_c], label=male_c, alpha=0.9)
            ax.plot(one["Year"], one[female_c], label=female_c, alpha=0.9)
            title = f"{race_label} in {county}"

        ax.set_xlabel("Year")
        ax.set_ylabel("Population")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    county_dd = widgets.Dropdown(
        options=county_options,
        value=county_options[1] if len(county_options) > 1 else county_options[0],
        description="County",
        layout=widgets.Layout(width="420px"),
    )
    race_dd = widgets.Dropdown(
        options=[p[0] for p in race_pairs],
        value="Asian",
        description="Group",
        layout=widgets.Layout(width="320px"),
    )

    return interactive(plot_race, county=county_dd, race_label=race_dd)


def census_race_multiselect_widget():
    """Compare whole-race totals (male + female combined) on one plot with multi-select.

    - Each race is a single series: Asian = Asian Male + Asian Female (same for other groups).
    - County dropdown plus multi-select races to overlay lines.
    - Shows a table of mean race total per county (average across years) for the selected races.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    plt.style.use("fivethirtyeight")

    raw = pd.read_csv(_data_path("census_bonus.csv"))
    grouped = raw.groupby(["Year", "County"], as_index=False).sum(numeric_only=True)

    race_pairs = [
        ("Asian", "Asian Male", "Asian Female"),
        ("Black", "Black Male", "Black Female"),
        ("White", "White Male", "White Female"),
        ("American Indian", "American Indian Male", "American Indian Female"),
        ("Native Hawaiian", "Native Hawaiian Male", "Native Hawaiian Female"),
        ("2+ Race", "2+ Race Male", "2+ Race Female"),
        ("Hispanic (ethnicity)", "Hispanic Male", "Hispanic Female"),
        ("Not Hispanic (ethnicity)", "Not Hispanic Male", "Not Hispanic Female"),
    ]

    labels = [p[0] for p in race_pairs]
    wide = grouped[["Year", "County"]].copy()
    for label, male_c, female_c in race_pairs:
        wide[label] = grouped[male_c].values + grouped[female_c].values

    county_options = ["All counties (average)"] + sorted(grouped["County"].unique())

    county_dd = widgets.Dropdown(
        options=county_options,
        value=county_options[1] if len(county_options) > 1 else county_options[0],
        description="County",
        layout=widgets.Layout(width="420px"),
    )
    race_checks = []
    default_selected = {"Asian", "Black", "White"}
    for label in labels:
        race_checks.append(
            widgets.Checkbox(
                value=label in default_selected,
                description=label,
                indent=False,
                layout=widgets.Layout(width="260px"),
            )
        )
    select_all_btn = widgets.Button(description="Select all", layout=widgets.Layout(width="120px"))
    clear_all_btn = widgets.Button(description="Clear all", layout=widgets.Layout(width="120px"))
    race_controls = widgets.VBox(
        [
            widgets.HTML("<b>Races</b>"),
            widgets.HBox([select_all_btn, clear_all_btn]),
            widgets.VBox(race_checks),
        ],
        layout=widgets.Layout(width="380px"),
    )
    out = widgets.Output()

    def selected_races():
        return tuple(cb.description for cb in race_checks if cb.value)

    def render(county, race_labels):
        with out:
            clear_output(wait=True)
            if not race_labels:
                display(
                    widgets.HTML(
                        "<p><b>Select at least one race</b> using the checkboxes.</p>"
                    )
                )
                return

            fig, ax = plt.subplots(figsize=(11, 5))
            n = len(race_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

            if county == "All counties (average)":
                for i, race in enumerate(race_labels):
                    agg = (
                        wide.groupby("Year")[race]
                        .mean()
                        .reset_index()
                        .sort_values("Year")
                    )
                    ax.plot(
                        agg["Year"],
                        agg[race],
                        label=f"{race} (mean across counties)",
                        color=colors[i % 10],
                        linewidth=2,
                    )
                overall_all = wide.groupby("Year")[labels].mean().mean(axis=1).reset_index()
                overall_all.columns = ["Year", "overall_avg_all_races"]
                ax.plot(
                    overall_all["Year"],
                    overall_all["overall_avg_all_races"],
                    color="black",
                    linewidth=3,
                    linestyle="--",
                    label="Overall average across all races",
                )
                ax.set_title("Mean whole-race total per year (averaged across counties)")
            else:
                sub = wide.loc[wide["County"] == county].sort_values("Year")
                for i, race in enumerate(race_labels):
                    ax.plot(
                        sub["Year"],
                        sub[race],
                        label=race,
                        color=colors[i % 10],
                        linewidth=2.4,
                    )
                overall_county = sub[labels].mean(axis=1)
                ax.plot(
                    sub["Year"],
                    overall_county,
                    color="black",
                    linewidth=3,
                    linestyle="--",
                    label="Overall average across all races",
                )
                ax.set_title(f"Whole-race totals (male + female) — {county}")

            ax.set_xlabel("Year")
            ax.set_ylabel("Population (race total)")
            ax.legend(loc="best", fontsize=11)
            plt.tight_layout()
            plt.show()
            plt.close(fig)

    def refresh(_=None):
        render(county_dd.value, selected_races())

    def on_select_all(_):
        for cb in race_checks:
            cb.value = True
        refresh()

    def on_clear_all(_):
        for cb in race_checks:
            cb.value = False
        refresh()

    county_dd.observe(refresh, names="value")
    for cb in race_checks:
        cb.observe(refresh, names="value")
    select_all_btn.on_click(on_select_all)
    clear_all_btn.on_click(on_clear_all)

    refresh()

    return widgets.VBox([widgets.HBox([county_dd, race_controls]), out])


# Part 9: MRSA scenario modeling widget
def mrsa_scenario_widget():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from ipywidgets import interact, IntSlider, FloatSlider

    plt.style.use('fivethirtyeight')

    # Load the MRSA data once for use in the widget model
    mrsa_for_model = pd.read_csv(_data_path("mrsa_merged.csv"))

    def estimate_post_intervention_growth_from_data(intervention_year=2015):
        """Estimate an average monthly growth rate after an 'intervention year' from real MRSA data."""
        # aggregate total infections per year across all counties
        by_year = (
            mrsa_for_model
            .groupby("Year")["Infection_Count"]
            .sum()
            .sort_index()
        )

        # use years strictly after the intervention_year as "post intervention"
        post = by_year[by_year.index > intervention_year]

        # safety check: if there are not enough years after intervention, fall back to zero growth
        if len(post) < 2:
            return 0.0

        # fit an exponential trend: log(count) ~ a + b * year
        years = post.index.values.astype(float)
        log_counts = np.log(post.values + 1e-6)  # avoid log(0)

        b_post, a_post = np.polyfit(years, log_counts, 1)
        annual_rate = np.exp(b_post) - 1.0
        # convert annual to monthly growth rate
        monthly_rate = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
        # return as a percentage (e.g., -3.2 means ~3.2% decrease per month)
        return monthly_rate * 100.0

    def mrsa_growth_model(initial_cases=20,
                          monthly_growth_percent=5.0,
                          intervention_month=6):
        """Simple scenario model linked to real MRSA data.

        Students control:
          - initial_cases
          - monthly_growth_percent before the intervention
          - intervention_month (in months)

        The post-intervention growth rate is estimated automatically
        from the MRSA dataset using trends after a chosen intervention year.
        """
        months = 24  # fixed 2-year window for the scenario

        # estimate post-intervention monthly growth from data
        post_intervention_growth_percent = estimate_post_intervention_growth_from_data()

        months_array = np.arange(0, months + 1)
        cases = [initial_cases]

        for m in range(1, months + 1):
            if m < intervention_month:
                r = monthly_growth_percent / 100.0
            else:
                r = post_intervention_growth_percent / 100.0
            next_val = cases[-1] * (1.0 + r)
            cases.append(next_val)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(months_array, cases, marker="o", label="Scenario cases")
        ax.axvline(intervention_month, color="red", linestyle="--", label="Intervention starts")

        # small annotation so students know where the post-intervention rate comes from
        ax.text(
            0.99,
            0.02,
            f"Data-driven growth after: {post_intervention_growth_percent:.1f}% / month",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Number of MRSA bloodstream infections (scenario)")
        ax.set_title("MRSA Scenario Model Linked to Real Data")
        ax.legend()
        ax.grid(True)

        return fig

    interact(
        mrsa_growth_model,
        initial_cases=IntSlider(min=1, max=200, step=1, value=20,
                                description="Init cases"),
        monthly_growth_percent=FloatSlider(min=-20, max=40, step=1, value=5,
                                           description="Growth before (%)"),
        intervention_month=IntSlider(min=1, max=24, step=1, value=6,
                                     description="Intervention mo."),
    )