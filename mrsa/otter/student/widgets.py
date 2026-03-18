# line graph widget
def infection_rates_per_county():
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
    
    mrsa_merged = pd.read_csv('mrsa_merged.csv')

    def line_county(county):
        plt.figure(figsize=(10, 5))

        # group by year and sum only the numeric Infection_Count column
        by_year = (
            mrsa_merged.loc[mrsa_merged['County'] == county]
            .groupby('Year')['Infection_Count']
            .sum()
        )

        x = by_year.index.tolist()
        y = by_year.values.tolist()

        sns.lineplot(x=x, y=y)
        title = 'Infection Count in '+county+' County Per Year'
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Infection Count");
        plt.show()
        return 

    wid_1 = widgets.Dropdown(
            options = mrsa_merged['County'].unique().tolist(),
            description = 'County',
            disabled = False
    )

    interact(line_county, county = wid_1);

    
# widget 2
def population_v_infection_by_county():
    
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.style.use('fivethirtyeight')
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, fixed, interact_manual
    
    mrsa_merged = pd.read_csv('mrsa_merged.csv')
    infec_pop_merge = pd.read_csv('infec_pop_merge.csv')
    
    def pop_v_infec_by_county(county):    
        df = infec_pop_merge.loc[infec_pop_merge['County'] == county]  
        p = sns.lmplot(x='Year',y='Infec_Div_Pop',data=df,ci=None,height=6,aspect=2)
        title = 'Infection Count Per 100,000 People in '+county+' County'
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Infection Rate")
        plt.setp(p.ax.lines,linewidth=2)

        ylims = (-.1,2)
        if (df['Infec_Div_Pop'].min()>=2) and (df['Infec_Div_Pop'].max()<=4):
            ylims = (1.9,4)

        plt.ylim(ylims[0],ylims[1])

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
    
    mrsa_merged = pd.read_csv('mrsa_merged.csv')
    infec_pop_merge = pd.read_csv('infec_pop_merge.csv')
    
    def pop_v_infec_by_year(year):    
    
        df = infec_pop_merge.loc[infec_pop_merge['Year'] == year].copy()
        df = df.drop(df['Total_Population'].idxmax())
        df['pop_by_100k'] = df['Total_Population'] / 100000

        p = sns.lmplot(x='pop_by_100k',y='Infection_Count',data=df,ci=None,height=6,aspect=2)
        title = 'Infection Count Across Counties in Year '+ str(year)
        plt.title(title)
        plt.xlabel("Total Population Unit 100,000 People")
        plt.ylabel("Infection Count")
        plt.setp(p.ax.lines,linewidth=2)

        plt.ylim(-5, 83)

        # compute correlation using only numeric columns
        numeric_corr = df[['pop_by_100k', 'Infection_Count']].corr()
        print('Slope of Regression Line: ', numeric_corr.loc['pop_by_100k', 'Infection_Count'])
        return 
    
    wid_year = widgets.Dropdown(
            options = infec_pop_merge['Year'].unique().tolist(),
            description = 'Year',
            disabled = False
    )

    interact(pop_v_infec_by_year, year = wid_year);


# Part 9: MRSA scenario modeling widget
def mrsa_scenario_widget():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from ipywidgets import interact, IntSlider, FloatSlider

    plt.style.use('fivethirtyeight')

    # Load the MRSA data once for use in the widget model
    mrsa_for_model = pd.read_csv("mrsa_merged.csv")

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