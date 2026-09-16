# NASA Landslide & Precipitation Analysis

**Gary Dhillon · IST 652 Final Project**

An analysis of reported landslides using NASA's Global Landslide Catalog and a rainfall dataset attributed to NASA POWER. This Python notebook combines data cleaning, exploratory analysis, geographic visualization, classification, natural language processing, and linear regression to investigate landslide patterns and the limits of the available data.

[View the project notebook](Dhillon_Gurdip_Final_Project.ipynb)

## Project overview

The project explores when and where landslides are reported, which triggers and event characteristics are associated with severe outcomes, and whether annual precipitation is associated with annual reported landslide counts.

| Component | Scope |
| --- | --- |
| Landslide dataset | 11,033 records |
| Recorded event dates | November 7, 1988–September 28, 2017 |
| Rainfall dataset | 252 records and 7 columns in the saved output |
| Text analysis | 10,171 nonmissing event descriptions |
| Classification | Histogram-based gradient boosting and an RBF support vector machine |
| Statistical analysis | Pearson correlation and simple linear regression |
| Environment | Python/Jupyter; notebook metadata records Python 3.13.5 |

## Research questions

- How do recorded landslide events vary by year, month, location, and size?
- Which categories and triggers are associated with reported fatalities?
- How do mining, construction, and other selected human-related triggers compare?
- Can event characteristics classify whether a storm name is recorded or distinguish fatality severity?
- What themes and sentiment appear in event descriptions?
- How do annual precipitation aggregates relate to annual reported landslide counts?

## Data sources

| Input | Source | Use |
| --- | --- | --- |
| `Global_Landslide_Catalog_Export.csv` | [Global Landslide Catalog export on Kaggle](https://www.kaggle.com/datasets/pratul007/global-landslide-catalog-export) | Event dates, coordinates, categories, triggers, size, fatalities, injuries, descriptions, and reporting sources |
| `Rainfall_data.csv` | [Rainfall Timeseries data on Kaggle](https://www.kaggle.com/datasets/poojag718/rainfall-timeseries-data) | Precipitation grouped by `Year` and joined to yearly landslide counts |
| `ne_110m_admin_0_countries.shp` and companion files | [Natural Earth: Admin 0 – Countries, 1:110m](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/) | Country boundaries for geographic visualizations |

The notebook attributes the rainfall data to NASA POWER. Its displayed records have monthly-looking dates with `Day = 1`, although the notebook narrative describes daily observations. Confirm the source's geographic coverage, temporal resolution, and precipitation units before interpreting the yearly sums as annual rainfall totals.

## Analysis workflow

1. **Clean and inspect the catalog.** Parse event dates, derive year/month/hour features, inspect missing values, flag repeated date/time/location combinations, and examine midnight timestamps.
2. **Explore temporal and geographic patterns.** Aggregate events by time, map coordinates using GeoPandas and Shapely, and compare overall records with records containing storm names.
3. **Examine severity and triggers.** Analyze size categories, fatality totals and outliers, seasonal trigger patterns, and selected human-related triggers.
4. **Train and evaluate classifiers.** Use stratified train/test splits, classification reports, and confusion matrices for storm-name presence and fatality severity.
5. **Analyze event descriptions.** Apply regular-expression tokenization, Snowball stemming, English stopword filtering, term counts, a word cloud, and VADER sentiment scoring.
6. **Compare precipitation and landslide counts.** Aggregate precipitation by year, join on year, visualize the series, and calculate correlation and regression statistics.

Reusable helper functions support date preparation, filtering, aggregation, normalization, plotting, geographic conversion, and classifier evaluation.

## Selected findings

The values below come from the notebook's saved outputs.

| Finding | Recorded result |
| --- | --- |
| Most frequent trigger | `downpour`, with 4,680 records |
| Most frequent size | `medium`, with 6,551 records |
| Midnight timestamps | 5,533 records, or 50.15%; potentially placeholder times |
| Mining share | 47.21% of the subset with mining, construction, dam embankment collapse, or leaking-pipe triggers |
| Text representation | 10,171 descriptions × 11,209 retained stemmed terms |
| Mean VADER compound score | Approximately −0.365 |
| Rainfall/count correlation | Approximately −0.06 |
| Rainfall regression | R² = 0.0030; slope p-value = 0.8456 |

The annual rainfall comparison shows little linear association in the joined sample. It does not establish that rainfall is unimportant to landslides: the analysis aggregates by year and does not match precipitation to individual event locations or dates.

### Geographic distribution

![Global distribution of reported landslides from the notebook](assets/global-landslide-map.png)

The map shows catalog records rather than a population-adjusted or reporting-adjusted measure of landslide risk.

### Precipitation comparison

![Annual precipitation aggregate versus reported landslide count, with fitted regression line](assets/rainfall-regression.png)

Both figures are extracted directly from the notebook's saved outputs.

## Classification experiments

### Storm-name presence

A `HistGradientBoostingClassifier(random_state=42)` classifies whether `storm_name` is nonempty. Predictors are longitude, latitude, event month, numeric size, fatality count, and injury count. The split is stratified, with 70% training data and 30% test data (`random_state=42`). The classifier uses the unscaled features; the separately calculated scaled arrays are not passed to this model.

| Metric | Saved test result |
| --- | ---: |
| Accuracy | 96.3% |
| Storm-present precision | 0.679 |
| Storm-present recall | 0.538 |
| Storm-present F1 | 0.600 |
| Test records | 3,310 |

Only 173 test records have a storm name. Always predicting the majority class would achieve approximately 94.8% accuracy, so minority-class recall and F1 provide essential context. The target measures recorded storm-name presence; a missing name does not prove an event was unrelated to a storm.

### Fatality severity

A pipeline combines mean imputation, standard scaling, and an RBF `SVC(class_weight="balanced")`. Predictors are longitude, latitude, month, numeric size, storm-name presence, and injury count. Fatality count defines the target and is excluded from the predictors.

The notebook retains records with `fatality_count < 1750`, which also excludes missing fatality counts, leaving 9,645 records. Severity labels are `none` (0), `low` (1–99), `medium` (100–499), and `high` (500 or more). No `high` examples remain in the saved filtered data. The stratified split uses 40% training data and 60% test data (`random_state=123`).

| Metric | Saved test result |
| --- | ---: |
| Accuracy | 65.9% |
| Macro F1 | 0.445 |
| Low-severity recall | 0.880 |
| Medium-severity F1 | 0.023 |
| Test records | 5,787 |

The test set contains only 18 medium-severity examples. An always-`none` baseline would achieve approximately 74.7% accuracy, exceeding the SVM's overall accuracy. This experiment demonstrates the difficulty of learning rare severity classes; it is not a validated hazard forecasting system.

## Running the notebook

### 1. Arrange the input files

Download the datasets and country boundary files from the links above. Use this structure to match the notebook's relative paths:

```text
.
├── README.md
├── Dhillon_Gurdip_Final_Project.ipynb
├── Global_Landslide_Catalog_Export.csv
├── Rainfall_data.csv
├── assets/
│   ├── global-landslide-map.png
│   └── rainfall-regression.png
└── world_shapefile/
    └── world_shapefile/
        ├── ne_110m_admin_0_countries.shp
        ├── ne_110m_admin_0_countries.shx
        ├── ne_110m_admin_0_countries.dbf
        └── ne_110m_admin_0_countries.prj
```

Keep the shapefile's companion files together, including any additional files supplied with the download. The nested `world_shapefile/world_shapefile/` directory matches the path used by `plot_world_events()`; update that function if you use another location. The README package includes the preview figures, while the notebook and input datasets must be added separately.

### 2. Install dependencies

In your Python environment, run:

```bash
python -m pip install jupyterlab ipykernel pandas numpy matplotlib seaborn geopandas shapely scipy scikit-learn nltk wordcloud
python -m nltk.downloader stopwords vader_lexicon
```

The NLTK resources are required for the text-analysis cells. Package versions are not pinned in the supplied notebook, so exact reproducibility across environments is not guaranteed.

### 3. Launch and execute

From the directory containing the notebook and CSV files, run:

```bash
python -m jupyterlab
```

Open `Dhillon_Gurdip_Final_Project.ipynb`, select a kernel with the installed dependencies, and run the cells from top to bottom. Maps, charts, tables, word clouds, and evaluation reports display inline. The SVM fitting step may take longer than the exploratory analysis cells.

## Interpretation and limitations

- **Reporting coverage:** These are reported events. Changes in source coverage and missing records can affect geographic and temporal comparisons. Candidate duplicate records are inspected but not removed from the main dataset.
- **Time and season labels:** Midnight may indicate an unknown time. The global seasonal analysis applies Northern Hemisphere month-to-season labels to all locations.
- **Missing values and filters:** Missing fatalities are not equivalent to zero fatalities. Some fatality charts exclude values above 4,000, while the severity classifier uses a separate cutoff of 1,750; these subsets answer different questions.
- **Model scope:** Injury and fatality information is known after an event. The storm-name model uses both, and the severity model uses injury counts, so these experiments should not be interpreted as forecasts made before an event. Random splits do not test performance on future years or unseen regions.
- **Rainfall alignment:** The join uses year alone. Unverified rainfall geography, resolution, and units limit the meaning of the resulting correlation.
- **Text and summary metrics:** VADER describes the tone of reports, not physical hazard severity. The population summary's `fatalities_count` column counts nonmissing fatality entries rather than summing deaths.

## Future improvements

- Match precipitation to landslide coordinates and dates, including antecedent rainfall windows.
- Resolve duplicate candidates and document a consistent missing-data strategy.
- Compare classifiers with explicit baselines, cross-validation, and temporal or geographic holdouts.
- Improve evaluation of rare classes and retain a separate analysis of catastrophic events.
- Pin dependency versions and add automated checks for input schemas and data quality.

## Review status

This README was checked against the notebook's code and saved outputs. The notebook was not rerun during this documentation review; the original CSV files and boundary data were not supplied with it. Reported results reflect the saved execution rather than a newly reproduced run.
