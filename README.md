# K-Means Clustering — Data Mining TP

This repository contains a practical assignment completed for a **Data Mining** module during a master's degree. The project demonstrates how the **K-Means clustering algorithm** can be applied to synthetic student data in order to discover distinct academic profiles without using predefined labels.

## Project overview

The program generates a dataset of 100 students described by age, grades, absences, and study hours. It then standardizes the selected features, evaluates several possible values of *k* with the elbow method, applies K-Means clustering, analyzes the resulting groups, visualizes the clusters in two and three dimensions, and exports the labeled dataset to CSV.

The generated data is designed around three illustrative profiles: high-performing students, average students, and students experiencing academic difficulty. These profiles are used for educational demonstration only and should not be interpreted as a real-world assessment of students.

## Workflow

1. Generate synthetic student data.
2. Select the clustering variables.
3. Standardize the variables with `StandardScaler`.
4. Compute within-cluster sum of squares for values of *k* from 2 to 10.
5. Select a value of *k* interactively and fit the K-Means model.
6. Display descriptive statistics for each cluster.
7. Produce 2D and 3D visualizations.
8. Export the clustered data to `resultats_kmeans.csv`.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Implementation language |
| NumPy | Numerical data generation |
| pandas | Data manipulation and CSV export |
| scikit-learn | Standardization and K-Means clustering |
| Matplotlib | Data visualization |
| Seaborn | Plot styling |

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

## Usage

```bash
python kmeans_TP.py
```

When prompted, enter the desired number of clusters. The default value is `3`, which corresponds to the three synthetic profiles used to generate the data.

The program generates the following files in the working directory:

- `methode_coude.png`
- `clusters_2d.png`
- `clusters_3d.png`
- `resultats_kmeans.csv`

## Learning objectives

This TP illustrates the main stages of an unsupervised-learning workflow: feature preparation, scaling, selection of the number of clusters, model fitting, cluster interpretation, visualization, and result export.
