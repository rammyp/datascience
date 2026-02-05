# Seaborn Complete Study Guide

> **With Explanations, Code Examples & Quick Reference**
> Python Statistical Data Visualization

---

## 1. Installation & Import

Seaborn is a statistical data visualization library built on top of Matplotlib. It provides a higher-level interface that makes it easier to create attractive, informative plots with less code. Seaborn is designed to work directly with Pandas DataFrames, automatically handles grouping and aggregation, and produces publication-quality visuals with sensible defaults. Think of it as Matplotlib's smarter, prettier sibling.

```bash
pip install seaborn
```

```python
import seaborn as sns
import matplotlib.pyplot as plt    # still needed for plt.show(), customization
import pandas as pd
import numpy as np
```

> ⚡ Seaborn works best with **tidy data** (each row is an observation, each column is a variable). If your data is in this format, Seaborn can do most of the heavy lifting automatically.

---

## 2. Built-in Datasets & Themes

Seaborn includes several practice datasets and built-in themes that instantly improve the look of any plot.

### Datasets

```python
# See all available datasets
sns.get_dataset_names()

# Load commonly used datasets
tips = sns.load_dataset('tips')           # restaurant tips
iris = sns.load_dataset('iris')           # flower measurements
penguins = sns.load_dataset('penguins')   # penguin species
titanic = sns.load_dataset('titanic')     # Titanic passengers
flights = sns.load_dataset('flights')     # airline passengers
diamonds = sns.load_dataset('diamonds')   # diamond prices
```

### Themes & Styles

Seaborn themes control the overall look of all plots. Set them once at the top of your script and every subsequent plot inherits the style.

```python
# Set theme (applies to all plots)
sns.set_theme()                           # default (nice and clean)
sns.set_theme(style='whitegrid')          # white background with grid
sns.set_theme(style='darkgrid')           # gray background with grid
sns.set_theme(style='white')              # white, no grid
sns.set_theme(style='dark')               # dark gray, no grid
sns.set_theme(style='ticks')              # white with tick marks

# Customize further
sns.set_theme(
    style='whitegrid',
    palette='deep',           # color palette
    font='Arial',             # font family
    font_scale=1.2            # scale all text 20% larger
)

# Color palettes
sns.set_palette('deep')        # default
sns.set_palette('muted')       # softer colors
sns.set_palette('pastel')      # light colors
sns.set_palette('bright')      # vivid colors
sns.set_palette('dark')        # darker variants
sns.set_palette('colorblind')  # accessible colors

# Custom palette
custom = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
sns.set_palette(custom)

# Remove top and right spines
sns.despine()
```

---

## 3. Relational Plots

Relational plots show the relationship between two or more numerical variables. `scatterplot` shows individual data points; `lineplot` connects them with lines (great for time series). The `hue`, `size`, and `style` parameters let you encode additional variables without creating separate plots.

### Scatter Plot

```python
tips = sns.load_dataset('tips')

# Basic scatter
sns.scatterplot(data=tips, x='total_bill', y='tip')
plt.show()

# With grouping variables
sns.scatterplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='smoker',         # color by smoker status
    size='size',          # point size by party size
    style='time',         # marker shape by meal time
    alpha=0.7
)
plt.title('Tips by Total Bill')
plt.show()
```

### Line Plot

```python
flights = sns.load_dataset('flights')

# Line plot with confidence interval
sns.lineplot(data=flights, x='year', y='passengers', hue='month')
plt.title('Airline Passengers Over Time')
plt.show()

# Aggregated line (mean ± CI)
sns.lineplot(data=flights, x='year', y='passengers')   # auto-aggregates
plt.show()
```

### relplot (Figure-level — supports faceting)

```python
# Create a grid of scatter plots
sns.relplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='smoker',
    col='time',          # separate columns for lunch/dinner
    row='sex',           # separate rows for male/female
    kind='scatter'
)
plt.show()
```

---

## 4. Distribution Plots

Distribution plots show how values in a dataset are spread out. They answer questions like "what's the most common value?" and "is my data symmetric or skewed?" Seaborn offers several ways to visualize distributions, from simple histograms to smooth density curves.

### Histogram

```python
# Basic histogram
sns.histplot(data=tips, x='total_bill', bins=20)
plt.show()

# With KDE overlay
sns.histplot(data=tips, x='total_bill', kde=True, color='steelblue')
plt.show()

# Grouped histogram
sns.histplot(data=tips, x='total_bill', hue='time', element='step')
plt.show()

# 2D histogram
sns.histplot(data=tips, x='total_bill', y='tip', cbar=True)
plt.show()
```

### KDE Plot (Kernel Density Estimate)

KDE creates a smooth curve that estimates the probability density of the data. It's like a smoothed-out histogram. Useful when you want to compare shapes of distributions without the jagged edges of histogram bins.

```python
# Single distribution
sns.kdeplot(data=tips, x='total_bill', fill=True, alpha=0.5)
plt.show()

# Compare multiple groups
sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True, alpha=0.4)
plt.show()

# 2D density
sns.kdeplot(data=tips, x='total_bill', y='tip', fill=True, cmap='Blues')
plt.show()
```

### displot (Figure-level)

```python
# Histogram with facets
sns.displot(data=tips, x='total_bill', hue='time', col='sex', kind='hist')
plt.show()

# KDE with facets
sns.displot(data=tips, x='total_bill', hue='smoker', kind='kde', fill=True)
plt.show()

# ECDF — cumulative distribution
sns.displot(data=tips, x='total_bill', hue='time', kind='ecdf')
plt.show()
```

---

## 5. Categorical Plots

Categorical plots visualize data where one variable is a category (like "day of week" or "gender"). Seaborn excels here — it automatically groups, sorts, and styles categorical data. These are some of the most frequently used Seaborn functions.

### Box Plot

Shows the distribution using quartiles. The box spans Q1 to Q3, the line inside is the median, and whiskers extend to 1.5×IQR. Points beyond are outliers. Great for comparing multiple groups at a glance.

```python
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex', palette='Set2')
plt.title('Bill Distribution by Day and Gender')
plt.show()
```

### Violin Plot

Combines a box plot with a KDE. The width shows the density of values at that level. More informative than a box plot because you can see the full shape of the distribution (bimodal, skewed, etc.).

```python
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True, inner='quartile')
plt.title('Bill Distribution (Violin)')
plt.show()
```

### Strip Plot & Swarm Plot

Show individual data points. Strip plots can overlap; swarm plots spread points out so every one is visible. Best for smaller datasets where you want to see every observation.

```python
# Strip (with jitter)
sns.stripplot(data=tips, x='day', y='total_bill', hue='sex', dodge=True, alpha=0.6)
plt.show()

# Swarm (no overlap — every point visible)
sns.swarmplot(data=tips, x='day', y='total_bill', hue='sex', dodge=True)
plt.show()
```

### Bar Plot (with error bars)

Shows the mean (or other statistic) of a numerical variable for each category. Error bars show 95% confidence intervals by default. This is NOT the same as a simple count bar chart — it's a statistical summary.

```python
sns.barplot(data=tips, x='day', y='total_bill', hue='sex', palette='coolwarm', errorbar='sd')
plt.title('Average Bill by Day')
plt.show()
```

### Count Plot

Counts the number of observations in each category. It's like a histogram for categorical data.

```python
sns.countplot(data=tips, x='day', hue='sex', palette='pastel')
plt.title('Visits by Day')
plt.show()
```

### catplot (Figure-level — supports faceting)

```python
sns.catplot(data=tips, x='day', y='total_bill', hue='sex',
            col='time', kind='box', height=5, aspect=1.2)
plt.show()
```

---

## 6. Matrix Plots (Heatmaps)

Heatmaps display 2D data as a color-coded grid. They're most commonly used for correlation matrices, confusion matrices, and pivot table summaries. Seaborn's `heatmap()` function is much easier to use than Matplotlib's raw `imshow()`.

### Correlation Heatmap

```python
# Compute correlation matrix
corr = tips[['total_bill', 'tip', 'size']].corr()

# Plot
sns.heatmap(
    corr,
    annot=True,            # show numbers in cells
    fmt='.2f',             # format to 2 decimal places
    cmap='coolwarm',       # red-blue color scale
    center=0,              # center colormap at 0
    vmin=-1, vmax=1,       # fix scale from -1 to 1
    square=True,           # square cells
    linewidths=0.5         # lines between cells
)
plt.title('Correlation Matrix')
plt.show()
```

### Pivot Heatmap

```python
# Pivot data first
flights = sns.load_dataset('flights')
pivot = flights.pivot(index='month', columns='year', values='passengers')

# Plot
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlGnBu', linewidths=0.5)
plt.title('Airline Passengers by Month and Year')
plt.show()
```

### Cluster Map

Automatically groups similar rows and columns together using hierarchical clustering.

```python
sns.clustermap(pivot, cmap='viridis', standard_scale=1, figsize=(10, 10))
plt.show()
```

---

## 7. Regression Plots

Regression plots show the relationship between variables and fit a regression line. `regplot` is axes-level (plots on a single axes); `lmplot` is figure-level (supports faceting). They automatically compute and display the regression line and confidence interval.

```python
# Basic regression
sns.regplot(data=tips, x='total_bill', y='tip', scatter_kws={'alpha': 0.5})
plt.title('Tip vs Total Bill (with regression line)')
plt.show()

# With facets
sns.lmplot(data=tips, x='total_bill', y='tip', hue='smoker', col='time')
plt.show()

# Polynomial regression
sns.regplot(data=tips, x='total_bill', y='tip', order=2)    # quadratic fit
plt.show()

# Residual plot — check if model is appropriate
sns.residplot(data=tips, x='total_bill', y='tip')
plt.title('Residuals')
plt.show()
```

---

## 8. Pair Plot

A pair plot creates a grid of scatter plots for every pair of numerical variables, with histograms on the diagonal. It's the fastest way to explore relationships across an entire dataset. Best for datasets with 2–6 numerical columns.

```python
# Basic pair plot
sns.pairplot(iris, hue='species')
plt.show()

# Customize
sns.pairplot(
    iris,
    hue='species',
    diag_kind='kde',                  # KDE instead of histogram on diagonal
    plot_kws={'alpha': 0.6},          # transparency for scatter points
    palette='Set2',
    corner=True                        # only lower triangle
)
plt.show()
```

---

## 9. Joint Plot

A joint plot combines a scatter plot (or hex/KDE) with marginal distributions on the x and y axes. It's a compact way to visualize the relationship AND distribution of two variables in one figure.

```python
# Scatter with marginal histograms
sns.jointplot(data=tips, x='total_bill', y='tip', kind='scatter')
plt.show()

# KDE (smooth density)
sns.jointplot(data=tips, x='total_bill', y='tip', kind='kde', fill=True)
plt.show()

# Hex (for large datasets)
sns.jointplot(data=tips, x='total_bill', y='tip', kind='hex')
plt.show()

# With hue
sns.jointplot(data=tips, x='total_bill', y='tip', hue='smoker')
plt.show()

# Regression line
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')
plt.show()
```

---

## 10. FacetGrid

FacetGrid creates a grid of plots, one for each combination of categorical variables. It's the most flexible way to create multi-panel figures. You define the grid structure, then map a plot function onto it.

```python
# Create grid and map a plot
g = sns.FacetGrid(tips, col='time', row='smoker', hue='sex', height=4)
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip', alpha=0.7)
g.add_legend()
plt.show()

# With histogram
g = sns.FacetGrid(tips, col='day', col_wrap=2, height=4)
g.map_dataframe(sns.histplot, x='total_bill', kde=True)
g.set_titles(col_template='{col_name}')
plt.show()
```

---

## 11. Axes-Level vs Figure-Level

Understanding this distinction is key to working effectively with Seaborn. Axes-level functions plot onto a specific Matplotlib axes. Figure-level functions create their own figure and support faceting with `row` and `col` parameters.

| Axes-Level | Figure-Level | Category |
|------------|-------------|----------|
| `scatterplot` | `relplot(kind='scatter')` | Relational |
| `lineplot` | `relplot(kind='line')` | Relational |
| `histplot` | `displot(kind='hist')` | Distribution |
| `kdeplot` | `displot(kind='kde')` | Distribution |
| `boxplot` | `catplot(kind='box')` | Categorical |
| `violinplot` | `catplot(kind='violin')` | Categorical |
| `barplot` | `catplot(kind='bar')` | Categorical |
| `countplot` | `catplot(kind='count')` | Categorical |
| `regplot` | `lmplot` | Regression |
| `heatmap` | `clustermap` | Matrix |

```python
# Axes-level — plots on a specific axes (more control)
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=tips, x='total_bill', ax=ax)
ax.set_title('Custom Title')
plt.show()

# Figure-level — creates its own figure (supports faceting)
g = sns.displot(data=tips, x='total_bill', col='time', kind='hist')
g.fig.suptitle('By Meal Time', y=1.02)
plt.show()
```

> ⚡ Use **axes-level** when you need precise Matplotlib control. Use **figure-level** when you want faceting across categories.

---

## 12. Color Palettes Deep Dive

Choosing the right color palette makes your plots clearer and more accessible. Seaborn has three types: sequential (low → high), diverging (negative ↔ positive), and qualitative (distinct categories).

```python
# View a palette
sns.palplot(sns.color_palette('Set2'))
plt.show()

# Sequential — for ordered data (light to dark)
sns.color_palette('Blues')
sns.color_palette('viridis')
sns.color_palette('rocket')

# Diverging — for data with a meaningful center
sns.color_palette('coolwarm')
sns.color_palette('RdBu')
sns.color_palette('vlag')

# Qualitative — for distinct categories
sns.color_palette('Set1')
sns.color_palette('Set2')
sns.color_palette('tab10')

# Custom number of colors
sns.color_palette('husl', 8)       # 8 evenly spaced colors

# Use as context manager
with sns.color_palette('pastel'):
    sns.countplot(data=tips, x='day')
    plt.show()
```

| Palette Type | Use When | Examples |
|-------------|----------|---------|
| Sequential | Data goes from low to high | `Blues`, `viridis`, `rocket` |
| Diverging | Data has a meaningful center | `coolwarm`, `RdBu`, `vlag` |
| Qualitative | Categories with no order | `Set1`, `Set2`, `tab10` |

---

## 13. Customization & Integration with Matplotlib

Since Seaborn is built on Matplotlib, you can mix both freely. Use Seaborn for the plot and Matplotlib for fine-tuning.

```python
# Seaborn plot + Matplotlib customization
fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=tips, x='day', y='total_bill', ax=ax, palette='Set3')

# Matplotlib customization
ax.set_title('Daily Bill Distribution', fontsize=16, fontweight='bold')
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Total Bill ($)', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('plot.png', dpi=300)
plt.show()
```

### Context for Scaling

`set_context()` adjusts element sizes for different output targets.

```python
sns.set_context('paper')        # small, for papers
sns.set_context('notebook')     # medium, default
sns.set_context('talk')         # large, for presentations
sns.set_context('poster')       # extra large, for posters
```

---

## 14. Complete Example — EDA Workflow

A typical exploratory data analysis using Seaborn follows this pattern.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load and inspect data
tips = sns.load_dataset('tips')
print(tips.info())
print(tips.describe())

# Set theme
sns.set_theme(style='whitegrid', palette='Set2', font_scale=1.1)

# 1. Overview: pair plot
sns.pairplot(tips, hue='time', diag_kind='kde')
plt.suptitle('Tips Dataset Overview', y=1.02)
plt.show()

# 2. Distribution of target variable
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(data=tips, x='tip', kde=True, ax=axes[0])
axes[0].set_title('Tip Distribution')
sns.boxplot(data=tips, x='day', y='tip', ax=axes[1])
axes[1].set_title('Tips by Day')
plt.tight_layout()
plt.show()

# 3. Relationships
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='smoker', ax=axes[0])
axes[0].set_title('Tip vs Bill')
sns.barplot(data=tips, x='day', y='tip', hue='sex', ax=axes[1])
axes[1].set_title('Average Tip by Day & Gender')
plt.tight_layout()
plt.show()

# 4. Correlation heatmap
corr = tips[['total_bill', 'tip', 'size']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()
```

---

## Quick Reference Cheat Sheet

| Plot Type | Function | Best For |
|-----------|----------|----------|
| **Scatter** | `scatterplot` / `relplot` | Relationship between 2 numeric vars |
| **Line** | `lineplot` / `relplot` | Trends over time |
| **Histogram** | `histplot` / `displot` | Distribution of one variable |
| **KDE** | `kdeplot` / `displot` | Smooth density estimate |
| **Box** | `boxplot` / `catplot` | Compare distributions across groups |
| **Violin** | `violinplot` / `catplot` | Distribution shape comparison |
| **Strip** | `stripplot` / `catplot` | Individual points by category |
| **Swarm** | `swarmplot` / `catplot` | Non-overlapping points by category |
| **Bar** | `barplot` / `catplot` | Mean + CI by category |
| **Count** | `countplot` / `catplot` | Count per category |
| **Heatmap** | `heatmap` | Correlation, confusion matrix |
| **Cluster** | `clustermap` | Heatmap with hierarchical grouping |
| **Regression** | `regplot` / `lmplot` | Scatter + fitted line |
| **Pair** | `pairplot` | All pairwise relationships |
| **Joint** | `jointplot` | 2 variables + marginal distributions |
| **Facet** | `FacetGrid` | Multi-panel by categories |

---

## Key Parameters Across All Plots

| Parameter | Effect | Example |
|-----------|--------|---------|
| `hue` | Color by category | `hue='species'` |
| `size` | Size by value | `size='weight'` |
| `style` | Marker/line style by category | `style='sex'` |
| `col` | Separate columns (figure-level) | `col='time'` |
| `row` | Separate rows (figure-level) | `row='smoker'` |
| `palette` | Color scheme | `palette='Set2'` |
| `alpha` | Transparency (0–1) | `alpha=0.7` |
| `order` | Custom category order | `order=['Mon','Tue']` |

---

## Performance Tips

1. **Start with `pairplot`** for a quick overview of any new dataset
2. **Use `hue` before creating subplots** — it's often enough to show group differences
3. **Figure-level functions** (`catplot`, `relplot`, `displot`) handle faceting automatically — use them for multi-panel displays
4. **Set the theme once** at the top of your script with `sns.set_theme()` for consistent styling
5. **Use `despine()`** to remove unnecessary borders for a cleaner look
6. **Switch to axes-level** functions when you need precise control over Matplotlib axes
