# Matplotlib Quick Reference Guide

> **With Explanations, Code Examples & Quick Reference**
> Python Data Visualization Fundamentals

---

## 1. Installation & Import

Matplotlib is Python's foundational plotting library. It can create virtually any type of static, animated, or interactive visualization. While newer libraries like Seaborn and Plotly are built on top of it, understanding Matplotlib is essential because it gives you full control over every element of a figure. Most data science libraries (Pandas, Seaborn) use Matplotlib as their rendering backend.

```bash
pip install matplotlib
```

```python
import matplotlib.pyplot as plt
import numpy as np
```

> ⚡ Always import `pyplot` as `plt` — this is the universal convention used everywhere.

---

## 2. Basic Plotting

The simplest way to create a plot is with `plt.plot()`. Matplotlib works in two modes: the **quick pyplot mode** (procedural, good for simple plots) and the **object-oriented mode** (more control, recommended for complex plots). Both produce the same output.

### Quick pyplot Mode

```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.title('My First Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.show()
```

### Object-Oriented Mode (Recommended)

```python
fig, ax = plt.subplots()        # create figure and axes
ax.plot(x, y)
ax.set_title('My First Plot')
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
plt.show()
```

### Saving Figures

```python
fig.savefig('plot.png', dpi=300, bbox_inches='tight')
fig.savefig('plot.pdf')       # vector format
fig.savefig('plot.svg')       # scalable vector
```

> ⚡ Use `bbox_inches='tight'` to prevent labels from being cut off when saving.

---

## 3. Line Plots

Line plots show trends over time or continuous data. They're the default plot type in Matplotlib. You can customize color, width, style, and markers to distinguish multiple lines.

```python
x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(10, 6))

# Multiple lines with different styles
ax.plot(x, np.sin(x), label='sin(x)', color='blue', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', color='red', linestyle='--')
ax.plot(x, np.sin(x) * 0.5, label='0.5·sin(x)', color='green', linestyle=':', marker='o', markevery=10)

ax.set_title('Trigonometric Functions', fontsize=16)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.show()
```

### Line Styles & Markers

| Style Code | Meaning | Marker Code | Shape |
|------------|---------|-------------|-------|
| `'-'` | Solid line | `'o'` | Circle |
| `'--'` | Dashed | `'s'` | Square |
| `'-.'` | Dash-dot | `'^'` | Triangle |
| `':'` | Dotted | `'*'` | Star |
| `''` | No line | `'D'` | Diamond |

---

## 4. Scatter Plots

Scatter plots show the relationship between two numerical variables. Each point represents one data sample. They're ideal for spotting correlations, clusters, and outliers. You can encode a third variable using color or size.

```python
np.random.seed(42)
x = np.random.rand(100)
y = x + np.random.normal(0, 0.2, 100)
colors = np.random.rand(100)
sizes = np.random.rand(100) * 200

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(x, y, c=colors, s=sizes, cmap='viridis', alpha=0.7, edgecolors='black')

ax.set_title('Scatter Plot with Color & Size', fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.colorbar(scatter, label='Color Value')
plt.show()
```

---

## 5. Bar Charts

Bar charts compare quantities across categories. Vertical bars (`bar`) are the default; horizontal bars (`barh`) are useful when category names are long. Grouped and stacked bars show sub-categories.

```python
categories = ['Python', 'Java', 'JavaScript', 'C++', 'Go']
values = [85, 72, 90, 60, 55]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, values, color=['#3498db','#e74c3c','#f39c12','#2ecc71','#9b59b6'])

# Add value labels on bars
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(val), ha='center', fontsize=11)

ax.set_title('Programming Language Popularity', fontsize=14)
ax.set_ylabel('Score')
plt.show()
```

### Grouped Bars

```python
x = np.arange(4)
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, [20, 35, 30, 25], width, label='2023')
ax.bar(x + width/2, [25, 40, 28, 32], width, label='2024')
ax.set_xticks(x)
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax.legend()
plt.show()
```

### Horizontal Bars

```python
ax.barh(categories, values, color='steelblue')
```

---

## 6. Histograms

Histograms show the distribution (shape) of a single numerical variable. They group values into bins and count how many fall into each bin. They answer "how is my data spread out?" Use them to spot normal distributions, skewness, and outliers.

```python
data = np.random.randn(1000)    # 1000 values from normal distribution

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)

ax.set_title('Distribution of Values', fontsize=14)
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
ax.legend()
plt.show()
```

### Overlapping Histograms

```python
fig, ax = plt.subplots()
ax.hist(data1, bins=30, alpha=0.5, label='Group A')
ax.hist(data2, bins=30, alpha=0.5, label='Group B')
ax.legend()
plt.show()
```

---

## 7. Pie Charts

Pie charts show proportions of a whole. While often criticized in data science (bar charts are usually clearer), they're still useful for simple breakdowns with few categories (3–5 max). Use `autopct` to show percentages.

```python
labels = ['Python', 'Java', 'JS', 'Other']
sizes = [40, 25, 20, 15]
explode = (0.05, 0, 0, 0)     # slightly separate first slice

fig, ax = plt.subplots(figsize=(7, 7))
ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
       shadow=True, startangle=90, colors=['#3498db','#e74c3c','#f39c12','#95a5a6'])
ax.set_title('Language Usage', fontsize=14)
plt.show()
```

---

## 8. Box Plots

Box plots (box-and-whisker) show the distribution of data using five summary numbers: minimum, Q1 (25th percentile), median (50th), Q3 (75th), and maximum. Outliers appear as individual points. They're excellent for comparing distributions across groups.

```python
data = [np.random.normal(0, 1, 100),
        np.random.normal(2, 1.5, 100),
        np.random.normal(-1, 0.5, 100)]

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(data, labels=['Group A', 'Group B', 'Group C'], patch_artist=True)

colors = ['#3498db', '#e74c3c', '#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Distribution Comparison', fontsize=14)
ax.set_ylabel('Value')
plt.show()
```

---

## 9. Subplots

Subplots let you place multiple plots in a grid within one figure. This is essential for comparing related visualizations side by side. Use `plt.subplots(rows, cols)` to create the grid, then plot on each axes object.

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: line
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Line Plot')

# Top-right: scatter
axes[0, 1].scatter(np.random.rand(50), np.random.rand(50))
axes[0, 1].set_title('Scatter Plot')

# Bottom-left: bar
axes[1, 0].bar(['A', 'B', 'C'], [10, 20, 15])
axes[1, 0].set_title('Bar Chart')

# Bottom-right: histogram
axes[1, 1].hist(np.random.randn(500), bins=20)
axes[1, 1].set_title('Histogram')

fig.suptitle('Four Plot Types', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Unequal Subplot Sizes

```python
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)    # 1 row, 2 cols, position 1 (left)
ax2 = fig.add_subplot(122)    # position 2 (right)

# Or with GridSpec for complex layouts
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])      # top row, full width
ax2 = fig.add_subplot(gs[1, 0:2])    # bottom row, left 2/3
ax3 = fig.add_subplot(gs[1, 2])      # bottom row, right 1/3
```

---

## 10. Customization & Styling

Matplotlib gives you complete control over every visual element. Colors, fonts, sizes, grids, legends, and annotations can all be customized. Learning these options lets you create publication-quality figures.

### Colors

```python
# Named colors
ax.plot(x, y, color='steelblue')
ax.plot(x, y, color='tomato')

# Hex colors
ax.plot(x, y, color='#3498db')

# RGB tuple (0-1 range)
ax.plot(x, y, color=(0.2, 0.4, 0.8))

# Built-in colormaps
plt.scatter(x, y, c=values, cmap='viridis')     # sequential
plt.scatter(x, y, c=values, cmap='coolwarm')     # diverging
plt.scatter(x, y, c=values, cmap='Set2')         # categorical
```

### Fonts & Text

```python
ax.set_title('Title', fontsize=16, fontweight='bold', fontfamily='serif')
ax.set_xlabel('X Label', fontsize=12, fontstyle='italic')

# Annotations
ax.annotate('Peak', xy=(3, 9), xytext=(4, 7),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=12, color='red')

# Text on plot
ax.text(0.5, 0.5, 'Center Text', transform=ax.transAxes,
        ha='center', fontsize=14)
```

### Grid, Spines & Ticks

```python
ax.grid(True, alpha=0.3, linestyle='--')

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Custom ticks
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(['Jan','Mar','May','Jul','Sep','Nov'], rotation=45)
ax.tick_params(axis='both', labelsize=10)

# Limits
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
```

### Built-in Styles

```python
print(plt.style.available)     # see all styles

plt.style.use('seaborn-v0_8')      # clean and modern
plt.style.use('ggplot')             # R-style
plt.style.use('dark_background')    # dark theme
plt.style.use('bmh')                # Bayesian Methods style
plt.style.use('default')            # reset to default
```

---

## 11. Heatmaps

Heatmaps display a matrix of values as colors. They're perfect for correlation matrices, confusion matrices, and any 2D data. Each cell's color represents its value.

```python
data = np.random.rand(5, 5)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data, cmap='YlOrRd')

# Add text annotations in each cell
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=10)

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(['A', 'B', 'C', 'D', 'E'])
ax.set_yticklabels(['V', 'W', 'X', 'Y', 'Z'])
fig.colorbar(im)
ax.set_title('Heatmap', fontsize=14)
plt.show()
```

---

## 12. Error Bars & Fill Between

Error bars show uncertainty or variability in measurements. Fill between shades the area between two curves, useful for showing confidence intervals or ranges.

```python
# Error bars
x = np.arange(5)
y = [20, 35, 30, 25, 40]
errors = [2, 3, 1.5, 2.5, 3.5]

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=errors, fmt='o-', capsize=5, color='steelblue')
ax.set_title('Measurements with Error Bars')
plt.show()
```

```python
# Fill between
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y, color='blue')
ax.fill_between(x, y - 0.3, y + 0.3, alpha=0.2, color='blue', label='±0.3 range')
ax.legend()
plt.show()
```

---

## 13. 3D Plots

Matplotlib can create 3D surface, scatter, and wireframe plots. You need to import `Axes3D` and create a 3D subplot.

```python
from mpl_toolkits.mplot3d import Axes3D

# 3D Scatter
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

ax.scatter(x, y, z, c=z, cmap='viridis', s=50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')
plt.show()
```

```python
# 3D Surface
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax.set_title('3D Surface Plot')
plt.show()
```

---

## 14. Plotting with Pandas

Pandas has built-in plotting that uses Matplotlib as the backend. It's a convenient shortcut — instead of extracting data and calling Matplotlib directly, you can plot straight from a DataFrame.

```python
import pandas as pd

df = pd.DataFrame({
    'month': ['Jan','Feb','Mar','Apr','May'],
    'sales': [100, 120, 115, 130, 150],
    'costs': [80, 90, 85, 95, 100]
})

# Line plot
df.plot(x='month', y=['sales', 'costs'], figsize=(8, 5), title='Sales vs Costs')
plt.show()

# Bar chart
df.plot.bar(x='month', y='sales', color='steelblue')

# Histogram
df['sales'].plot.hist(bins=10)

# Scatter
df.plot.scatter(x='sales', y='costs', c='red', s=100)

# Box plot
df[['sales', 'costs']].plot.box()
```

---

## Quick Reference Cheat Sheet

| Plot Type | Function | Best For |
|-----------|----------|----------|
| **Line** | `ax.plot(x, y)` | Trends over time |
| **Scatter** | `ax.scatter(x, y)` | Relationships between 2 variables |
| **Bar** | `ax.bar(x, height)` | Comparing categories |
| **Horizontal Bar** | `ax.barh(y, width)` | Long category names |
| **Histogram** | `ax.hist(data, bins)` | Distribution of one variable |
| **Pie** | `ax.pie(sizes)` | Proportions of a whole |
| **Box** | `ax.boxplot(data)` | Distribution comparison |
| **Heatmap** | `ax.imshow(matrix)` | 2D data as colors |
| **Error Bar** | `ax.errorbar(x, y, yerr)` | Measurements with uncertainty |
| **Fill** | `ax.fill_between(x, y1, y2)` | Confidence intervals |
| **3D Scatter** | `ax.scatter(x, y, z)` | 3D data relationships |
| **3D Surface** | `ax.plot_surface(X, Y, Z)` | Mathematical surfaces |

---

## Performance Tips

1. **Use object-oriented mode** (`fig, ax = plt.subplots()`) for anything beyond quick exploration
2. **Always use `tight_layout()`** or `bbox_inches='tight'` to prevent overlapping labels
3. **Save as vector format** (PDF, SVG) for publications — they scale without losing quality
4. **Use `figsize`** to control figure dimensions from the start, not after plotting
5. **Set a style** with `plt.style.use()` early to get a consistent, polished look with minimal effort
6. **Close figures** with `plt.close()` in loops to prevent memory leaks
