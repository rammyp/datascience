# Pandas Quick Reference Guide

> **With Explanations, Code Examples & Quick Reference**
> Python Data Science Fundamentals

---

## 1. Installation & Import

Pandas is the most popular Python library for data manipulation and analysis. It provides two primary data structures — Series (1D) and DataFrame (2D) — that make it easy to clean, transform, analyze, and visualize structured data. Pandas is built on top of NumPy and is the backbone of nearly every data science workflow in Python.

```bash
pip install pandas
```

```python
import pandas as pd    # 'pd' is the universal convention
import numpy as np     # often used alongside Pandas
```

> ⚡ Always import as `pd` — this is a universal convention used in all documentation, tutorials, and production code.

---

## 2. Core Data Structures

Pandas has two main data structures. A **Series** is a one-dimensional labeled array, like a single column of a spreadsheet. A **DataFrame** is a two-dimensional labeled table, like an entire spreadsheet. Both can hold mixed data types and use labels (index) to access data instead of just positions.

### Series

A Series is essentially a NumPy array with labels. It has an index (row labels) and values. If you don't provide an index, Pandas creates one automatically (0, 1, 2, ...).

```python
# From a list
s = pd.Series([10, 20, 30, 40])
# 0    10
# 1    20
# 2    30
# 3    40

# With custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
# a    10
# b    20
# c    30

# From a dictionary
s = pd.Series({'apples': 5, 'bananas': 3, 'oranges': 8})

# Access values
s['a']        # 10        — by label
s.iloc[0]     # 10        — by position
s.values      # array([10, 20, 30])  — underlying NumPy array
```

### DataFrame

A DataFrame is like a dictionary of Series that share the same index. Each column can have a different data type. It is the most commonly used Pandas object and what you'll work with 90% of the time.

```python
# From a dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'score': [85.5, 92.0, 78.3, 95.1]
})
#       name  age  score
# 0    Alice   25   85.5
# 1      Bob   30   92.0
# 2  Charlie   35   78.3
# 3    David   28   95.1

# From a list of dictionaries
df = pd.DataFrame([
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30}
])

# From a NumPy array
df = pd.DataFrame(
    np.random.rand(3, 4),
    columns=['A', 'B', 'C', 'D']
)
```

---

## 3. Reading & Writing Data

Pandas can read data from many file formats. `read_csv()` is by far the most common. Each read function has a corresponding write function. The `index=False` parameter prevents Pandas from writing the row numbers to the file, which is usually what you want.

```python
# CSV (most common)
df = pd.read_csv('data.csv')
df.to_csv('output.csv', index=False)

# Excel
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df.to_excel('output.xlsx', index=False)

# JSON
df = pd.read_json('data.json')
df.to_json('output.json')

# From clipboard (useful for quick copy-paste)
df = pd.read_clipboard()

# SQL database
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table_name', conn)
df.to_sql('table_name', conn, if_exists='replace', index=False)

# Parquet (fast, compressed — popular in big data)
df = pd.read_parquet('data.parquet')
df.to_parquet('output.parquet', index=False)
```

> ⚡ Use `nrows=5` in `read_csv()` to preview large files without loading everything: `pd.read_csv('big.csv', nrows=5)`

---

## 4. Exploring Data

Before doing any analysis, you need to understand your data's structure, types, and quality. These functions give you a quick overview without modifying anything. Always start with `head()`, `info()`, and `describe()` when working with a new dataset.

```python
df.head()          # first 5 rows
df.head(10)        # first 10 rows
df.tail()          # last 5 rows
df.shape           # (rows, columns) as tuple
df.info()          # column names, types, non-null counts
df.describe()      # statistics for numeric columns
df.dtypes          # data type of each column
df.columns         # list of column names
df.index           # row index
df.nunique()       # count of unique values per column
df.sample(5)       # 5 random rows
df.memory_usage()  # memory per column in bytes
```

### Value Counts

`value_counts()` counts occurrences of each unique value. It is one of the most useful functions for understanding categorical data and finding data quality issues.

```python
df['city'].value_counts()
# New York    45
# London      38
# Tokyo       22

df['city'].value_counts(normalize=True)   # as percentages
df['city'].value_counts(dropna=False)     # include NaN counts
```

---

## 5. Selecting Data

Selecting data is the most fundamental Pandas operation. There are several ways to do it, and understanding the difference between `loc` (label-based) and `iloc` (position-based) is essential. Using the wrong one is a common source of bugs.

### Select Columns

```python
# Single column (returns Series)
df['name']
df.name              # dot notation (only for simple names)

# Multiple columns (returns DataFrame)
df[['name', 'age']]

# Select by data type
df.select_dtypes(include='number')     # numeric columns only
df.select_dtypes(include='object')     # string columns only
```

### Select Rows with loc & iloc

`loc` uses labels (names); `iloc` uses integer positions. This is the single most important distinction to learn in Pandas.

```python
# loc — label-based (inclusive on both ends)
df.loc[0]                  # row with label 0
df.loc[0:3]                # rows labeled 0 through 3 (inclusive!)
df.loc[0, 'name']          # specific cell
df.loc[:, 'name':'score']  # all rows, columns 'name' to 'score'

# iloc — position-based (exclusive on end, like Python)
df.iloc[0]                 # first row
df.iloc[0:3]               # first 3 rows (exclusive end)
df.iloc[0, 1]              # row 0, column 1
df.iloc[:, 0:2]            # all rows, first 2 columns
```

### Boolean Selection (Filtering)

Filtering is how you extract rows that meet a condition. You can combine conditions with `&` (and), `|` (or), and `~` (not). Always wrap each condition in parentheses.

```python
# Single condition
df[df['age'] > 30]

# Multiple conditions
df[(df['age'] > 25) & (df['score'] > 80)]

# isin — match against a list
df[df['city'].isin(['New York', 'London'])]

# String contains
df[df['name'].str.contains('Ali')]

# Between
df[df['age'].between(25, 35)]

# Query method (cleaner syntax)
df.query('age > 25 and score > 80')
```

---

## 6. Adding & Modifying Data

DataFrames are mutable — you can add columns, modify values, and delete columns at any time. Creating new columns from existing ones is one of the most common operations in data preparation.

### Add / Modify Columns

```python
# New column from calculation
df['bonus'] = df['score'] * 10

# New column from condition
df['passed'] = df['score'] > 80

# Multiple conditions — np.where
df['grade'] = np.where(df['score'] >= 90, 'A', 'B')

# Multiple conditions — np.select
conditions = [
    df['score'] >= 90,
    df['score'] >= 80,
    df['score'] >= 70
]
choices = ['A', 'B', 'C']
df['grade'] = np.select(conditions, choices, default='F')

# Apply a function to a column
df['name_upper'] = df['name'].apply(str.upper)

# Apply custom function
df['age_group'] = df['age'].apply(lambda x: 'Young' if x < 30 else 'Senior')

# Rename columns
df.rename(columns={'name': 'full_name', 'age': 'years'}, inplace=True)
```

### Delete Columns & Rows

```python
# Drop columns
df.drop(columns=['bonus', 'grade'], inplace=True)
df.drop('bonus', axis=1, inplace=True)    # axis=1 means column

# Drop rows by index
df.drop(index=[0, 1], inplace=True)
df.drop(df[df['score'] < 70].index, inplace=True)  # drop by condition
```

> ⚡ Use `inplace=True` to modify the DataFrame directly. Without it, Pandas returns a new DataFrame and the original stays unchanged.

---

## 7. Handling Missing Data

Real-world data almost always has missing values. Pandas represents them as `NaN` (Not a Number). Knowing how to detect, remove, and fill missing values is critical for data cleaning. The choice between dropping and filling depends on how much data is missing and the context.

### Detect Missing Values

```python
df.isnull()              # True/False for each cell
df.isnull().sum()        # count NaN per column
df.isnull().sum().sum()  # total NaN in entire DataFrame
df.notnull()             # opposite of isnull
df['age'].isna()         # isna() is alias for isnull()
```

### Drop Missing Values

```python
df.dropna()                    # drop rows with ANY NaN
df.dropna(how='all')           # drop rows where ALL values are NaN
df.dropna(subset=['age'])      # drop only if 'age' is NaN
df.dropna(thresh=3)            # keep rows with at least 3 non-NaN values
```

### Fill Missing Values

```python
df.fillna(0)                         # fill all NaN with 0
df['age'].fillna(df['age'].mean())   # fill with column mean
df['age'].fillna(df['age'].median()) # fill with median
df.fillna(method='ffill')            # forward fill (copy from above)
df.fillna(method='bfill')            # backward fill (copy from below)
df.interpolate()                     # linear interpolation
```

### Replace Values

```python
df.replace('Unknown', np.nan)                    # single value
df.replace({'Male': 'M', 'Female': 'F'})         # multiple values
df['grade'].replace({'A': 4, 'B': 3, 'C': 2})   # column-specific
```

---

## 8. Data Types & Conversion

Each column in a DataFrame has a data type (dtype). Incorrect dtypes cause errors in calculations and waste memory. For example, a "price" column stored as a string can't be summed. Converting dtypes is a common data cleaning step.

```python
# Check types
df.dtypes

# Convert types
df['age'] = df['age'].astype(int)
df['score'] = df['score'].astype(float)
df['name'] = df['name'].astype(str)

# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Convert to numeric (handles errors)
df['price'] = pd.to_numeric(df['price'], errors='coerce')  # invalid → NaN

# Convert to category (saves memory for repeated strings)
df['city'] = df['city'].astype('category')

# Check memory savings
df.memory_usage(deep=True)
```

| dtype | Meaning | Example |
|-------|---------|---------|
| `int64` | Integer numbers | 1, 42, -7 |
| `float64` | Decimal numbers | 3.14, -0.5 |
| `object` | Strings (text) | 'hello', 'NY' |
| `bool` | True / False | True, False |
| `datetime64` | Dates and times | 2024-01-15 |
| `category` | Fixed set of values | 'M', 'F', 'Other' |

---

## 9. String Operations

Pandas provides vectorized string methods through the `.str` accessor. These work on entire columns at once, avoiding slow Python loops. Every standard Python string method has a Pandas equivalent.

```python
s = pd.Series(['  Alice Smith  ', 'BOB JONES', 'charlie brown'])

s.str.lower()            # ['  alice smith  ', 'bob jones', 'charlie brown']
s.str.upper()            # ['  ALICE SMITH  ', 'BOB JONES', 'CHARLIE BROWN']
s.str.title()            # ['  Alice Smith  ', 'Bob Jones', 'Charlie Brown']
s.str.strip()            # ['Alice Smith', 'BOB JONES', 'charlie brown']
s.str.len()              # [15, 9, 13]
s.str.contains('alice', case=False)   # [True, False, False]
s.str.startswith('A')    # [False, False, False]  (leading space!)
s.str.replace('Smith', 'Lee')
s.str.split(' ')         # split into lists
s.str[0:5]               # slice first 5 characters
s.str.extract(r'(\w+)')  # regex extract first word
```

### Common Pattern: Clean Then Process

```python
# Typical string cleaning pipeline
df['name'] = (df['name']
    .str.strip()          # remove whitespace
    .str.lower()          # standardize case
    .str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation
)
```

---

## 10. Sorting

Sorting arranges rows by one or more columns. `sort_values()` sorts by column values, `sort_index()` sorts by the row index. Sorting is essential for finding top/bottom values, creating ranked lists, and preparing data for display.

```python
# Sort by single column
df.sort_values('score')                          # ascending (default)
df.sort_values('score', ascending=False)         # descending

# Sort by multiple columns
df.sort_values(['grade', 'score'], ascending=[True, False])

# Sort by index
df.sort_index()
df.sort_index(ascending=False)

# Get top/bottom N
df.nlargest(5, 'score')     # top 5 by score
df.nsmallest(3, 'age')      # bottom 3 by age

# Rank values
df['rank'] = df['score'].rank(ascending=False)
```

---

## 11. Grouping & Aggregation

`groupby()` is one of Pandas' most powerful features. It splits data into groups, applies a function to each group, and combines the results. Think of it as the equivalent of SQL's GROUP BY. The pattern is always: **split → apply → combine**.

### Basic GroupBy

```python
# Average score by city
df.groupby('city')['score'].mean()

# Multiple aggregations
df.groupby('city')['score'].agg(['mean', 'min', 'max', 'count'])

# Group by multiple columns
df.groupby(['city', 'grade'])['score'].mean()

# Named aggregations (clean output)
df.groupby('city').agg(
    avg_score=('score', 'mean'),
    total_students=('name', 'count'),
    max_age=('age', 'max')
)
```

### Common Aggregation Functions

```python
df.groupby('city')['score'].sum()       # total
df.groupby('city')['score'].mean()      # average
df.groupby('city')['score'].median()    # middle value
df.groupby('city')['score'].std()       # standard deviation
df.groupby('city')['score'].count()     # count (excludes NaN)
df.groupby('city')['score'].size()      # count (includes NaN)
df.groupby('city')['score'].first()     # first value
df.groupby('city')['score'].last()      # last value
df.groupby('city')['name'].nunique()    # unique count
```

### Transform & Filter

`transform()` returns a value for every row (same shape as input). `filter()` returns entire groups that meet a condition.

```python
# Add group mean as new column
df['city_avg'] = df.groupby('city')['score'].transform('mean')

# Deviation from group mean
df['diff'] = df['score'] - df.groupby('city')['score'].transform('mean')

# Keep only cities with more than 10 students
df.groupby('city').filter(lambda x: len(x) > 10)
```

---

## 12. Merging & Joining

Merging combines two DataFrames based on shared column values, similar to SQL JOINs. `merge()` is the primary function. The `how` parameter controls which rows to keep: `inner` keeps only matches, `outer` keeps everything, `left` keeps all rows from the left DataFrame.

```python
# Sample DataFrames
students = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'name': ['Alice', 'Bob', 'Charlie', 'David']
})
scores = pd.DataFrame({
    'id': [1, 2, 3, 5],
    'score': [85, 92, 78, 95]
})

# Inner join (only matching rows)
pd.merge(students, scores, on='id', how='inner')
#    id     name  score
# 0   1    Alice     85
# 1   2      Bob     92
# 2   3  Charlie     78

# Left join (all from left, match from right)
pd.merge(students, scores, on='id', how='left')
# David has NaN score (no match)

# Outer join (all from both)
pd.merge(students, scores, on='id', how='outer')
# David has NaN score, id=5 has NaN name

# Different column names
pd.merge(students, scores, left_on='student_id', right_on='id')
```

### Concatenation

`concat()` stacks DataFrames on top of each other (vertically) or side by side (horizontally). Unlike merge, it doesn't match on values — it just glues DataFrames together.

```python
# Vertical (stack rows)
pd.concat([df1, df2], ignore_index=True)

# Horizontal (side by side)
pd.concat([df1, df2], axis=1)
```

| Join Type | Keeps |
|-----------|-------|
| `inner` | Only rows with matches in BOTH tables |
| `left` | All rows from left table + matches from right |
| `right` | All rows from right table + matches from left |
| `outer` | All rows from BOTH tables |

---

## 13. Pivot Tables & Reshaping

Pivot tables summarize data by grouping and aggregating, similar to Excel pivot tables. Melting does the opposite — it converts wide-format data into long-format. These are essential for preparing data for visualization and analysis.

### Pivot Table

```python
# Average score by city and grade
pd.pivot_table(
    df,
    values='score',
    index='city',
    columns='grade',
    aggfunc='mean'
)

# Multiple aggregations
pd.pivot_table(
    df,
    values='score',
    index='city',
    aggfunc=['mean', 'count'],
    margins=True       # adds row/column totals
)
```

### Melt (Wide → Long)

```python
# Wide format
#    name  math  science  english
# 0  Alice   85       90       88

df_long = pd.melt(
    df,
    id_vars='name',
    value_vars=['math', 'science', 'english'],
    var_name='subject',
    value_name='score'
)
# Long format
#    name  subject  score
# 0  Alice     math     85
# 1  Alice  science     90
# 2  Alice  english     88
```

### Pivot (Long → Wide)

```python
df_wide = df_long.pivot(index='name', columns='subject', values='score')
```

### Cross Tabulation

```python
pd.crosstab(df['city'], df['grade'])           # counts
pd.crosstab(df['city'], df['grade'], normalize=True)  # percentages
```

---

## 14. Date & Time

Pandas has powerful datetime support. Converting strings to datetime objects unlocks time-based indexing, resampling, and calculations. The `dt` accessor gives you access to individual date components (year, month, day, etc.).

### Convert to Datetime

```python
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # invalid → NaT
```

### Extract Components

```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()      # 'Monday', etc.
df['quarter'] = df['date'].dt.quarter
df['hour'] = df['date'].dt.hour
df['is_weekend'] = df['date'].dt.dayofweek >= 5
```

### Date Arithmetic

```python
# Difference between dates
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days

# Add/subtract time
df['next_week'] = df['date'] + pd.Timedelta(days=7)
df['next_month'] = df['date'] + pd.DateOffset(months=1)
```

### Date Ranges & Resampling

```python
# Create date range
dates = pd.date_range('2024-01-01', periods=12, freq='M')

# Set date as index, then resample
df.set_index('date', inplace=True)
df.resample('M').mean()      # monthly average
df.resample('Q').sum()       # quarterly total
df.resample('Y').count()     # yearly count
```

---

## 15. Apply, Map & Applymap

These functions let you apply custom logic to DataFrames. `apply()` works on rows or columns, `map()` works on a single Series element by element, and `applymap()` works on every cell in a DataFrame.

```python
# apply — on columns or rows
df['score'].apply(lambda x: 'Pass' if x >= 70 else 'Fail')

df.apply(lambda row: row['score'] / row['age'], axis=1)  # row-wise

# map — on a Series (element-wise)
df['grade'].map({'A': 4, 'B': 3, 'C': 2, 'F': 0})

# map with function
df['name'].map(str.upper)

# applymap — on every cell in DataFrame (Pandas < 2.0)
# In Pandas 2.0+, use df.map() instead
df[['score', 'age']].applymap(lambda x: round(x, 1))
```

### When to Use Which

| Function | Works On | Use Case |
|----------|----------|----------|
| `apply()` | Series or DataFrame | Complex row/column logic |
| `map()` | Series only | Replace values, simple transforms |
| `applymap()` | Entire DataFrame | Format every cell |

> ⚡ Prefer built-in vectorized operations over `apply()` whenever possible — they're 10–100× faster.

---

## 16. Duplicates

Duplicate rows can skew analysis and inflate counts. Pandas makes it easy to find and remove them. `duplicated()` marks duplicate rows as True, and `drop_duplicates()` removes them.

```python
# Find duplicates
df.duplicated()                              # True for duplicate rows
df.duplicated(subset=['name'])               # based on specific columns
df.duplicated(keep='first')                  # mark all except first
df.duplicated(keep='last')                   # mark all except last
df.duplicated(keep=False)                    # mark ALL duplicates
df[df.duplicated()]                          # show duplicate rows

# Count duplicates
df.duplicated().sum()

# Remove duplicates
df.drop_duplicates()                         # keep first occurrence
df.drop_duplicates(subset=['name'])          # based on column
df.drop_duplicates(keep='last')              # keep last occurrence
```

---

## 17. Multi-Index

A MultiIndex (hierarchical index) allows you to have multiple levels of row or column labels. This is useful for representing higher-dimensional data in a 2D table. GroupBy operations often produce MultiIndex results.

```python
# Create MultiIndex DataFrame
df.set_index(['city', 'name'], inplace=True)

# Access with MultiIndex
df.loc['New York']                      # all rows for New York
df.loc[('New York', 'Alice')]           # specific row

# Reset to regular index
df.reset_index(inplace=True)

# Swap levels
df.swaplevel()

# Sort MultiIndex
df.sort_index(level=0)
```

---

## 18. Window Functions

Window functions compute values over a sliding window of rows. `rolling()` creates a moving window, `expanding()` creates a growing window from the start, and `shift()` moves data up or down. These are essential for time series analysis, calculating running averages, and detecting trends.

```python
# Rolling (moving) calculations
df['score'].rolling(3).mean()       # 3-period moving average
df['score'].rolling(5).sum()        # 5-period moving sum
df['score'].rolling(3).std()        # 3-period moving std

# Expanding (cumulative) calculations
df['score'].expanding().mean()      # running average
df['score'].expanding().sum()       # running total
df['score'].cumsum()                # shortcut for expanding sum
df['score'].cummax()                # running maximum

# Shift — move data up or down
df['prev_score'] = df['score'].shift(1)     # previous row's value
df['next_score'] = df['score'].shift(-1)    # next row's value
df['score_change'] = df['score'] - df['score'].shift(1)  # period change

# Percentage change
df['pct_change'] = df['score'].pct_change()   # % change from previous
```

---

## 19. Working with Categories

Categorical data represents values from a fixed set (like grades, colors, sizes). Converting to `category` dtype saves memory and enables ordering. This is especially useful for columns with many repeated string values.

```python
# Convert to category
df['grade'] = df['grade'].astype('category')

# Ordered category
df['size'] = pd.Categorical(
    df['size'],
    categories=['S', 'M', 'L', 'XL'],
    ordered=True
)

# Now comparisons work
df[df['size'] > 'M']     # L and XL only

# Add/remove categories
df['grade'].cat.add_categories(['A+'])
df['grade'].cat.remove_unused_categories()

# Rename categories
df['grade'].cat.rename_categories({'A': 'Excellent', 'B': 'Good'})
```

---

## 20. Performance & Memory

Large datasets require attention to performance. Small changes in data types and techniques can make your code orders of magnitude faster. Here are the most impactful optimizations.

### Memory Optimization

```python
# Check memory usage
df.memory_usage(deep=True)

# Downcast numeric types
df['age'] = pd.to_numeric(df['age'], downcast='integer')      # int64 → int8
df['score'] = pd.to_numeric(df['score'], downcast='float')    # float64 → float32

# Use category for repeated strings
df['city'] = df['city'].astype('category')   # often 90%+ savings

# Read only needed columns
df = pd.read_csv('big.csv', usecols=['name', 'score'])

# Read in chunks for very large files
for chunk in pd.read_csv('huge.csv', chunksize=10000):
    process(chunk)
```

### Speed Tips

```python
# ❌ Slow — Python loop
for i in range(len(df)):
    df.loc[i, 'bonus'] = df.loc[i, 'score'] * 10

# ✅ Fast — vectorized
df['bonus'] = df['score'] * 10

# ❌ Slow — apply with lambda
df['grade'] = df['score'].apply(lambda x: 'A' if x > 90 else 'B')

# ✅ Fast — np.where
df['grade'] = np.where(df['score'] > 90, 'A', 'B')

# ❌ Slow — growing DataFrame with append/concat in loop
# ✅ Fast — collect in list, concat once
rows = []
for item in data:
    rows.append(process(item))
df = pd.DataFrame(rows)
```

---

## 21. Common Patterns & Recipes

These are patterns you'll use again and again in real data work.

### Binning / Discretization

Convert continuous values into categories (bins).

```python
# Equal-width bins
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['Youth', 'Young Adult', 'Adult', 'Senior'])

# Equal-frequency bins (same number in each bin)
df['score_quartile'] = pd.qcut(df['score'], q=4, labels=['Q1','Q2','Q3','Q4'])
```

### One-Hot Encoding

Convert categorical columns into binary columns for machine learning.

```python
pd.get_dummies(df, columns=['city'], drop_first=True)
```

### Chaining Operations

Pandas supports method chaining for clean, readable pipelines.

```python
result = (df
    .query('age > 20')
    .assign(bonus=lambda x: x['score'] * 10)
    .sort_values('bonus', ascending=False)
    .head(10)
)
```

### Explode Lists

Turn list values in a column into separate rows.

```python
df = pd.DataFrame({'name': ['Alice', 'Bob'], 'hobbies': [['chess','art'], ['golf']]})
df.explode('hobbies')
#     name hobbies
# 0  Alice   chess
# 0  Alice     art
# 1    Bob    golf
```

---

## 22. Useful Shortcuts

```python
# Quick DataFrame from dictionary
pd.DataFrame({'a': [1], 'b': [2]})

# Set display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.2f}'.format)

# Reset all options
pd.reset_option('all')

# Pipe for custom functions in chains
def add_prefix(df, col, prefix):
    df[col] = prefix + df[col]
    return df

df.pipe(add_prefix, 'name', 'Mr. ')
```

---

## Quick Reference Cheat Sheet

| Category | Functions | Purpose |
|----------|-----------|---------|
| **Create** | `DataFrame(), Series(), read_csv(), read_excel()` | Build or load data |
| **Explore** | `head, tail, shape, info, describe, dtypes, nunique` | Understand your data |
| **Select** | `loc, iloc, [], query, isin, between` | Access rows and columns |
| **Filter** | `df[condition], query(), isin(), str.contains()` | Extract matching rows |
| **Modify** | `assign, rename, drop, replace, astype` | Change data and structure |
| **Missing** | `isnull, dropna, fillna, interpolate` | Handle NaN values |
| **Strings** | `str.lower, upper, strip, contains, replace, split` | Text cleaning and matching |
| **Sort** | `sort_values, sort_index, nlargest, nsmallest, rank` | Order data |
| **Group** | `groupby, agg, transform, filter, pivot_table` | Summarize by category |
| **Merge** | `merge, concat, join` | Combine DataFrames |
| **Reshape** | `pivot, melt, stack, unstack, crosstab, explode` | Change data layout |
| **Datetime** | `to_datetime, dt.year/month/day, resample, shift` | Time-based operations |
| **Apply** | `apply, map, applymap, pipe` | Custom transformations |
| **Window** | `rolling, expanding, shift, cumsum, pct_change` | Moving calculations |
| **Duplicates** | `duplicated, drop_duplicates` | Find and remove repeats |
| **I/O** | `to_csv, to_excel, to_json, to_parquet, to_sql` | Save data to files |

---

## Performance Tips

1. **Use vectorized operations** — avoid `for` loops and `iterrows()`; use built-in Pandas/NumPy functions (10–100× faster)
2. **Choose the right dtype** — use `category` for repeated strings, downcast numerics to smaller types
3. **Read only what you need** — use `usecols` and `nrows` in `read_csv()` for large files
4. **Build DataFrames at once** — collect rows in a list and create the DataFrame once, not row by row
5. **Use `query()` for readability** — cleaner syntax than bracket notation for complex filters
6. **Profile before optimizing** — use `df.memory_usage(deep=True)` and `%timeit` to find real bottlenecks
