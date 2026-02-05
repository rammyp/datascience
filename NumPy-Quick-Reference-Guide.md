# NumPy Quick Reference Guide

> **With Explanations, Code Examples & Quick Reference**
> Python Data Science Fundamentals

---

## 1. Installation & Import

NumPy (Numerical Python) is the core library for numerical computing in Python. It provides fast, memory-efficient multi-dimensional arrays and a large collection of mathematical functions. Nearly every data science library (Pandas, SciPy, Scikit-learn) is built on top of NumPy.

```bash
pip install numpy
```

```python
import numpy as np   # 'np' is the universal convention
```

> ⚡ Always import as `np` — this is a universal convention in the Python community and all documentation uses it.

---

## 2. Creating Arrays

A NumPy array (`ndarray`) is a grid of values, all of the same data type. Unlike Python lists, NumPy arrays are stored in contiguous memory, making them much faster for mathematical operations. Arrays can be 1D (vectors), 2D (matrices), or multi-dimensional (tensors).

### From Python Lists

The most basic way to create an array is by passing a Python list to `np.array()`. NumPy automatically determines the data type.

```python
a = np.array([1, 2, 3])             # 1D array
b = np.array([[1, 2], [3, 4]])      # 2D array (matrix)
c = np.array([1.0, 2, 3])           # float64 (auto-detected)
```

### Zeros, Ones & Empty

These functions create arrays pre-filled with specific values. `zeros()` fills with 0.0, `ones()` fills with 1.0, and `empty()` allocates memory without initializing it (values are garbage). These are useful for creating placeholder arrays before filling them with data.

```python
np.zeros(5)            # [0. 0. 0. 0. 0.]
np.zeros((2, 3))       # 2x3 matrix of zeros
np.ones((3, 3))        # 3x3 matrix of ones
np.full((2, 2), 7)     # 2x2 matrix filled with 7
np.empty((2, 2))       # uninitialized (random garbage values)
```

### Ranges & Sequences

`arange()` works like Python's `range()` but returns an array. `linspace()` creates evenly spaced numbers between two endpoints and is especially useful for plotting. The key difference: arange uses a step size, linspace uses a count of points.

```python
np.arange(0, 10, 2)    # [0 2 4 6 8]  — start, stop, step
np.linspace(0, 1, 5)   # [0. 0.25 0.5 0.75 1.] — start, stop, count
```

### Special Arrays

`eye()` creates an identity matrix (1s on the diagonal, 0s elsewhere), which is fundamental in linear algebra. Random functions generate arrays with random values from various distributions.

```python
np.eye(3)                      # 3x3 identity matrix
np.random.rand(3)              # 3 random floats in [0, 1)
np.random.randint(1, 10, 5)    # 5 random integers in [1, 10)
np.random.randn(3)             # 3 values from normal distribution
```

---

## 3. Array Attributes

Every NumPy array carries metadata that describes its structure. These attributes let you inspect the array's dimensions, size, and memory layout without modifying it. Understanding attributes is essential for debugging shape mismatches in calculations.

```python
a = np.array([[1, 2, 3], [4, 5, 6]])

a.shape      # (2, 3)  — 2 rows, 3 columns
a.ndim       # 2       — number of dimensions
a.size       # 6       — total number of elements
a.dtype      # int64   — data type of elements
a.itemsize   # 8       — bytes per element
a.nbytes     # 48      — total memory (size × itemsize)
```

| Attribute | Returns | Meaning |
|-----------|---------|---------|
| `.shape` | tuple | Dimensions as (rows, cols, ...) |
| `.ndim` | int | Number of dimensions (1D=1, 2D=2, etc.) |
| `.size` | int | Total element count across all dimensions |
| `.dtype` | dtype object | Data type (int64, float64, bool, etc.) |
| `.itemsize` | int | Bytes consumed by each individual element |
| `.nbytes` | int | Total memory usage in bytes (size × itemsize) |

---

## 4. Indexing & Slicing

Indexing retrieves a single element; slicing retrieves a sub-array. NumPy uses zero-based indexing like Python. The key syntax is `array[start:stop:step]`, where start is inclusive and stop is exclusive. Negative indices count from the end. For 2D arrays, use `array[row, col]`.

### 1D Indexing

```python
a = np.array([10, 20, 30, 40, 50])

a[0]       # 10     — first element
a[-1]      # 50     — last element
a[1:4]     # [20 30 40]  — index 1 to 3
a[::2]     # [10 30 50]  — every other element
a[::-1]    # [50 40 30 20 10]  — reversed
```

### 2D Indexing

For 2D arrays, the first index selects rows and the second selects columns. A colon `:` means "all elements along that axis." This is one of the most frequently used patterns in data manipulation.

```python
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

b[0, 0]      # 1         — row 0, col 0
b[1, :]      # [4 5 6]   — entire row 1
b[:, 1]      # [2 5 8]   — entire column 1
b[0:2, 1:]   # [[2 3] [5 6]]  — submatrix
```

---

## 5. Boolean Indexing (Filtering)

Boolean indexing lets you filter array elements using conditions. When you write `a > 3`, NumPy creates a boolean array (True/False for each element). Passing this boolean array as an index returns only the True elements. You can combine conditions with `&` (and), `|` (or), and `~` (not).

```python
a = np.array([1, 2, 3, 4, 5])

a[a > 3]               # [4 5]     — elements greater than 3
a[a % 2 == 0]          # [2 4]     — even numbers only
a[(a > 1) & (a < 5)]   # [2 3 4]   — combined conditions
a[~(a > 3)]            # [1 2 3]   — NOT greater than 3
```

> ⚡ Use `&` `|` `~` for combining conditions (not `and` `or` `not`). Always wrap each condition in parentheses.

---

## 6. Reshaping Arrays

Reshaping changes the dimensions of an array without changing its data. The total number of elements must stay the same (e.g., a 12-element array can become 3×4, 4×3, 2×2×3, etc.). Use `-1` to let NumPy auto-calculate one dimension. `flatten()` returns a copy; `ravel()` returns a view (more memory efficient).

```python
a = np.arange(12)     # [0 1 2 3 4 5 6 7 8 9 10 11]

a.reshape(3, 4)    # 3 rows × 4 cols matrix
a.reshape(2, -1)   # 2 rows, auto cols → (2, 6)
a.reshape(-1, 3)   # auto rows, 3 cols → (4, 3)

a.flatten()        # always returns a new copy as 1D
a.ravel()          # returns a 1D view (shares memory)
a.reshape(3,4).T   # transpose — swaps rows and columns
```

> ⚡ `reshape(-1)` is a common shortcut to flatten an array. `-1` tells NumPy to figure out the size automatically.

---

## 7. Stacking & Splitting

Stacking combines multiple arrays into one. `hstack` joins horizontally (side by side), `vstack` joins vertically (on top of each other). Splitting does the reverse: it breaks one array into multiple smaller arrays. These operations are essential when preparing data for machine learning.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Stacking
np.hstack((a, b))       # [1 2 3 4 5 6]       — horizontal
np.vstack((a, b))       # [[1 2 3] [4 5 6]]    — vertical
np.concatenate((a, b))  # [1 2 3 4 5 6]        — general

# Splitting
c = np.arange(9)
np.split(c, 3)   # [array([0,1,2]), array([3,4,5]), array([6,7,8])]
```

---

## 8. Math Operations

NumPy performs element-wise operations by default: when you add two arrays, it adds each corresponding pair of elements. This is much faster than looping through elements in Python because NumPy uses optimized C code internally. Standard operators (`+`, `-`, `*`, `/`, `**`) all work element-wise on arrays.

### Element-wise Operations

```python
a = np.array([1, 2, 3, 4])

a + 10      # [11 12 13 14]   — add scalar to each
a * 2       # [2 4 6 8]       — multiply each by 2
a ** 2      # [1 4 9 16]      — square each element
a + a       # [2 4 6 8]       — add arrays element-wise
a * a       # [1 4 9 16]      — multiply element-wise
```

### Mathematical Functions

NumPy provides vectorized versions of common math functions. These apply the function to every element at once, avoiding slow Python loops. The "universal functions" (ufuncs) include trigonometric, exponential, logarithmic, and rounding functions.

```python
np.sqrt(a)     # [1.  1.41  1.73  2.]  — square root
np.exp(a)      # [2.71  7.38  20.08  54.59]  — e^x
np.log(a)      # [0.  0.69  1.09  1.38]  — natural log
np.sin(a)      # sine of each element
np.abs(a)      # absolute value of each
np.round(a, 2) # round to 2 decimal places
```

---

## 9. Aggregate Functions

Aggregate (or reduction) functions collapse an array into a single summary value. `sum()` adds all elements, `mean()` computes the average, etc. For 2D arrays, the `axis` parameter controls the direction: `axis=0` collapses rows (operates down each column), `axis=1` collapses columns (operates across each row).

```python
a = np.array([1, 2, 3, 4, 5])

np.sum(a)       # 15        — total of all elements
np.mean(a)      # 3.0       — average
np.median(a)    # 3.0       — middle value
np.std(a)       # 1.414     — standard deviation
np.var(a)       # 2.0       — variance
np.min(a)       # 1         — smallest element
np.max(a)       # 5         — largest element
np.argmin(a)    # 0         — INDEX of smallest
np.argmax(a)    # 4         — INDEX of largest
```

### Axis Parameter (2D)

The axis parameter is one of the most confusing parts of NumPy. Think of it as: `axis=0` means "operate along rows" (result has one value per column), `axis=1` means "operate along columns" (result has one value per row). No axis means operate on the entire array.

```python
b = np.array([[1, 2],
              [3, 4]])

np.sum(b)          # 10       — sum everything
np.sum(b, axis=0)  # [4 6]    — sum down each column
np.sum(b, axis=1)  # [3 7]    — sum across each row
```

> ⚡ Memory trick: `axis=0` collapses rows (result shrinks vertically), `axis=1` collapses columns (result shrinks horizontally).

---

## 10. Sorting

`np.sort()` returns a new sorted array (does not modify the original). `argsort()` returns the indices that would sort the array, which is incredibly useful when you need to sort one array based on the values of another (e.g., sort student names by their scores).

```python
a = np.array([3, 1, 4, 1, 5])

np.sort(a)         # [1 1 3 4 5]     — sorted copy
np.argsort(a)      # [1 3 0 2 4]     — indices that sort
np.sort(a)[::-1]   # [5 4 3 1 1]     — descending

# Sort 2D
b = np.array([[3, 1], [4, 2]])
np.sort(b, axis=0)  # sort each column
np.sort(b, axis=1)  # sort each row
```

---

## 11. Searching & Conditions

`np.where()` is the NumPy equivalent of an if-else applied to every element. With one argument, it returns indices where the condition is True. With three arguments `(condition, x, y)`, it returns `x` where True, `y` where False. `np.any()` and `np.all()` check if any or all elements satisfy a condition.

```python
a = np.array([10, 20, 30, 40, 50])

np.where(a == 30)        # (array([2]),)      — index of 30
np.where(a > 25)         # (array([2,3,4]),)   — indices > 25
np.where(a > 25, a, 0)   # [0 0 30 40 50]     — if-else

# Find closest to a value
x = 33
idx = np.argmin(np.abs(a - x))  # index 2
a[idx]                          # 30

np.any(a > 40)   # True  — at least one > 40?
np.all(a > 5)    # True  — every element > 5?
```

---

## 12. Broadcasting

Broadcasting is NumPy's way of handling operations between arrays of different shapes. When you add a 1D array to a 2D array, NumPy automatically "stretches" the smaller array to match the larger one. This avoids writing explicit loops and makes code both shorter and faster. Broadcasting follows specific rules about compatible shapes.

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])     # shape (2, 3)
b = np.array([10, 20, 30])    # shape (3,)

a + b
# [[11 22 33]
#  [14 25 36]]
# b is 'broadcast' across each row of a
```

| Shape A | Shape B | Compatible? |
|---------|---------|-------------|
| (2, 3) | (3,) | ✅ Yes |
| (2, 3) | (2, 1) | ✅ Yes |
| (2, 3) | (2,) | ❌ No — Error |

---

## 13. View vs Copy

This is a critical concept. A **view** shares memory with the original array — changing the view changes the original. A **copy** is independent — changes to the copy do not affect the original. Slicing creates a view (fast, no extra memory), while `.copy()` creates a true copy. Failing to understand this is the #1 source of unexpected bugs in NumPy.

```python
a = np.array([1, 2, 3])

# VIEW — shares memory (dangerous!)
b = a.view()       # or b = a[:]
b[0] = 99
print(a)  # [99 2 3]  ← original changed!

# COPY — independent (safe)
c = a.copy()
c[0] = 100
print(a)  # [99 2 3]  ← original unchanged
```

> ⚡ Rule of thumb: Use `.copy()` when you want to modify an array without affecting the original.

---

## 14. Linear Algebra

NumPy's `linalg` module provides standard linear algebra operations. Matrix multiplication (`dot` or `@`) is different from element-wise multiplication (`*`). The `@` operator was introduced in Python 3.5 and is the preferred way to write matrix multiplication. These operations are fundamental for machine learning, physics simulations, and data transformations.

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# Matrix multiplication (3 equivalent ways)
np.dot(a, b)        # dot product
np.matmul(a, b)     # matrix multiply
a @ b                # preferred syntax (Python 3.5+)

# Other operations
np.linalg.inv(a)     # inverse matrix
np.linalg.det(a)     # determinant (-2.0)
np.linalg.eig(a)     # eigenvalues & eigenvectors
np.linalg.norm(a)    # matrix norm
```

---

## 15. Structured Arrays

Structured arrays let you store records with named fields, similar to a database row or spreadsheet. Each field can have a different data type (string, int, float). They are useful for lightweight tabular data when you don't want to use Pandas. You define a dtype with field names and types, then sort, filter, and index by field name.

```python
dt = np.dtype([('name', 'U10'), ('score', 'i4')])
students = np.array([
    ('Alice', 85),
    ('Bob', 92),
    ('Charlie', 78)
], dtype=dt)

students['name']     # ['Alice' 'Bob' 'Charlie']
students['score']    # [85 92 78]

# Sort by score
np.sort(students, order='score')
# [('Charlie', 78) ('Alice', 85) ('Bob', 92)]
```

---

## 16. Useful Utility Functions

NumPy includes many convenience functions that save you from writing common patterns manually. `unique()` removes duplicates, `clip()` constrains values to a range, `diff()` computes differences between consecutive elements, and `cumsum()` gives a running total.

```python
a = np.array([3, 1, 4, 1, 5, 9, 2, 6])

np.unique(a)           # [1 2 3 4 5 6 9]   — remove duplicates
np.flip(a)             # [6 2 9 5 1 4 1 3]  — reverse
np.clip(a, 2, 6)       # [3 2 4 2 5 6 2 6]  — clamp to [2, 6]
np.diff(a)             # [-2 3 -3 4 4 -7 4]  — differences
np.cumsum(a)           # [3 4 8 9 14 23 25 31]  — running total
np.prod(a)             # 6480    — product of all
np.percentile(a, 50)   # 3.5     — 50th percentile (median)
```

---

## 17. Random Number Generation

NumPy's random module generates pseudo-random numbers for simulations, sampling, and testing. Setting a seed makes results reproducible — the same seed always produces the same sequence of random numbers. This is essential for scientific experiments and debugging.

```python
np.random.seed(42)               # set seed for reproducibility

np.random.rand(3)                # [0.37 0.95 0.73]  uniform [0, 1)
np.random.randint(1, 100, 5)     # 5 random ints in [1, 100)
np.random.randn(3)               # normal distribution (mean=0, std=1)
np.random.choice([1,2,3], 2)     # randomly pick 2 elements
np.random.shuffle(a)             # shuffle array IN PLACE
np.random.permutation(a)         # return shuffled COPY
```

> ⚡ Always set `np.random.seed()` at the top of your script when you need reproducible results for testing or sharing.

---

## 18. Saving & Loading Data

NumPy can save arrays to disk in binary format (`.npy` for single arrays, `.npz` for multiple) or text format (`.txt`, `.csv`). Binary format is faster and preserves the exact data type. Text format is human-readable and compatible with other tools like Excel. Use `savez` to bundle multiple arrays into a single file.

```python
a = np.array([1, 2, 3, 4, 5])

# Binary format (fast, exact)
np.save('data.npy', a)
b = np.load('data.npy')

# Text format (human-readable)
np.savetxt('data.csv', a, delimiter=',')
c = np.loadtxt('data.csv', delimiter=',')

# Multiple arrays in one file
np.savez('data.npz', scores=a, names=b)
data = np.load('data.npz')
data['scores']    # access by name
```

---

## Quick Reference Cheat Sheet

| Category | Functions | Purpose |
|----------|-----------|---------|
| **Creation** | `array, zeros, ones, arange, linspace, eye, full` | Build new arrays with specific values or patterns |
| **Attributes** | `shape, ndim, size, dtype, itemsize, nbytes` | Inspect array structure and memory layout |
| **Indexing** | `a[0], a[1:3], a[::2], a[a>5], a[row, col]` | Access individual elements or sub-arrays |
| **Reshape** | `reshape, flatten, ravel, T, resize` | Change array dimensions without altering data |
| **Math** | `+, -, *, /, **, sqrt, exp, log, sin, abs` | Element-wise arithmetic and math functions |
| **Aggregates** | `sum, mean, min, max, std, var, argmin, argmax` | Reduce arrays to summary statistics |
| **Sorting** | `sort, argsort, searchsorted` | Order elements or find insertion indices |
| **Searching** | `where, any, all, argmin, argmax, nonzero` | Find elements matching conditions |
| **Stacking** | `hstack, vstack, concatenate, column_stack` | Combine multiple arrays into one |
| **Splitting** | `split, hsplit, vsplit, array_split` | Break one array into multiple arrays |
| **Linear Algebra** | `dot, matmul, @, inv, det, eig, norm` | Matrix operations for ML and science |
| **Random** | `rand, randint, randn, choice, shuffle, seed` | Generate random data for simulations |
| **File I/O** | `save, load, savetxt, loadtxt, savez` | Persist arrays to disk and reload them |
| **Utilities** | `unique, flip, clip, diff, cumsum, prod` | Common convenience operations |

---

## Performance Tips

1. **Avoid Python loops** — use vectorized NumPy operations instead (10–100× faster)
2. **Use views instead of copies** when you don't need to modify data independently
3. **Choose the right dtype** — `float32` uses half the memory of `float64` with acceptable precision
4. **Preallocate arrays** with `zeros()` or `empty()` instead of growing with `append()`
5. **Use boolean indexing** instead of loops for filtering — it's compiled C under the hood
