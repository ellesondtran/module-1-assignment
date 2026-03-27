# Chocolate Sales Exploratory Data Analysis
# INST414 - Data Science Techniques
# Dataset: Chocolate Sales 2023-2024 (Kaggle)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
# Note: sales.csv is ~71MB. We sampled the first 5,000 rows using:
#   head -5001 sales.csv > sales_sample.csv

sales     = pd.read_csv('sales_sample.csv')
products  = pd.read_csv('products.csv')
stores    = pd.read_csv('stores.csv')
customers = pd.read_csv('customers.csv')
calendar  = pd.read_csv('calendar.csv')

print("Sales shape:", sales.shape)
print(sales.head())

# ── 2. MERGE TABLES ───────────────────────────────────────────────────────────
df = sales.merge(products,  on='product_id',  how='left')
df = df.merge(stores,       on='store_id',    how='left')
df = df.merge(customers,    on='customer_id', how='left')

# Parse dates
df['order_date'] = pd.to_datetime(df['order_date'])
df['month']      = df['order_date'].dt.month
df['year']       = df['order_date'].dt.year

print("\nMerged shape:", df.shape)
print("Columns:", df.columns.tolist())

# ── 3. EXPLORATORY ANALYSIS ───────────────────────────────────────────────────

# Revenue by store type
print("\n--- Revenue by Store Type ---")
print(df.groupby('store_type')['revenue'].agg(['sum', 'mean', 'count']).round(2))

# Revenue by product category
print("\n--- Revenue by Product Category ---")
print(df.groupby('category')['revenue'].agg(['sum', 'mean', 'count']).round(2))

# Revenue by brand
print("\n--- Revenue by Brand ---")
print(df.groupby('brand')['revenue'].agg(['sum', 'mean', 'count'])
        .sort_values('sum', ascending=False).round(2))

# Revenue by month
print("\n--- Monthly Revenue ---")
print(df.groupby(['year', 'month'])['revenue'].sum().round(2))

# Loyalty member vs non-member
print("\n--- Loyalty Member vs Not ---")
print(df.groupby('loyalty_member')['revenue'].agg(['sum', 'mean', 'count']).round(2))

# ── 4. VISUALIZATIONS ─────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('What Drives Chocolate Sales? (2023–2024)', fontsize=16, fontweight='bold')

# Helper formatter
fmt = mticker.FuncFormatter(lambda x, _: f'${x:,.0f}')

# Plot 1: Revenue by Store Type
store_rev = df.groupby('store_type')['revenue'].sum().sort_values(ascending=False)
axes[0, 0].bar(store_rev.index, store_rev.values, color=sns.color_palette('muted', len(store_rev)))
axes[0, 0].set_title('Total Revenue by Store Type')
axes[0, 0].set_xlabel('Store Type')
axes[0, 0].set_ylabel('Total Revenue ($)')
axes[0, 0].yaxis.set_major_formatter(fmt)

# Plot 2: Revenue by Product Category
cat_rev = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
axes[0, 1].bar(cat_rev.index, cat_rev.values, color=sns.color_palette('muted', len(cat_rev)))
axes[0, 1].set_title('Total Revenue by Product Category')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Total Revenue ($)')
axes[0, 1].yaxis.set_major_formatter(fmt)

# Plot 3: Monthly Revenue Trend
monthly = df.groupby(['year', 'month'])['revenue'].sum().reset_index()
monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
axes[1, 0].plot(monthly['date'], monthly['revenue'], marker='o', linewidth=2, color='steelblue')
axes[1, 0].set_title('Monthly Revenue Trend (2023–2024)')
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Revenue ($)')
axes[1, 0].yaxis.set_major_formatter(fmt)
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Revenue by Brand
brand_rev = df.groupby('brand')['revenue'].sum().sort_values(ascending=False)
axes[1, 1].bar(brand_rev.index, brand_rev.values, color=sns.color_palette('muted', len(brand_rev)))
axes[1, 1].set_title('Total Revenue by Brand')
axes[1, 1].set_xlabel('Brand')
axes[1, 1].set_ylabel('Total Revenue ($)')
axes[1, 1].yaxis.set_major_formatter(fmt)

plt.tight_layout()
plt.savefig('chocolate_eda_charts.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nChart saved as chocolate_eda_charts.png")
