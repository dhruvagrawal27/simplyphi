import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# STEP 1: DATA LOADING AND INITIAL INSPECTION
# ============================================================================

def load_and_inspect_data(filepath):
    """Load data and perform initial inspection"""
    print("="*80)
    print("STEP 1: DATA LOADING AND INITIAL INSPECTION")
    print("="*80)
    
    df = pd.read_csv(filepath)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 Column Names and Types:")
    print(df.dtypes)
    
    print(f"\n🔍 First 5 Rows:")
    print(df.head())
    
    print(f"\n📈 Basic Statistics:")
    print(df.describe())
    
    print(f"\n🔢 Memory Usage:")
    print(df.memory_usage(deep=True))
    
    return df

# ============================================================================
# STEP 2: DATA QUALITY ASSESSMENT
# ============================================================================

def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    print("\n" + "="*80)
    print("STEP 2: DATA QUALITY ASSESSMENT")
    print("="*80)
    
    # Missing values analysis
    print("\n🔴 Missing Values Analysis:")
    missing = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing = missing[missing['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    print(missing)
    
    # Duplicate analysis
    print(f"\n🔄 Duplicate Rows: {df.duplicated().sum()}")
    
    # Data type issues
    print("\n📊 Data Type Analysis:")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"  {col}: {unique_count} unique values")
    
    # Outlier detection for numerical columns
    print("\n⚠️ Potential Outliers (using IQR method):")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)][col]
        if len(outliers) > 0:
            print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
    
    return missing

# ============================================================================
# STEP 3: FEATURE ENGINEERING & DATA CLEANING
# ============================================================================

def clean_and_engineer_features(df):
    """Clean data and create new features"""
    print("\n" + "="*80)
    print("STEP 3: DATA CLEANING & FEATURE ENGINEERING")
    print("="*80)
    
    df_clean = df.copy()
    
    # Convert date column to datetime
    if 'listing_update_date' in df_clean.columns:
        df_clean['listing_update_date'] = pd.to_datetime(df_clean['listing_update_date'], errors='coerce')
        df_clean['listing_year'] = df_clean['listing_update_date'].dt.year
        df_clean['listing_month'] = df_clean['listing_update_date'].dt.month
        df_clean['listing_day_of_week'] = df_clean['listing_update_date'].dt.dayofweek
        df_clean['days_since_update'] = (datetime.now() - df_clean['listing_update_date']).dt.days
    
    # Standardize property type
    df_clean['type_standardized'] = df_clean['type'].str.lower().str.strip()
    
    # Create price categories
    df_clean['price_category'] = pd.cut(df_clean['price'], 
                                         bins=[0, 1000, 2000, 3000, 5000, float('inf')],
                                         labels=['Budget', 'Affordable', 'Mid-Range', 'Premium', 'Luxury'])
    
    # Total rooms
    df_clean['total_rooms'] = df_clean['bedrooms'].fillna(0) + df_clean['bathrooms'].fillna(0)
    
    # Price per bedroom (handle 0 bedrooms)
    df_clean['price_per_bedroom'] = df_clean.apply(
        lambda x: x['price'] / x['bedrooms'] if x['bedrooms'] > 0 else x['price'], axis=1
    )
    
    # Binary flags
    df_clean['has_flood_risk'] = df_clean['flood_risk'].notna() & (df_clean['flood_risk'] != 'None')
    df_clean['is_studio'] = df_clean['bedrooms'] == 0
    
    # Crime score handling
    df_clean['crime_score_weight'] = pd.to_numeric(df_clean['crime_score_weight'], errors='coerce')
    df_clean['crime_category'] = pd.cut(df_clean['crime_score_weight'], 
                                         bins=[0, 3, 6, 10],
                                         labels=['Low Crime', 'Medium Crime', 'High Crime'])
    
    print(f"✅ Original columns: {len(df.columns)}")
    print(f"✅ New columns created: {len(df_clean.columns) - len(df.columns)}")
    print(f"✅ New feature names: {list(set(df_clean.columns) - set(df.columns))}")
    
    return df_clean

# ============================================================================
# STEP 4: UNIVARIATE ANALYSIS
# ============================================================================

def univariate_analysis(df):
    """Analyze individual variables"""
    print("\n" + "="*80)
    print("STEP 4: UNIVARIATE ANALYSIS")
    print("="*80)
    
    # Price distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Price Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Histogram
    axes[0, 0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Price Distribution (Histogram)')
    axes[0, 0].set_xlabel('Price')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['price'].mean(), color='red', linestyle='--', label=f'Mean: £{df["price"].mean():.0f}')
    axes[0, 0].axvline(df['price'].median(), color='green', linestyle='--', label=f'Median: £{df["price"].median():.0f}')
    axes[0, 0].legend()
    
    # Box plot
    axes[0, 1].boxplot(df['price'].dropna(), vert=True)
    axes[0, 1].set_title('Price Box Plot (Outlier Detection)')
    axes[0, 1].set_ylabel('Price')
    
    # Log scale
    axes[1, 0].hist(np.log10(df['price'][df['price'] > 0]), bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 0].set_title('Price Distribution (Log Scale)')
    axes[1, 0].set_xlabel('Log10(Price)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Q-Q plot
    stats.probplot(df['price'].dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normality Check)')
    
    plt.tight_layout()
    plt.savefig('univariate_price_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💰 Price Statistics:")
    print(f"  Mean: £{df['price'].mean():.2f}")
    print(f"  Median: £{df['price'].median():.2f}")
    print(f"  Std Dev: £{df['price'].std():.2f}")
    print(f"  Min: £{df['price'].min():.2f}")
    print(f"  Max: £{df['price'].max():.2f}")
    print(f"  Skewness: {df['price'].skew():.2f}")
    print(f"  Kurtosis: {df['price'].kurtosis():.2f}")
    
    # Bedrooms and Bathrooms distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    df['bedrooms'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Bedrooms Distribution')
    axes[0].set_xlabel('Number of Bedrooms')
    axes[0].set_ylabel('Count')
    
    df['bathrooms'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color='lightcoral', edgecolor='black')
    axes[1].set_title('Bathrooms Distribution')
    axes[1].set_xlabel('Number of Bathrooms')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('rooms_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# STEP 5: BIVARIATE ANALYSIS
# ============================================================================

def bivariate_analysis(df):
    """Analyze relationships between variables"""
    print("\n" + "="*80)
    print("STEP 5: BIVARIATE ANALYSIS")
    print("="*80)
    
    # Price vs Bedrooms
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bivariate Relationships with Price', fontsize=16, fontweight='bold')
    
    # Price vs Bedrooms (Box plot)
    df.boxplot(column='price', by='bedrooms', ax=axes[0, 0])
    axes[0, 0].set_title('Price by Number of Bedrooms')
    axes[0, 0].set_xlabel('Bedrooms')
    axes[0, 0].set_ylabel('Price')
    plt.sca(axes[0, 0])
    plt.xticks(rotation=0)
    
    # Price vs Bathrooms (Box plot)
    df.boxplot(column='price', by='bathrooms', ax=axes[0, 1])
    axes[0, 1].set_title('Price by Number of Bathrooms')
    axes[0, 1].set_xlabel('Bathrooms')
    axes[0, 1].set_ylabel('Price')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=0)
    
    # Price vs Property Type
    price_by_type = df.groupby('type_standardized')['price'].median().sort_values(ascending=False).head(10)
    price_by_type.plot(kind='barh', ax=axes[1, 0], color='teal')
    axes[1, 0].set_title('Median Price by Property Type (Top 10)')
    axes[1, 0].set_xlabel('Median Price')
    axes[1, 0].set_ylabel('Property Type')
    
    # Price vs Crime Score
    axes[1, 1].scatter(df['crime_score_weight'], df['price'], alpha=0.5, s=10)
    axes[1, 1].set_title('Price vs Crime Score')
    axes[1, 1].set_xlabel('Crime Score Weight')
    axes[1, 1].set_ylabel('Price')
    
    plt.tight_layout()
    plt.savefig('bivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation analysis
    numerical_cols = ['price', 'bedrooms', 'bathrooms', 'crime_score_weight', 'total_rooms', 'price_per_bedroom']
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🔗 Correlation with Price:")
    print(corr_matrix['price'].sort_values(ascending=False))

# ============================================================================
# STEP 6: GEOGRAPHICAL ANALYSIS
# ============================================================================

def geographical_analysis(df):
    """Analyze location-based patterns"""
    print("\n" + "="*80)
    print("STEP 6: GEOGRAPHICAL ANALYSIS")
    print("="*80)
    
    # Top locations by count
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Geographical Analysis', fontsize=16, fontweight='bold')
    
    # Property count by location
    top_locations = df['address'].value_counts().head(15)
    top_locations.plot(kind='barh', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Top 15 Locations by Property Count')
    axes[0, 0].set_xlabel('Number of Properties')
    axes[0, 0].set_ylabel('Location')
    
    # Average price by location
    avg_price_location = df.groupby('address')['price'].mean().sort_values(ascending=False).head(15)
    avg_price_location.plot(kind='barh', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Top 15 Most Expensive Locations (Avg Price)')
    axes[0, 1].set_xlabel('Average Price')
    axes[0, 1].set_ylabel('Location')
    
    # Crime score by location
    crime_by_location = df.groupby('address')['crime_score_weight'].mean().sort_values(ascending=False).head(15)
    crime_by_location.plot(kind='barh', ax=axes[1, 0], color='darkred')
    axes[1, 0].set_title('Top 15 Locations by Crime Score')
    axes[1, 0].set_xlabel('Average Crime Score')
    axes[1, 0].set_ylabel('Location')
    
    # Price distribution across top locations
    top_10_locations = df['address'].value_counts().head(10).index
    df_top_locations = df[df['address'].isin(top_10_locations)]
    df_top_locations.boxplot(column='price', by='address', ax=axes[1, 1], rot=45)
    axes[1, 1].set_title('Price Distribution - Top 10 Locations')
    axes[1, 1].set_xlabel('Location')
    axes[1, 1].set_ylabel('Price')
    
    plt.tight_layout()
    plt.savefig('geographical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📍 Total Unique Locations: {df['address'].nunique()}")
    print(f"\n🏆 Top 5 Locations by Property Count:")
    print(top_locations.head())
    print(f"\n💎 Top 5 Most Expensive Locations:")
    print(avg_price_location.head())

# ============================================================================
# STEP 7: TEMPORAL ANALYSIS
# ============================================================================

def temporal_analysis(df):
    """Analyze time-based patterns"""
    print("\n" + "="*80)
    print("STEP 7: TEMPORAL ANALYSIS")
    print("="*80)
    
    if 'listing_update_date' not in df.columns:
        print("⚠️ No temporal data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Temporal Analysis', fontsize=16, fontweight='bold')
    
    # Listings by year
    df['listing_year'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='mediumpurple')
    axes[0, 0].set_title('Listings by Year')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Listings by month
    df['listing_month'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 1], color='lightseagreen')
    axes[0, 1].set_title('Listings by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Listings by day of week
    df['listing_day_of_week'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 0], color='gold')
    axes[1, 0].set_title('Listings by Day of Week')
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    
    # Price trend over time (monthly average)
    monthly_price = df.groupby(['listing_year', 'listing_month'])['price'].mean()
    monthly_price.plot(ax=axes[1, 1], marker='o', color='darkgreen')
    axes[1, 1].set_title('Average Price Trend Over Time')
    axes[1, 1].set_xlabel('Year-Month')
    axes[1, 1].set_ylabel('Average Price')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# STEP 8: MULTIVARIATE ANALYSIS
# ============================================================================

def multivariate_analysis(df):
    """Analyze complex relationships"""
    print("\n" + "="*80)
    print("STEP 8: MULTIVARIATE ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multivariate Analysis', fontsize=16, fontweight='bold')
    
    # Price by bedrooms and property type
    pivot_data = df.pivot_table(values='price', index='bedrooms', 
                                  columns='type_standardized', aggfunc='median')
    pivot_data = pivot_data.iloc[:, :10]  # Top 10 property types
    pivot_data.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('Median Price by Bedrooms & Property Type')
    axes[0, 0].set_xlabel('Number of Bedrooms')
    axes[0, 0].set_ylabel('Median Price')
    axes[0, 0].legend(title='Property Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # 3D scatter: Bedrooms, Bathrooms, Price (using size and color)
    scatter = axes[0, 1].scatter(df['bedrooms'], df['bathrooms'], 
                                  c=df['price'], s=df['price']/50, 
                                  alpha=0.6, cmap='viridis')
    axes[0, 1].set_title('Bedrooms vs Bathrooms (colored by Price)')
    axes[0, 1].set_xlabel('Bedrooms')
    axes[0, 1].set_ylabel('Bathrooms')
    plt.colorbar(scatter, ax=axes[0, 1], label='Price')
    
    # Crime score impact on price by property type
    top_types = df['type_standardized'].value_counts().head(5).index
    df_top_types = df[df['type_standardized'].isin(top_types)]
    for prop_type in top_types:
        subset = df_top_types[df_top_types['type_standardized'] == prop_type]
        axes[1, 0].scatter(subset['crime_score_weight'], subset['price'], 
                          label=prop_type, alpha=0.6, s=20)
    axes[1, 0].set_title('Crime Score vs Price by Property Type')
    axes[1, 0].set_xlabel('Crime Score Weight')
    axes[1, 0].set_ylabel('Price')
    axes[1, 0].legend(fontsize=8)
    
    # Price distribution by bedroom count and crime category
    if 'crime_category' in df.columns:
        df_filtered = df[df['bedrooms'].isin([1, 2, 3, 4])]
        df_filtered.boxplot(column='price', by=['bedrooms', 'crime_category'], 
                           ax=axes[1, 1], rot=45)
        axes[1, 1].set_title('Price by Bedrooms & Crime Category')
        axes[1, 1].set_xlabel('Bedrooms - Crime Category')
        axes[1, 1].set_ylabel('Price')
    
    plt.tight_layout()
    plt.savefig('multivariate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# STEP 9: ADVANCED INSIGHTS FOR RAG APPLICATION
# ============================================================================

def rag_specific_insights(df):
    """Generate insights specifically useful for RAG queries"""
    print("\n" + "="*80)
    print("STEP 9: RAG-SPECIFIC INSIGHTS")
    print("="*80)
    
    # Answer common RAG queries
    print("\n💡 Pre-computed Insights for RAG Queries:\n")
    
    # 1. Average price by bedroom count
    print("1️⃣ Average Price by Bedroom Count:")
    avg_price_bedrooms = df.groupby('bedrooms')['price'].mean().sort_index()
    for bed, price in avg_price_bedrooms.items():
        print(f"   {bed} bedroom(s): £{price:.2f}")
    
    # 2. Properties under budget with conditions
    print("\n2️⃣ Budget Properties (under £1000) with 2+ bathrooms:")
    budget_props = df[(df['price'] < 1000) & (df['bathrooms'] >= 2)]
    print(f"   Found: {len(budget_props)} properties")
    print(f"   Average price: £{budget_props['price'].mean():.2f}")
    
    # 3. High crime areas
    print("\n3️⃣ Areas with Highest Crime:")
    high_crime = df.groupby('address')['crime_score_weight'].mean().sort_values(ascending=False).head(10)
    for location, score in high_crime.items():
        print(f"   {location}: {score:.2f}")
    
    # 4. Studio vs 2-bed comparison
    print("\n4️⃣ Studio vs 2-Bedroom Comparison:")
    studio_price = df[df['bedrooms'] == 0]['price'].mean()
    two_bed_price = df[df['bedrooms'] == 2]['price'].mean()
    print(f"   Studio average: £{studio_price:.2f}")
    print(f"   2-bedroom average: £{two_bed_price:.2f}")
    print(f"   Difference: £{two_bed_price - studio_price:.2f} ({((two_bed_price/studio_price)-1)*100:.1f}% more)")
    
    # 5. Best value properties
    print("\n5️⃣ Best Value (Price per Bedroom) by Property Type:")
    best_value = df.groupby('type_standardized')['price_per_bedroom'].mean().sort_values().head(10)
    for prop_type, value in best_value.items():
        print(f"   {prop_type}: £{value:.2f} per bedroom")
    
    # 6. Flood risk properties
    print("\n6️⃣ Flood Risk Analysis:")
    flood_count = df['has_flood_risk'].sum()
    print(f"   Properties with flood risk: {flood_count} ({flood_count/len(df)*100:.2f}%)")
    
    # 7. New homes
    print("\n7️⃣ New Homes:")
    new_homes = df[df['is_new_home'] == True]
    print(f"   New homes: {len(new_homes)} ({len(new_homes)/len(df)*100:.2f}%)")
    print(f"   Avg price (new): £{new_homes['price'].mean():.2f}")
    print(f"   Avg price (existing): £{df[df['is_new_home'] == False]['price'].mean():.2f}")
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('RAG Query Insights Summary', fontsize=16, fontweight='bold')
    
    # Average price by bedrooms
    avg_price_bedrooms.plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Average Price by Bedroom Count')
    axes[0, 0].set_xlabel('Bedrooms')
    axes[0, 0].set_ylabel('Average Price (£)')
    axes[0, 0].tick_params(axis='x', rotation=0)
    
    # Top 10 high crime areas
    high_crime.plot(kind='barh', ax=axes[0, 1], color='darkred')
    axes[0, 1].set_title('Top 10 High Crime Areas')
    axes[0, 1].set_xlabel('Average Crime Score')
    
    # Price comparison: Studio vs Multi-bedroom
    bedroom_categories = df.groupby(df['bedrooms'].apply(lambda x: 'Studio' if x == 0 else f'{int(x)}-bed'))['price'].mean()
    bedroom_categories.plot(kind='bar', ax=axes[1, 0], color='teal')
    axes[1, 0].set_title('Price Comparison by Bedroom Category')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Average Price (£)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Property type distribution
    df['type_standardized'].value_counts().head(10).plot(kind='pie', ax=axes[1, 1], autopct='%1.1f%%')
    axes[1, 1].set_title('Property Type Distribution (Top 10)')
    axes[1, 1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('rag_insights_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# STEP 10: GENERATE COMPREHENSIVE REPORT
# ============================================================================

def generate_report(df, df_clean, missing_info):
    """Generate final comprehensive report"""
    print("\n" + "="*80)
    print("STEP 10: COMPREHENSIVE DATA REPORT")
    print("="*80)
    
    report = f"""
{'='*80}
PROPERTY DATA EDA - COMPREHENSIVE REPORT
{'='*80}

1. DATASET OVERVIEW
   - Total Properties: {len(df):,}
   - Total Features: {len(df.columns)}
   - Date Range: {df['listing_update_date'].min()} to {df['listing_update_date'].max()}
   - Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

2. DATA QUALITY
   - Missing Values: {df.isnull().sum().sum()} ({df.isnull().sum().sum()/df.size*100:.2f}%)
   - Duplicate Rows: {df.duplicated().sum()}
   - Unique Locations: {df['address'].nunique()}
   - Unique Property Types: {df['type'].nunique()}

3. KEY STATISTICS
   Price Analysis:
   - Mean Price: £{df['price'].mean():.2f}
   - Median Price: £{df['price'].median():.2f}
   - Price Range: £{df['price'].min():.2f} - £{df['price'].max():.2f}
   - Standard Deviation: £{df['price'].std():.2f}
   
   Property Characteristics:
   - Avg Bedrooms: {df['bedrooms'].mean():.2f}
   - Avg Bathrooms: {df['bathrooms'].mean():.2f}
   - Most Common Type: {df['type'].mode()[0]}
   
   Safety & Risk:
   - Avg Crime Score: {df['crime_score_weight'].mean():.2f}
   - Properties with Flood Risk: {df_clean['has_flood_risk'].sum()} ({df_clean['has_flood_risk'].sum()/len(df)*100:.2f}%)
   - New Homes: {df[df['is_new_home']==True].shape[0]} ({df[df['is_new_home']==True].shape[0]/len(df)*100:.2f}%)

4. TOP INSIGHTS FOR RAG APPLICATION
   - Most expensive location: {df.groupby('address')['price'].mean().idxmax()}
   - Safest location (lowest crime): {df.groupby('address')['crime_score_weight'].mean().idxmin()}
   - Best value property type: {df.groupby('type_standardized')['price_per_bedroom'].mean().idxmin()}
   - Peak listing month: {df['listing_month'].mode()[0] if 'listing_month' in df.columns else 'N/A'}

5. RECOMMENDATIONS FOR RAG IMPLEMENTATION
   ✓ Index property descriptions with location, price, bedrooms, bathrooms
   ✓ Create embeddings for natural language queries about pricing
   ✓ Store crime score data for safety-related queries
   ✓ Enable filtering by price range, bedroom count, and location
   ✓ Implement comparison queries (studio vs multi-bedroom)
   ✓ Add temporal context for listing freshness
   ✓ Include flood risk and new home status in metadata

6. DATA QUALITY ISSUES TO ADDRESS
   {missing_info.to_string() if not missing_info.empty else '   No significant missing values'}

{'='*80}
END OF REPORT
{'='*80}
    """
    
    print(report)
    
    # Save report to file
    with open('property_data_eda_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✅ Report saved to: property_data_eda_report.txt")

# ============================================================================
# STEP 11: EXPORT CLEANED DATA
# ============================================================================

def export_cleaned_data(df_clean):
    """Export cleaned dataset for RAG application"""
    print("\n" + "="*80)
    print("STEP 11: EXPORTING CLEANED DATA")
    print("="*80)
    
    # Save full cleaned dataset
    output_path = 'property_data_cleaned.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Cleaned dataset saved to: {output_path}")
    
    # Create a summary dataset for quick lookups
    summary_df = df_clean.groupby('address').agg({
        'price': ['mean', 'median', 'min', 'max', 'count'],
        'bedrooms': 'mean',
        'bathrooms': 'mean',
        'crime_score_weight': 'mean',
        'has_flood_risk': 'sum'
    }).round(2)
    
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df.to_csv('property_summary_by_location.csv')
    print(f"✅ Location summary saved to: property_summary_by_location.csv")
    
    # Create embeddings-ready dataset
    embeddings_df = df_clean[[
        'type_standardized', 'bedrooms', 'bathrooms', 'price', 
        'address', 'crime_score_weight', 'flood_risk', 'is_new_home',
        'property_type_full_description', 'price_category'
    ]].copy()
    
    # Create text description for embedding
    embeddings_df['text_description'] = embeddings_df.apply(
        lambda x: f"{x['property_type_full_description'] if pd.notna(x['property_type_full_description']) else x['type_standardized']} "
                  f"with {x['bedrooms']} bedrooms and {x['bathrooms']} bathrooms in {x['address']}. "
                  f"Price: £{x['price']}. Crime score: {x['crime_score_weight']}. "
                  f"{'New home. ' if x['is_new_home'] else ''}"
                  f"{'Flood risk area. ' if pd.notna(x['flood_risk']) and x['flood_risk'] != 'None' else ''}",
        axis=1
    )
    
    embeddings_df.to_csv('property_embeddings_ready.csv', index=False)
    print(f"✅ Embeddings-ready dataset saved to: property_embeddings_ready.csv")
    
    print(f"\n📊 Exported Files Summary:")
    print(f"   1. property_data_cleaned.csv - Full cleaned dataset ({len(df_clean)} rows)")
    print(f"   2. property_summary_by_location.csv - Location aggregates ({len(summary_df)} locations)")
    print(f"   3. property_embeddings_ready.csv - Ready for vector embeddings ({len(embeddings_df)} rows)")

# ============================================================================
# STEP 12: SAMPLE QUERIES FOR RAG TESTING
# ============================================================================

def generate_sample_queries(df):
    """Generate sample queries and expected answers for RAG testing"""
    print("\n" + "="*80)
    print("STEP 12: SAMPLE QUERIES FOR RAG TESTING")
    print("="*80)
    
    queries = {
        "Query 1": {
            "question": "What's the average price of 3-bedroom homes?",
            "answer": f"£{df[df['bedrooms'] == 3]['price'].mean():.2f}",
            "count": len(df[df['bedrooms'] == 3])
        },
        "Query 2": {
            "question": "Find properties under £1000 with 2+ bathrooms",
            "answer": f"{len(df[(df['price'] < 1000) & (df['bathrooms'] >= 2)])} properties found",
            "avg_price": f"£{df[(df['price'] < 1000) & (df['bathrooms'] >= 2)]['price'].mean():.2f}"
        },
        "Query 3": {
            "question": "Which area has the most crime?",
            "answer": df.groupby('address')['crime_score_weight'].mean().idxmax(),
            "crime_score": f"{df.groupby('address')['crime_score_weight'].mean().max():.2f}"
        },
        "Query 4": {
            "question": "Compare prices between studio and 2-bed homes",
            "studio_avg": f"£{df[df['bedrooms'] == 0]['price'].mean():.2f}",
            "two_bed_avg": f"£{df[df['bedrooms'] == 2]['price'].mean():.2f}",
            "difference": f"£{df[df['bedrooms'] == 2]['price'].mean() - df[df['bedrooms'] == 0]['price'].mean():.2f}"
        },
        "Query 5": {
            "question": "What's the most affordable location?",
            "answer": df.groupby('address')['price'].mean().idxmin(),
            "avg_price": f"£{df.groupby('address')['price'].mean().min():.2f}"
        },
        "Query 6": {
            "question": "How many new homes are available?",
            "answer": f"{len(df[df['is_new_home'] == True])} new homes",
            "percentage": f"{len(df[df['is_new_home'] == True])/len(df)*100:.2f}%"
        },
        "Query 7": {
            "question": "Properties with flood risk in low crime areas?",
            "answer": len(df[(df['has_flood_risk'] == True) & (df['crime_score_weight'] <= 3)]),
            "avg_price": f"£{df[(df['has_flood_risk'] == True) & (df['crime_score_weight'] <= 3)]['price'].mean():.2f}"
        },
        "Query 8": {
            "question": "Best value property type (price per bedroom)?",
            "answer": df.groupby('type_standardized')['price_per_bedroom'].mean().idxmin(),
            "value": f"£{df.groupby('type_standardized')['price_per_bedroom'].mean().min():.2f}/bedroom"
        }
    }
    
    print("\n🔍 Sample Test Queries with Expected Answers:\n")
    for key, query in queries.items():
        print(f"\n{key}:")
        for k, v in query.items():
            print(f"   {k}: {v}")
    
    # Save to JSON for RAG testing
    import json
    with open('rag_test_queries.json', 'w') as f:
        json.dump(queries, f, indent=2)
    
    print("\n✅ Test queries saved to: rag_test_queries.json")

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Execute complete EDA pipeline"""
    print("\n" + "🏠"*40)
    print("PROPERTY DATA - COMPREHENSIVE EDA PIPELINE")
    print("🏠"*40 + "\n")
    
    # File path
    filepath = r"F:\simplyphi\data Science Project\data Science Project\Property_data.csv"
    
    try:
        # Execute all steps
        df = load_and_inspect_data(filepath)
        missing_info = assess_data_quality(df)
        df_clean = clean_and_engineer_features(df)
        
        univariate_analysis(df_clean)
        bivariate_analysis(df_clean)
        geographical_analysis(df_clean)
        temporal_analysis(df_clean)
        multivariate_analysis(df_clean)
        rag_specific_insights(df_clean)
        
        generate_report(df, df_clean, missing_info)
        export_cleaned_data(df_clean)
        generate_sample_queries(df_clean)
        
        print("\n" + "="*80)
        print("✅ EDA PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n📁 Generated Files:")
        print("   1. univariate_price_analysis.png")
        print("   2. rooms_distribution.png")
        print("   3. bivariate_analysis.png")
        print("   4. correlation_matrix.png")
        print("   5. geographical_analysis.png")
        print("   6. temporal_analysis.png")
        print("   7. multivariate_analysis.png")
        print("   8. rag_insights_summary.png")
        print("   9. property_data_cleaned.csv")
        print("   10. property_summary_by_location.csv")
        print("   11. property_embeddings_ready.csv")
        print("   12. property_data_eda_report.txt")
        print("   13. rag_test_queries.json")
        
        print("\n🚀 Next Steps for RAG Application:")
        print("   1. Use 'property_embeddings_ready.csv' for creating vector embeddings")
        print("   2. Store embeddings in your chosen vector DB (Pinecone/Weaviate/ChromaDB)")
        print("   3. Use 'rag_test_queries.json' for testing your RAG system")
        print("   4. Refer to 'property_data_eda_report.txt' for data insights")
        print("   5. Use correlation insights for feature engineering in your LLM prompts")
        
        return df_clean
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    df_cleaned = main()