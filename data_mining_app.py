import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from apyori import apriori
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Crime Pattern Discovery & Analysis",
    page_icon="🔍",
    layout="wide"
)

# Main title
st.title("🔍 Crime Pattern Discovery using Data Mining")
st.markdown("### Discover hidden patterns and associations in crime data")

# Load and cache data
@st.cache_data
def load_data():
    """Load the crime dataset"""
    try:
        data = pd.read_csv('sample_crime_data.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset not found. Please upload sample_crime_data.csv file.")
        return None

# Clustering analysis
@st.cache_resource
def perform_clustering(data):
    """Perform K-means clustering on crime data"""
    # Prepare features for clustering
    le_city = LabelEncoder()
    le_crime = LabelEncoder()

    features_df = data.copy()
    features_df['City_encoded'] = le_city.fit_transform(data['City'])
    features_df['Crime_Type_encoded'] = le_crime.fit_transform(data['Crime_Type'])

    # Select features for clustering
    cluster_features = ['City_encoded', 'Crime_Type_encoded', 'Year', 'Month', 
                       'Population_Density', 'Economic_Index', 'Police_Stations', 
                       'Unemployment_Rate']

    X = features_df[cluster_features]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    features_df['Cluster'] = clusters

    return features_df, le_city, le_crime, kmeans

# Association Rules Mining
def mine_association_rules(data):
    """Mine association rules from crime data"""
    # Prepare transaction data
    transactions = []

    for _, row in data.iterrows():
        transaction = [
            f"City_{row['City']}",
            f"Crime_{row['Crime_Type']}",
            f"Year_{row['Year']}",
            f"Month_{row['Month']}",
            f"Density_{pd.cut([row['Population_Density']], bins=3, labels=['Low', 'Medium', 'High'])[0]}",
            f"Economic_{pd.cut([row['Economic_Index']], bins=3, labels=['Poor', 'Average', 'Good'])[0]}",
            f"Police_{pd.cut([row['Police_Stations']], bins=3, labels=['Few', 'Medium', 'Many'])[0]}",
            f"Unemployment_{pd.cut([row['Unemployment_Rate']], bins=3, labels=['Low', 'Medium', 'High'])[0]}"
        ]
        transactions.append(transaction)

    # Mine association rules
    rules = list(apriori(transactions, min_support=0.01, min_confidence=0.6))

    return rules

# Frequent pattern analysis
def analyze_frequent_patterns(data):
    """Analyze frequent crime patterns"""
    patterns = {}

    # Crime type frequency
    patterns['crime_types'] = data['Crime_Type'].value_counts()

    # City frequency
    patterns['cities'] = data['City'].value_counts()

    # Time patterns
    patterns['monthly'] = data['Month'].value_counts()
    patterns['yearly'] = data['Year'].value_counts()

    # Crime combinations
    patterns['city_crime'] = data.groupby(['City', 'Crime_Type']).size().reset_index(name='Count')

    return patterns

# Main app
def main():
    # Load data
    data = load_data()

    if data is not None:
        # Sidebar for analysis selection
        st.sidebar.header("Data Mining Analysis")
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type:",
            ["Clustering Analysis", "Association Rules", "Frequent Patterns", "Crime Hotspots"]
        )

        # Display data info
        st.sidebar.header("Dataset Information")
        st.sidebar.write(f"Total Records: {len(data)}")
        st.sidebar.write(f"Cities: {data['City'].nunique()}")
        st.sidebar.write(f"Crime Types: {data['Crime_Type'].nunique()}")
        st.sidebar.write(f"Date Range: {data['Year'].min()} - {data['Year'].max()}")

        if analysis_type == "Clustering Analysis":
            st.header("🎯 Crime Clustering Analysis")

            # Perform clustering
            with st.spinner("Performing clustering analysis..."):
                clustered_data, le_city, le_crime, kmeans = perform_clustering(data)

            st.success("Clustering analysis completed!")

            # Display cluster information
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Cluster Distribution")
                cluster_counts = clustered_data['Cluster'].value_counts().sort_index()
                fig_pie = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                                title="Crime Records by Cluster")
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.subheader("Cluster Characteristics")
                for cluster_id in sorted(clustered_data['Cluster'].unique()):
                    cluster_data = clustered_data[clustered_data['Cluster'] == cluster_id]
                    most_common_city = cluster_data['City'].mode()[0]
                    most_common_crime = cluster_data['Crime_Type'].mode()[0]
                    avg_crime_count = cluster_data['Crime_Count'].mean()

                    st.write(f"**Cluster {cluster_id}:**")
                    st.write(f"- Size: {len(cluster_data)} records")
                    st.write(f"- Main City: {most_common_city}")
                    st.write(f"- Main Crime: {most_common_crime}")
                    st.write(f"- Avg Crime Count: {avg_crime_count:.1f}")
                    st.write("---")

            # 3D Scatter plot of clusters
            st.subheader("3D Cluster Visualization")
            fig_3d = px.scatter_3d(
                clustered_data, 
                x='Population_Density', 
                y='Economic_Index', 
                z='Crime_Count',
                color='Cluster',
                hover_data=['City', 'Crime_Type'],
                title="Crime Data Clusters in 3D Space"
            )
            st.plotly_chart(fig_3d, use_container_width=True)

        elif analysis_type == "Association Rules":
            st.header("🔗 Association Rules Mining")

            with st.spinner("Mining association rules..."):
                rules = mine_association_rules(data)

            if rules:
                st.success(f"Found {len(rules)} association rules!")

                # Display top rules
                st.subheader("Top Association Rules")

                rule_data = []
                for rule in rules[:20]:  # Show top 20 rules
                    antecedent = list(rule.ordered_statistics[0].items_base)
                    consequent = list(rule.ordered_statistics[0].items_add)
                    support = rule.support
                    confidence = rule.ordered_statistics[0].confidence

                    rule_data.append({
                        'Antecedent': ' AND '.join(antecedent),
                        'Consequent': ' AND '.join(consequent),
                        'Support': f"{support:.3f}",
                        'Confidence': f"{confidence:.3f}"
                    })

                rules_df = pd.DataFrame(rule_data)
                st.dataframe(rules_df)

                # Interpretation
                st.subheader("Rule Interpretation")
                st.write("**Support**: How frequently the rule appears in the dataset")
                st.write("**Confidence**: How often the consequent is true when the antecedent is true")
                st.write("**Example**: If 'City_Mumbai AND Crime_Theft' → 'Density_High' with confidence 0.8, it means 80% of theft cases in Mumbai occur in high-density areas.")
            else:
                st.warning("No significant association rules found with current parameters.")

        elif analysis_type == "Frequent Patterns":
            st.header("📊 Frequent Pattern Analysis")

            # Analyze frequent patterns
            patterns = analyze_frequent_patterns(data)

            # Crime Type Patterns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Most Frequent Crime Types")
                fig_crime = px.bar(
                    x=patterns['crime_types'].values,
                    y=patterns['crime_types'].index,
                    orientation='h',
                    title="Crime Type Frequency"
                )
                st.plotly_chart(fig_crime, use_container_width=True)

            with col2:
                st.subheader("Most Affected Cities")
                fig_city = px.bar(
                    x=patterns['cities'].values,
                    y=patterns['cities'].index,
                    orientation='h',
                    title="City-wise Crime Frequency"
                )
                st.plotly_chart(fig_city, use_container_width=True)

            # Temporal Patterns
            st.subheader("Temporal Crime Patterns")

            col3, col4 = st.columns(2)

            with col3:
                fig_month = px.line(
                    x=patterns['monthly'].index,
                    y=patterns['monthly'].values,
                    title="Monthly Crime Pattern"
                )
                fig_month.update_xaxis(title="Month")
                fig_month.update_yaxis(title="Crime Count")
                st.plotly_chart(fig_month, use_container_width=True)

            with col4:
                fig_year = px.bar(
                    x=patterns['yearly'].index,
                    y=patterns['yearly'].values,
                    title="Yearly Crime Trends"
                )
                fig_year.update_xaxis(title="Year")
                fig_year.update_yaxis(title="Crime Count")
                st.plotly_chart(fig_year, use_container_width=True)

            # City-Crime Combination Heatmap
            st.subheader("Crime Pattern Heatmap")
            pivot_data = data.pivot_table(values='Crime_Count', index='City', columns='Crime_Type', aggfunc='sum', fill_value=0)
            fig_heatmap = px.imshow(
                pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                aspect="auto",
                title="City vs Crime Type Heatmap"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        elif analysis_type == "Crime Hotspots":
            st.header("🗺️ Crime Hotspot Analysis")

            # City selection for detailed analysis
            selected_city = st.selectbox("Select City for Analysis:", data['City'].unique())

            city_data = data[data['City'] == selected_city]

            col1, col2 = st.columns(2)

            with col1:
                # Crime distribution in selected city
                crime_dist = city_data['Crime_Type'].value_counts()
                fig_city_crime = px.pie(
                    values=crime_dist.values,
                    names=crime_dist.index,
                    title=f"Crime Distribution in {selected_city}"
                )
                st.plotly_chart(fig_city_crime, use_container_width=True)

            with col2:
                # Risk factors analysis
                st.subheader("Risk Factors Analysis")
                avg_unemployment = city_data['Unemployment_Rate'].mean()
                avg_density = city_data['Population_Density'].mean()
                avg_economic = city_data['Economic_Index'].mean()

                st.metric("Average Unemployment Rate", f"{avg_unemployment:.1f}%")
                st.metric("Average Population Density", f"{avg_density:.0f} per sq km")
                st.metric("Average Economic Index", f"{avg_economic:.2f}")

                # Risk level calculation
                risk_score = (avg_unemployment / 15) + (avg_density / 10000) + (1 - avg_economic)
                if risk_score > 1.5:
                    risk_level = "🔴 High Risk"
                elif risk_score > 1.0:
                    risk_level = "🟡 Medium Risk"
                else:
                    risk_level = "🟢 Low Risk"

                st.metric("Overall Risk Level", risk_level)

            # Temporal hotspot analysis
            st.subheader("Temporal Hotspot Pattern")
            temporal_pattern = city_data.groupby(['Month', 'Crime_Type']).size().reset_index(name='Count')
            fig_temporal = px.bar(
                temporal_pattern,
                x='Month',
                y='Count',
                color='Crime_Type',
                title=f"Monthly Crime Pattern in {selected_city}",
                barmode='stack'
            )
            st.plotly_chart(fig_temporal, use_container_width=True)

        # Display insights
        st.header("💡 Key Insights")

        insights = [
            "🔍 **Pattern Discovery**: Data mining reveals hidden patterns in crime data that traditional analysis might miss.",
            "📊 **Clustering**: Groups similar crime incidents together to identify common characteristics.",
            "🔗 **Association Rules**: Find relationships between different crime factors (e.g., location, time, type).",
            "📈 **Frequent Patterns**: Identify the most common crime types, locations, and time periods.",
            "🗺️ **Hotspot Analysis**: Pinpoint areas with high crime concentration for targeted prevention.",
            "⏰ **Temporal Analysis**: Discover time-based patterns for better resource allocation.",
        ]

        for insight in insights:
            st.write(insight)

        # Show sample data
        if st.checkbox("Show Raw Data Sample"):
            st.subheader("Sample Crime Data")
            st.dataframe(data.head(10))

if __name__ == "__main__":
    main()
