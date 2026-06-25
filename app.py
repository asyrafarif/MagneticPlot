import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Page Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Peak Analysis & Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom header styling
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    </style>
""", unsafe_allowed_html=True)

# ==========================================
# 2. Sidebar: Logos & Developer Metadata
# ==========================================
st.sidebar.image(
    "https://brand.umpsa.edu.my/images/logo-umpsa-full-color2.png", 
    use_container_width=True
)

st.sidebar.image(
    "https://www.majalahsains.com/wp-content/uploads/2012/05/Logo-Agensi-Nuklear-Malaysia.png",
    use_container_width=True
)    

st.sidebar.markdown("## Data Analytics Sidebar")
st.sidebar.markdown("---")

# File Upload Panel
st.sidebar.header("📥 Upload Datasets")
uploaded_files = st.sidebar.file_uploader(
    "Upload Text (.txt) or CSV files",
    type=["txt", "csv"],
    accept_multiple_files=True
)

# Sidebar Hyperparameters Control
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Clustering Configuration")
cluster_features = st.sidebar.selectbox(
    "Select Clustering Features",
    ["Peak Y-Value Only", "Both Peak X and Y-Values"]
)
n_clusters = st.sidebar.slider("Number of Clusters (K)", min_value=2, max_value=5, value=2, step=1)

# Developer Info Block
st.sidebar.markdown("---")
st.sidebar.markdown("### Developers:")
st.sidebar.write("**Asyraf Arif Bin Abu Bakar**")
st.sidebar.caption("Leading Edge NDT Group\nAgensi Nuklear Malaysia\nEmail: asyrafarif@nm.gov.my")

st.sidebar.write("**Dr. Hanafi Ithnin**")
st.sidebar.caption("Bahagian Teknologi Industri (BTI)\nAgensi Nuklear Malaysia\nEmail: hanafi_i@nm.gov.my")

# ==========================================
# 3. Robust Data Parsing Engine
# ==========================================
def parse_raw_file(file_obj):
    """ Cleans metadata headers/annotations and extracts numeric tabular data """
    try:
        content = file_obj.read().decode("utf-8")
        # Remove metadata text injected by logs or indexing tags like [source: XXX]
        clean_content = re.sub(r'\[source:\s*\d+\]', '', content)
        
        # Parse data using space separation, treating rows starting with % as comments
        df = pd.read_csv(io.StringIO(clean_content), comment='%', sep=r'\s+', header=None)
        
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ['x', 'y']
            return df
    except Exception as e:
        # Secondary tokenized fallback parser if formatting has irregular linebreaks
        try:
            tokens = []
            for line in clean_content.split('\n'):
                line = line.strip()
                if not line or line.startswith('%'):
                    continue
                tokens.extend(line.replace(',', ' ').split())
            numeric_tokens = [float(t) for t in tokens if t]
            if len(numeric_tokens) >= 2:
                x = numeric_tokens[0::2]
                y = numeric_tokens[1::2]
                min_len = min(len(x), len(y))
                return pd.DataFrame({'x': x[:min_len], 'y': y[:min_len]})
        except Exception:
            return None
    return None

# ==========================================
# 4. Main Application Layout
# ==========================================
st.markdown('<div class="main-title">Peak Analysis & Clustering Dashboard</div>', unsafe_allowed_html=True)
st.markdown('<div class="sub-title">An interactive EDA platform for curve plotting, feature extraction, and K-Means clustering allocation</div>', unsafe_allowed_html=True)

if uploaded_files:
    all_data = {}
    peak_records = []
    total_data_points = 0
    
    # Process files sequentially
    for f in uploaded_files:
        df_parsed = parse_raw_file(f)
        if df_parsed is not None and not df_parsed.empty:
            all_data[f.name] = df_parsed
            total_data_points += len(df_parsed)
            
            # Extract Peak Features (Global Maxima per curve)
            max_idx = df_parsed['y'].idxmax()
            peak_x = df_parsed.loc[max_idx, 'x']
            peak_y = df_parsed.loc[max_idx, 'y']
            
            peak_records.append({
                "Filename": f.name,
                "Peak_X": peak_x,
                "Peak_Y": peak_y,
                "Total_Points": len(df_parsed)
            })
    
    if peak_records:
        df_peaks = pd.DataFrame(peak_records)
        
        # 4a. Summary High-Level KPI Cards
        st.subheader("📊 Executive Overview Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Uploaded Files Count", len(all_data))
        m_col2.metric("Aggregated Data Points", total_data_points)
        m_col3.metric("Minimum Peak $y$", f"{df_peaks['Peak_Y'].min():.5f}")
        m_col4.metric("Maximum Peak $y$", f"{df_peaks['Peak_Y'].max():.5f}")
        st.markdown("---")
        
        # 4b. Multi-Tab Visualization & Analytics Layout
        tab1, tab2, tab3 = st.tabs(["📈 Combined Curve Plots", "🎯 Extracted Peak Features", "🧬 Peak Value Clustering"])
        
        with tab1:
            st.subheader("Interactive Multi-Curve Visualization")
            fig_curves = go.Figure()
            for name, df in all_data.items():
                fig_curves.add_trace(go.Scatter(x=df['x'], y=df['y'], mode='lines', name=name))
            
            fig_curves.update_layout(
                xaxis_title="Independent Variable (x)",
                yaxis_title="Response Intensity (y)",
                legend_title="Datasets",
                template="plotly_white",
                height=550
            )
            st.plotly_chart(fig_curves, use_container_width=True)
            
            # Show a data preview sample from the first dataset
            st.subheader("Data Table Sample Preview")
            first_key = list(all_data.keys())[0]
            st.write(f"Displaying top 5 rows sample from: `{first_key}`")
            st.dataframe(all_data[first_key].head(), use_container_width=True)
            
        with tab2:
            st.subheader("Extracted Peak Values Matrix")
            st.markdown("The global maximum value pair $(x, y)$ is parsed and indexed for each unique file:")
            st.dataframe(df_peaks, use_container_width=True)
            
        with tab3:
            st.subheader("K-Means Peak Clustering Groupings")
            
            if len(df_peaks) >= n_clusters:
                # Select features based on sidebar choice
                if cluster_features == "Peak Y-Value Only":
                    X_feat = df_peaks[['Peak_Y']].copy()
                else:
                    X_feat = df_peaks[['Peak_X', 'Peak_Y']].copy()
                
                # Scale features for numerical convergence stability
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_feat)
                
                # Run K-Means Clustering Model
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df_peaks['Cluster_Label'] = kmeans.fit_predict(X_scaled)
                df_peaks['Cluster_Label'] = df_peaks['Cluster_Label'].apply(lambda l: f"Cluster {l}")
                
                # Plotly Scatter Plot for Clusters
                fig_cluster = px.scatter(
                    df_peaks,
                    x="Peak_X" if cluster_features != "Peak Y-Value Only" else "Filename",
                    y="Peak_Y",
                    color="Cluster_Label",
                    hover_data=["Filename", "Peak_X", "Peak_Y"],
                    title="Cluster Classification Map of Peaks",
                    labels={"Cluster_Label": "Assigned Cluster"},
                    size_max=12
                )
                fig_cluster.update_traces(marker=dict(size=14, line=dict(width=1.5, color='DarkSlateGrey')))
                fig_cluster.update_layout(template="plotly_white", height=500)
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                # Show organized Clustered Results Table
                st.subheader("Clustered Dataset Allocation Summary")
                st.dataframe(df_peaks.sort_values(by="Cluster_Label"), use_container_width=True)
                
                # Export and Download Section
                st.subheader("💾 Export Analysis Results")
                csv_buffer = io.StringIO()
                df_peaks.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Peak Clusters Report as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="peaks_clustering_report.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"⚠️ Insufficient datasets available to perform cluster groupings. Please upload at least {n_clusters} files or decrease the cluster slider value in the sidebar.")
    else:
        st.error("❌ Unable to parse any data from the uploaded files. Ensure files contain numeric columns.")
else:
    st.info("💡 Welcome! Please upload one or multiple curve response dataset files (.txt or .csv) from the sidebar file management panel to start the automated curve peak analysis and clustering dashboard.")
