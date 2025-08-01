## pages/4_🎮_3D_World_View.py

# For all page files (1_📊_Overview.py, 2_📈_Statistical_Analysis.py, etc.)
# Replace the import section with this pattern:

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now you can import from src
from src.analysis.statistical import EducationalStatisticsAnalyzer
from src.analysis.time_series import TimeSeriesEducationAnalyzer
from src.visualization.plots import EducationalVisualizer
from src.utils.helpers import DataProcessor, StreamlitHelpers, AnalyticsHelpers

st.set_page_config(page_title="3D World View", page_icon="🎮", layout="wide")

st.title("🎮 3D World Visualization")
st.markdown("### Explore student movements and interactions in the Minecraft world")

if 'datasets' in st.session_state and st.session_state.datasets:
    datasets = st.session_state.datasets
    
    # Visualization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Movement Heatmap", "Zone Activity", "Collaboration Network", "Building Clusters"]
        )
    
    with col2:
        time_filter = st.slider(
            "Time Range (Days)",
            1, 30, (1, 7)
        )
    
    with col3:
        student_filter = st.selectbox(
            "Filter by Student",
            ["All Students"] + list(datasets['students']['student_id'].unique())
        )
    
    st.markdown("---")
    
    # Filter data based on selections
    movements = datasets['movements'].copy()
    movements['date'] = pd.to_datetime(movements['timestamp']).dt.date
    date_range = pd.date_range(
        movements['date'].min(), 
        movements['date'].min() + pd.Timedelta(days=30)
    )[time_filter[0]-1:time_filter[1]]
    
    filtered_movements = movements[movements['date'].isin(date_range)]
    if student_filter != "All Students":
        filtered_movements = filtered_movements[filtered_movements['student_id'] == student_filter]
    
    if viz_type == "Movement Heatmap":
        st.header("3D Movement Heatmap")
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add movement traces
        for zone in filtered_movements['zone'].unique():
            zone_data = filtered_movements[filtered_movements['zone'] == zone]
            
            fig.add_trace(go.Scatter3d(
                x=zone_data['x'],
                y=zone_data['z'],
                z=zone_data['y'],
                mode='markers',
                name=zone,
                marker=dict(
                    size=3,
                    opacity=0.6,
                    color=zone_data.index,
                    colorscale='Viridis'
                )
            ))
        
        # Add zone boundaries
        zones_config = {
            'spawn': {'x': 0, 'z': 0, 'radius': 30, 'color': 'red'},
            'tutorial': {'x': 50, 'z': 0, 'radius': 40, 'color': 'blue'},
            'building_area': {'x': 100, 'z': 100, 'radius': 100, 'color': 'green'},
            'collaboration_space': {'x': -100, 'z': 100, 'radius': 80, 'color': 'purple'},
            'challenge_zone': {'x': 0, 'z': 200, 'radius': 60, 'color': 'orange'}
        }
        
        # Add zone circles
        for zone, config in zones_config.items():
            theta = np.linspace(0, 2*np.pi, 50)
            x_circle = config['x'] + config['radius'] * np.cos(theta)
            z_circle = config['z'] + config['radius'] * np.sin(theta)
            y_circle = np.ones_like(theta) * 65
            
            fig.add_trace(go.Scatter3d(
                x=x_circle,
                y=z_circle,
                z=y_circle,
                mode='lines',
                name=f"{zone} boundary",
                line=dict(color=config['color'], width=3),
                showlegend=False
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Z Coordinate",
                zaxis_title="Y Coordinate",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700,
            title="Student Movement Patterns in 3D Space"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Movement statistics
        st.markdown("### Movement Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_movements = len(filtered_movements)
            st.metric("Total Movements", f"{total_movements:,}")
        
        with col2:
            unique_zones = filtered_movements['zone'].nunique()
            st.metric("Zones Visited", unique_zones)
        
        with col3:
            avg_distance = np.sqrt(
                filtered_movements.groupby('session_id')[['x', 'z']].diff().pow(2).sum(axis=1)
            ).mean()
            st.metric("Avg Distance/Session", f"{avg_distance:.1f} blocks")
    
    elif viz_type == "Zone Activity":
        st.header("Zone Activity Analysis")
        
        # Calculate zone statistics
        zone_stats = filtered_movements.groupby('zone').agg({
            'student_id': 'nunique',
            'timestamp': 'count'
        }).rename(columns={'student_id': 'unique_students', 'timestamp': 'total_visits'})
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            labels=['World'] + zone_stats.index.tolist(),
            parents=[''] + ['World'] * len(zone_stats),
            values=[zone_stats['total_visits'].sum()] + zone_stats['total_visits'].tolist(),
            branchvalues="total"
        ))
        
        fig.update_layout(
            title="Zone Activity Distribution",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time spent in each zone
        time_in_zones = filtered_movements.groupby(['student_id', 'zone']).size().reset_index(name='time_units')
        
        fig_box = px.box(
            time_in_zones,
            x='zone',
            y='time_units',
            title="Time Distribution Across Zones"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    elif viz_type == "Collaboration Network":
        st.header("Student Collaboration Network")
        
        # Filter collaborations
        collabs = datasets['collaborations'].copy()
        collabs['date'] = pd.to_datetime(collabs['timestamp']).dt.date
        filtered_collabs = collabs[collabs['date'].isin(date_range)]
        
        if len(filtered_collabs) > 0:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (students)
            students_in_collabs = set(filtered_collabs['student_1'].unique()) | set(filtered_collabs['student_2'].unique())
            G.add_nodes_from(students_in_collabs)
            
            # Add edges (collaborations)
            edge_weights = {}
            for _, collab in filtered_collabs.iterrows():
                edge = tuple(sorted([collab['student_1'], collab['student_2']]))
                edge_weights[edge] = edge_weights.get(edge, 0) + 1
            
            for edge, weight in edge_weights.items():
                G.add_edge(edge[0], edge[1], weight=weight)
            
            # Calculate layout
            pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
            
            # Create 3D network visualization
            edge_trace = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = G[edge[0]][edge[1]]['weight']
                
                edge_trace.append(go.Scatter3d(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[0, 0, None],
                    mode='lines',
                    line=dict(width=weight, color='gray'),
                    hoverinfo='none',
                    showlegend=False
                ))
            
            # Node trace
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_z = [0 for _ in G.nodes()]
            
            # Calculate node sizes based on degree
            node_sizes = [20 + 10 * G.degree(node) for node in G.nodes()]
            
            node_trace = go.Scatter3d(
                x=node_x,
                y=node_y,
                z=node_z,
                mode='markers+text',
                text=[node for node in G.nodes()],
                textposition="top center",
                marker=dict(
                    size=node_sizes,
                    color=node_sizes,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Connections")
                ),
                hovertemplate='Student: %{text}<br>Connections: %{marker.color}<extra></extra>'
            )
            
            fig = go.Figure(data=edge_trace + [node_trace])
            
            fig.update_layout(
                title="3D Collaboration Network",
                showlegend=False,
                scene=dict(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ),
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Network metrics
            st.markdown("### Network Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                st.metric("Avg Connections", f"{avg_degree:.1f}")
            
            with col2:
                if len(G.nodes()) > 1:
                    density = nx.density(G)
                    st.metric("Network Density", f"{density:.2%}")
                else:
                    st.metric("Network Density", "N/A")
            
            with col3:
                components = nx.number_connected_components(G)
                st.metric("Connected Groups", components)
        
        else:
            st.info("No collaborations found in the selected time range")
    
    else:  # Building Clusters
        st.header("Building Activity Clusters")
        
        # Filter building data
        buildings = datasets['buildings'].copy()
        buildings['date'] = pd.to_datetime(buildings['timestamp']).dt.date
        filtered_buildings = buildings[buildings['date'].isin(date_range)]
        
        if student_filter != "All Students":
            filtered_buildings = filtered_buildings[filtered_buildings['student_id'] == student_filter]
        
        # 3D scatter of building locations
        fig = go.Figure()
        
        # Color by block type
        for block_type in filtered_buildings['block_type'].unique():
            block_data = filtered_buildings[filtered_buildings['block_type'] == block_type]
            
            fig.add_trace(go.Scatter3d(
                x=block_data['x'],
                y=block_data['z'],
                z=block_data['y'],
                mode='markers',
                name=block_type,
                marker=dict(
                    size=5,
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title="X Coordinate",
                yaxis_title="Z Coordinate", 
                zaxis_title="Y Coordinate"
            ),
            height=700,
            title="3D Building Distribution by Block Type"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Building statistics
        st.markdown("### Building Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            block_counts = filtered_buildings['block_type'].value_counts()
            fig_pie = px.pie(
                values=block_counts.values,
                names=block_counts.index,
                title="Block Type Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            complexity_by_student = filtered_buildings.groupby('student_id')['complexity_score'].mean().sort_values(ascending=False).head(10)
            fig_bar = px.bar(
                x=complexity_by_student.values,
                y=complexity_by_student.index,
                orientation='h',
                title="Top 10 Students by Building Complexity"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.warning("⚠️ Please generate data first using the sidebar!")

