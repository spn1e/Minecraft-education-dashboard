## pages/2_📈_Statistical_Analysis.py
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

# Now import from src
from src.analysis.statistical import EducationalStatisticsAnalyzer

st.set_page_config(page_title="Statistical Analysis", page_icon="📈", layout="wide")

st.title("📈 Statistical Analysis Suite")
st.markdown("### Comprehensive statistical testing for educational data")

if 'datasets' in st.session_state and st.session_state.datasets:
    datasets = st.session_state.datasets
    analyzer = EducationalStatisticsAnalyzer()
    
    # Prepare merged dataset
    full_data = datasets['students'].merge(
        datasets['learning_analytics'], on='student_id'
    )
    
    # Analysis selector
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["T-Test Comparison", "Multi-Factor ANOVA", "Correlation Analysis", "Effect Size Calculator"]
    )
    
    st.markdown("---")
    
    if analysis_type == "T-Test Comparison":
        st.header("Independent Samples T-Test")
        
        col1, col2 = st.columns(2)
        
        with col1:
            group_var = st.selectbox(
                "Select Grouping Variable",
                ['prior_minecraft_experience', 'collaboration_preference', 'learning_style']
            )
            
            # Create binary groups
            if group_var == 'prior_minecraft_experience':
                full_data['group_binary'] = full_data[group_var].apply(
                    lambda x: 'experienced' if x in ['intermediate', 'advanced'] else 'novice'
                )
            elif group_var == 'collaboration_preference':
                full_data['group_binary'] = full_data[group_var].apply(
                    lambda x: 'collaborative' if x in ['pairs', 'groups'] else 'solo'
                )
            else:
                # For learning style, compare visual vs others
                full_data['group_binary'] = full_data[group_var].apply(
                    lambda x: 'visual' if x == 'visual' else 'other'
                )
        
        with col2:
            outcome_var = st.selectbox(
                "Select Outcome Variable",
                ['engagement_score', 'quest_completion_rate', 'learning_gain', 'skill_progression']
            )
        
        if st.button("Run T-Test Analysis", type="primary"):
            # Run analysis
            results = analyzer.compare_learning_methods(
                full_data, 'group_binary', outcome_var
            )
            
            # Display results
            st.markdown("### Results")
            
            # Create results visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Group Distributions", "Effect Size Visualization")
            )
            
            # Box plot
            for i, (group_name, group_data) in enumerate(results['groups'].items()):
                y_values = full_data[full_data['group_binary'] == group_name][outcome_var]
                fig.add_trace(
                    go.Box(y=y_values, name=group_name, boxpoints='outliers'),
                    row=1, col=1
                )
            
            # Effect size visualization
            effect_size = results['effect_size']
            fig.add_trace(
                go.Bar(
                    x=['Effect Size'],
                    y=[abs(effect_size)],
                    text=[f"d = {effect_size:.3f}"],
                    textposition='outside',
                    marker_color='lightblue' if effect_size > 0 else 'lightcoral'
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Test Statistic", f"{results['test_statistic']:.3f}")
                st.metric("P-Value", f"{results['p_value']:.4f}")
            
            with col2:
                st.metric("Effect Size (Cohen's d)", f"{results['effect_size']:.3f}")
                st.info(f"**{results['interpretation']}**")
            
            with col3:
                st.metric("Statistical Power", f"{results['statistical_power']:.2%}")
                significance = "✅ Significant" if results['significant'] else "❌ Not Significant"
                st.success(significance) if results['significant'] else st.warning(significance)
            
            # Assumptions testing
            with st.expander("View Statistical Assumptions"):
                st.write("**Normality Tests (Shapiro-Wilk):**")
                for group, data in results['groups'].items():
                    norm_result = "✅ Normal" if data['normality_p'] > 0.05 else "⚠️ Not Normal"
                    st.write(f"- {group}: p = {data['normality_p']:.4f} {norm_result}")
                
                st.write(f"\n**Homogeneity of Variances (Levene's Test):**")
                levene_result = "✅ Equal variances" if results['levene_test']['p_value'] > 0.05 else "⚠️ Unequal variances"
                st.write(f"p = {results['levene_test']['p_value']:.4f} {levene_result}")
                
                st.write(f"\n**Test Used:** {results['test_type']}")

else:
    st.warning("⚠️ Please generate data first using the sidebar!")
