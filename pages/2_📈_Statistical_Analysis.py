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

# Fix imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.analysis.statistical import EducationalStatisticsAnalyzer
except ImportError:
    st.error("Could not import EducationalStatisticsAnalyzer. Please check your project structure.")
    st.stop()

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
    
    elif analysis_type == "Multi-Factor ANOVA":
        st.header("Multi-Factor ANOVA Analysis")
        
        # Factor selection
        col1, col2 = st.columns(2)
        
        with col1:
            factors = st.multiselect(
                "Select Factors",
                ['grade_level', 'learning_style', 'prior_minecraft_experience'],
                default=['grade_level', 'learning_style']
            )
        
        with col2:
            dependent_var = st.selectbox(
                "Select Dependent Variable",
                ['engagement_score', 'quest_completion_rate', 'learning_gain']
            )
        
        if st.button("Run ANOVA", type="primary") and len(factors) >= 1:
            # Run ANOVA
            results = analyzer.analyze_multi_factor_anova(
                full_data, dependent_var, factors
            )
            
            # ANOVA table
            st.markdown("### ANOVA Results Table")
            anova_df = pd.DataFrame(results['anova_table'])
            
            # Highlight significant results
            def highlight_significant(val):
                if isinstance(val, float) and val < 0.05:
                    return 'background-color: yellow'
                return ''
            
            styled_anova = anova_df.style.applymap(
                highlight_significant, subset=['p-unc']
            ).format({
                'p-unc': '{:.4f}',
                'eta_squared': '{:.3f}',
                'F': '{:.2f}'
            })
            
            st.dataframe(styled_anova)
            
            # Effect size interpretation
            st.markdown("### Effect Size Interpretation")
            for source, interp in results['interpretation'].items():
                if interp['significance'] == 'significant':
                    st.success(f"**{source}**: {interp['significance']} (p={interp['p_value']:.4f}), "
                             f"{interp['effect_size']} effect (η²={interp['eta_squared']:.3f})")
                else:
                    st.info(f"**{source}**: {interp['significance']} (p={interp['p_value']:.4f})")
            
            # Post-hoc tests
            if results['post_hoc']:
                st.markdown("### Post-hoc Tests (Bonferroni Corrected)")
                for factor, tests in results['post_hoc'].items():
                    st.subheader(f"Pairwise Comparisons for {factor}")
                    posthoc_df = pd.DataFrame(tests)
                    st.dataframe(posthoc_df[['A', 'B', 'diff', 'p-corr', 'hedges']].round(4))
    
    elif analysis_type == "Correlation Analysis":
        st.header("Correlation Analysis")
        
        # Select variables for correlation
        numeric_cols = ['engagement_score', 'quest_completion_rate', 'avg_attempts_per_quest',
                       'building_complexity_avg', 'total_blocks_placed', 'collaboration_events',
                       'days_active', 'skill_progression', 'stem_interest_pre', 'stem_interest_post']
        
        selected_vars = st.multiselect(
            "Select Variables for Correlation Matrix",
            numeric_cols,
            default=numeric_cols[:6]
        )
        
        if len(selected_vars) >= 2:
            # Calculate correlations
            corr_matrix = full_data[selected_vars].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=selected_vars,
                y=selected_vars,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Correlation Heatmap"
            )
            
            # Add correlation values
            for i in range(len(selected_vars)):
                for j in range(len(selected_vars)):
                    fig.add_annotation(
                        x=i, y=j,
                        text=f"{corr_matrix.iloc[j, i]:.2f}",
                        showarrow=False,
                        font=dict(size=10)
                    )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Significant correlations
            st.markdown("### Significant Correlations (|r| > 0.3)")
            
            significant_corrs = []
            for i in range(len(selected_vars)):
                for j in range(i+1, len(selected_vars)):
                    r = corr_matrix.iloc[i, j]
                    if abs(r) > 0.3:
                        significant_corrs.append({
                            'Variable 1': selected_vars[i],
                            'Variable 2': selected_vars[j],
                            'Correlation': r,
                            'Strength': 'Strong' if abs(r) > 0.7 else 'Moderate'
                        })
            
            if significant_corrs:
                st.dataframe(pd.DataFrame(significant_corrs).round(3))
            else:
                st.info("No correlations above |r| = 0.3 found")
    
    else:  # Effect Size Calculator
        st.header("Effect Size Calculator")
        
        st.markdown("""
        Calculate and interpret effect sizes for your analyses. This tool helps you understand
        the practical significance of your findings beyond p-values.
        """)
        
        calc_type = st.radio(
            "Select Calculation Type",
            ["Cohen's d (Two Groups)", "Eta Squared (ANOVA)", "Correlation to Cohen's d"]
        )
        
        if calc_type == "Cohen's d (Two Groups)":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Group 1")
                mean1 = st.number_input("Mean", value=0.0, key="mean1")
                sd1 = st.number_input("Standard Deviation", value=1.0, min_value=0.001, key="sd1")
                n1 = st.number_input("Sample Size", value=30, min_value=2, key="n1")
            
            with col2:
                st.subheader("Group 2")
                mean2 = st.number_input("Mean", value=0.5, key="mean2")
                sd2 = st.number_input("Standard Deviation", value=1.0, min_value=0.001, key="sd2")
                n2 = st.number_input("Sample Size", value=30, min_value=2, key="n2")
            
            # Calculate Cohen's d
            pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2))
            cohens_d = (mean1 - mean2) / pooled_sd
            
            # Display results
            st.markdown("---")
            st.metric("Cohen's d", f"{cohens_d:.3f}")
            
            # Interpretation
            if abs(cohens_d) < 0.2:
                interpretation = "Negligible effect"
                color = "gray"
            elif abs(cohens_d) < 0.5:
                interpretation = "Small effect"
                color = "yellow"
            elif abs(cohens_d) < 0.8:
                interpretation = "Medium effect"
                color = "orange"
            else:
                interpretation = "Large effect"
                color = "red"
            
            st.markdown(f"**Interpretation:** <span style='color: {color}'>{interpretation}</span>", 
                       unsafe_allow_html=True)
            
            # Visual representation
            fig = go.Figure()
            
            # Add normal distributions
            x = np.linspace(-4, 4, 100)
            y1 = stats.norm.pdf(x, 0, 1)
            y2 = stats.norm.pdf(x, cohens_d, 1)
            
            fig.add_trace(go.Scatter(x=x, y=y1, name="Group 1", fill='tozeroy', opacity=0.5))
            fig.add_trace(go.Scatter(x=x, y=y2, name="Group 2", fill='tozeroy', opacity=0.5))
            
            fig.update_layout(
                title="Effect Size Visualization",
                xaxis_title="Standardized Score",
                yaxis_title="Density",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ Please generate data first using the sidebar!")
