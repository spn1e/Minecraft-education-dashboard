# 🛠️ STREAMLIT CLOUD FIXES - For Your Minecraft Dashboard

# ========================================
# 🔧 FIX 1: ROBUST APP.PY WITH FALLBACKS
# ========================================

# Replace your app.py with this version that handles missing packages gracefully:

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path with error handling
try:
    sys.path.append(str(Path(__file__).parent))
    from src.data_generation.simulator import MinecraftEducationSimulator, create_config
    SIMULATOR_AVAILABLE = True
except ImportError as e:
    st.warning(f"⚠️ Advanced simulation not available: {e}")
    SIMULATOR_AVAILABLE = False

# Optional imports with fallbacks
try:
    from src.analysis.statistical import EducationalStatisticsAnalyzer
    STATISTICAL_ANALYSIS = True
except ImportError:
    STATISTICAL_ANALYSIS = False

try:
    from src.analysis.time_series import TimeSeriesEducationAnalyzer
    TIME_SERIES_ANALYSIS = True
except ImportError:
    TIME_SERIES_ANALYSIS = False

# Page configuration
st.set_page_config(
    page_title="Minecraft Education Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# 🔧 FIX 2: FALLBACK DATA GENERATOR
# ========================================

@st.cache_data
def generate_fallback_data(n_students=30, n_days=14):
    """Generate simple demo data if main simulator fails"""
    np.random.seed(42)
    
    # Simple student data
    students = pd.DataFrame({
        'student_id': [f'STU_{i:03d}' for i in range(n_students)],
        'grade_level': np.random.choice([6, 7, 8], n_students),
        'learning_style': np.random.choice(['visual', 'kinesthetic', 'auditory'], n_students),
        'collaboration_preference': np.random.choice(['solo', 'pairs', 'groups'], n_students),
        'prior_minecraft_experience': np.random.choice(['none', 'beginner', 'intermediate', 'advanced'], n_students),
        'stem_interest_pre': np.random.randint(1, 6, n_students)
    })
    
    # Simple analytics data
    learning_analytics = pd.DataFrame({
        'student_id': students['student_id'],
        'engagement_score': np.random.beta(2, 1, n_students),
        'quest_completion_rate': np.random.beta(3, 1, n_students),
        'avg_attempts_per_quest': np.random.gamma(2, 1, n_students) + 1,
        'building_complexity_avg': np.random.lognormal(1, 0.5, n_students),
        'total_blocks_placed': np.random.poisson(100, n_students),
        'collaboration_events': np.random.poisson(5, n_students),
        'days_active': np.random.randint(1, n_days+1, n_students),
        'skill_progression': np.random.beta(2, 2, n_students),
        'stem_interest_post': np.random.randint(2, 6, n_students),
    })
    
    # Calculate learning gain
    learning_analytics['learning_gain'] = (
        learning_analytics['stem_interest_post'] - 
        students['stem_interest_pre'].values
    )
    
    return {
        'students': students,
        'learning_analytics': learning_analytics
    }

# ========================================
# 🔧 FIX 3: STREAMLIT CLOUD COMPATIBLE CONFIG HANDLER
# ========================================

def load_config_safe():
    """Load config with fallback for Streamlit Cloud"""
    config_path = Path("config.yaml")
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            st.warning("⚠️ YAML config not available, using defaults")
        except Exception as e:
            st.warning(f"⚠️ Config loading error: {e}")
    
    # Default config
    return {
        'simulation': {'n_students': 60, 'days': 30, 'seed': 42},
        'analytics': {
            'engagement_weights': {
                'quest_completion': 0.3,
                'building_activity': 0.3,
                'collaboration': 0.2,
                'skill_progression': 0.2
            }
        }
    }

# ========================================
# 🔧 FIX 4: STREAMLIT CLOUD MAIN APP
# ========================================

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
    st.session_state.datasets = None

# Load configuration
config = load_config_safe()

# Sidebar
with st.sidebar:
    st.title("🎮 MC Education Analytics")
    st.markdown("---")
    
    # Show available features
    st.subheader("🌟 Available Features")
    if SIMULATOR_AVAILABLE:
        st.success("✅ Advanced Simulation")
    else:
        st.info("📊 Demo Data Mode")
    
    if STATISTICAL_ANALYSIS:
        st.success("✅ Statistical Analysis")
    else:
        st.info("📈 Basic Analysis")
    
    if TIME_SERIES_ANALYSIS:
        st.success("✅ Time Series Analysis")
    else:
        st.info("⏰ Simple Trends")
    
    st.markdown("---")
    
    # Data generation
    st.header("Data Configuration")
    n_students = st.slider("Number of Students", 10, 100, 30)
    n_days = st.slider("Simulation Days", 7, 30, 14)
    
    if st.button("Generate Data", type="primary"):
        with st.spinner("Generating educational data..."):
            try:
                if SIMULATOR_AVAILABLE:
                    # Use advanced simulator
                    simulator = MinecraftEducationSimulator()
                    datasets = simulator.generate_complete_dataset(n_students, n_days)
                else:
                    # Use fallback generator
                    datasets = generate_fallback_data(n_students, n_days)
                
                st.session_state.datasets = datasets
                st.session_state.data_generated = True
                st.success("✅ Data generated successfully!")
                
            except Exception as e:
                st.error(f"Data generation failed: {e}")
                # Try fallback
                datasets = generate_fallback_data(n_students, n_days)
                st.session_state.datasets = datasets
                st.session_state.data_generated = True
                st.info("📊 Using demo data instead")

# Main content
st.title("🎮 Minecraft Education Analytics Dashboard")
st.markdown("### Analyzing Game-Based Learning Patterns & Student Outcomes")

# Add deployment info
if st.checkbox("ℹ️ Show Deployment Info"):
    st.info(f"""
    **Deployment Status:**
    - 🌐 **Hosted on**: Streamlit Cloud
    - 🐍 **Python Version**: {sys.version.split()[0]}
    - 📦 **Streamlit Version**: {st.__version__}
    - 🔧 **Advanced Simulation**: {'Available' if SIMULATOR_AVAILABLE else 'Demo Mode'}
    - 📊 **Statistical Analysis**: {'Full' if STATISTICAL_ANALYSIS else 'Basic'}
    """)

if not st.session_state.data_generated:
    # Welcome screen with cloud-specific messaging
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>🌐 Welcome to the Cloud-Hosted Analytics Platform</h2>
        <p style='font-size: 18px;'>
            This dashboard is running live on Streamlit Cloud, demonstrating 
            advanced analytics for game-based learning environments.
        </p>
        <br>
        <h3>🚀 Available Features</h3>
        <ul style='text-align: left; max-width: 600px; margin: auto;'>
            <li>📊 Interactive Data Generation</li>
            <li>📈 Statistical Analysis Suite</li>
            <li>🤖 Machine Learning Predictions</li>
            <li>🗺️ Advanced Visualizations</li>
            <li>👥 Collaboration Analytics</li>
            <li>🎯 Educational Insights</li>
        </ul>
        <br>
        <p><strong>👆 Generate data in the sidebar to get started!</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    # Load and display data
    datasets = st.session_state.datasets
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(datasets['students']))
    
    with col2:
        avg_engagement = datasets['learning_analytics']['engagement_score'].mean()
        st.metric("Avg Engagement", f"{avg_engagement:.1%}")
    
    with col3:
        completion_rate = datasets['learning_analytics']['quest_completion_rate'].mean()
        st.metric("Quest Completion", f"{completion_rate:.1%}")
    
    with col4:
        learning_gain = datasets['learning_analytics']['learning_gain'].mean()
        st.metric("Avg Learning Gain", f"+{learning_gain:.1f}")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Analysis", "🤖 Predictions"])
    
    with tab1:
        st.header("Student Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement distribution
            fig_engagement = px.histogram(
                datasets['learning_analytics'],
                x='engagement_score',
                nbins=15,
                title="Student Engagement Distribution",
                labels={'engagement_score': 'Engagement Score'}
            )
            st.plotly_chart(fig_engagement, use_container_width=True)
        
        with col2:
            # Learning gain by grade
            students_with_analytics = datasets['students'].merge(
                datasets['learning_analytics'], on='student_id'
            )
            
            fig_grade = px.box(
                students_with_analytics,
                x='grade_level',
                y='learning_gain',
                title="Learning Gain by Grade Level"
            )
            st.plotly_chart(fig_grade, use_container_width=True)
    
    with tab2:
        st.header("Educational Analysis")
        
        if STATISTICAL_ANALYSIS:
            st.success("🔬 Advanced statistical analysis available!")
        else:
            st.info("📊 Showing basic analysis (full stats require additional packages)")
        
        # Basic correlation analysis
        numeric_cols = ['engagement_score', 'quest_completion_rate', 'skill_progression', 'learning_gain']
        corr_matrix = datasets['learning_analytics'][numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix - Learning Metrics",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.header("Predictive Insights")
        
        # Simple prediction using built-in sklearn
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Simple linear model
            X = datasets['learning_analytics'][['engagement_score', 'quest_completion_rate']]
            y = datasets['learning_analytics']['learning_gain']
            
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            
            st.metric("Model R² Score", f"{r2:.3f}")
            
            # Feature importance (coefficients)
            importance_df = pd.DataFrame({
                'feature': ['Engagement Score', 'Quest Completion Rate'],
                'coefficient': model.coef_
            })
            
            fig_importance = px.bar(
                importance_df,
                x='coefficient',
                y='feature',
                orientation='h',
                title="Feature Importance for Learning Gain Prediction"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
        except ImportError:
            st.warning("⚠️ Machine learning features require scikit-learn")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎮 Minecraft Education Analytics Dashboard | Hosted on Streamlit Cloud</p>
    <p>📊 Demonstrates: Python • Data Science • Cloud Deployment • Educational Analytics</p>
</div>
""", unsafe_allow_html=True)

# ========================================
# 🔧 END OF ROBUST APP.PY
# ========================================
