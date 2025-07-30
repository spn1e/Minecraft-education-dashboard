# Consolidated imports for Streamlit Cloud compatibility
# Place this file at the root level (same as app.py)

# Re-export all necessary components
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.data_generation.simulator import MinecraftEducationSimulator, create_config
    from src.analysis.statistical import EducationalStatisticsAnalyzer  
    from src.analysis.time_series import TimeSeriesEducationAnalyzer
    from src.visualization.plots import EducationalVisualizer
    from src.utils.helpers import DataProcessor, StreamlitHelpers, AnalyticsHelpers
except ImportError:
    # Fallback: try direct imports if src structure doesn't work
    from data_generation.simulator import MinecraftEducationSimulator, create_config
    from analysis.statistical import EducationalStatisticsAnalyzer
    from analysis.time_series import TimeSeriesEducationAnalyzer
    from visualization.plots import EducationalVisualizer
    from utils.helpers import DataProcessor, StreamlitHelpers, AnalyticsHelpers

# Make them available for import
__all__ = [
    'MinecraftEducationSimulator',
    'create_config',
    'EducationalStatisticsAnalyzer',
    'TimeSeriesEducationAnalyzer',
    'EducationalVisualizer',
    'DataProcessor',
    'StreamlitHelpers',
    'AnalyticsHelpers'
]
