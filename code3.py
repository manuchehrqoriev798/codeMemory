"""
Data Processing and Visualization Module

This module provides classes and functions for processing, analyzing,
and visualizing data from various sources.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple
import json
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Base class for data processing operations."""
    
    def __init__(self, data_source: Union[str, pd.DataFrame]):
        """Initialize with a data source (file path or DataFrame)."""
        self.raw_data = None
        self.processed_data = None
        
        if isinstance(data_source, str):
            self.load_from_file(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.raw_data = data_source.copy()
        else:
            raise TypeError("data_source must be a file path or pandas DataFrame")
        
        logger.info("DataProcessor initialized with %s", 
                   "DataFrame" if isinstance(data_source, pd.DataFrame) else data_source)
    
    def load_from_file(self, file_path: str) -> None:
        """Load data from a file based on its extension."""
        if file_path.endswith('.csv'):
            self.raw_data = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            self.raw_data = pd.read_json(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.raw_data = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        logger.info("Loaded data from %s", file_path)
    
    def clean_data(self) -> pd.DataFrame:
        """Basic data cleaning operations."""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_from_file first.")
        
        # Make a copy to avoid modifying the original
        df = self.raw_data.copy()
        
        # Drop rows with any NaN values
        df = df.dropna()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        self.processed_data = df
        logger.info("Data cleaned: %d rows remaining", len(df))
        
        return df
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for numerical columns."""
        if self.processed_data is None:
            if self.raw_data is None:
                raise ValueError("No data loaded.")
            data = self.raw_data
        else:
            data = self.processed_data
        
        numeric_data = data.select_dtypes(include=[np.number])
        stats = {}
        
        for column in numeric_data.columns:
            stats[column] = {
                "mean": numeric_data[column].mean(),
                "median": numeric_data[column].median(),
                "std": numeric_data[column].std(),
                "min": numeric_data[column].min(),
                "max": numeric_data[column].max()
            }
        
        return stats
    
    def save_processed_data(self, output_path: str) -> None:
        """Save processed data to a file."""
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data first.")
        
        if output_path.endswith('.csv'):
            self.processed_data.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            self.processed_data.to_json(output_path, orient='records')
        elif output_path.endswith('.xlsx'):
            self.processed_data.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path}")
        
        logger.info("Saved processed data to %s", output_path)

class DataVisualizer:
    """Class for creating data visualizations."""
    
    def __init__(self, data: pd.DataFrame):
        """Initialize with a pandas DataFrame."""
        self.data = data
        self.fig = None
        self.ax = None
    
    def create_histogram(self, column: str, bins: int = 10, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a histogram of a numerical column."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.hist(self.data[column], bins=bins)
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(f'Histogram of {column}')
        
        self.ax.set_xlabel(column)
        self.ax.set_ylabel('Frequency')
        
        return self.fig, self.ax
    
    def create_scatter_plot(self, x_column: str, y_column: str, 
                           color_column: Optional[str] = None, 
                           title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a scatter plot of two numerical columns."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        if color_column:
            scatter = self.ax.scatter(self.data[x_column], self.data[y_column], 
                                     c=self.data[color_column], cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, ax=self.ax, label=color_column)
        else:
            self.ax.scatter(self.data[x_column], self.data[y_column], alpha=0.7)
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(f'{y_column} vs {x_column}')
        
        self.ax.set_xlabel(x_column)
        self.ax.set_ylabel(y_column)
        
        return self.fig, self.ax
    
    def create_bar_chart(self, x_column: str, y_column: str, 
                        title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a bar chart comparing categories."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        self.ax.bar(self.data[x_column], self.data[y_column])
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(f'{y_column} by {x_column}')
        
        self.ax.set_xlabel(x_column)
        self.ax.set_ylabel(y_column)
        
        plt.xticks(rotation=45, ha='right')
        
        return self.fig, self.ax
    
    def create_line_plot(self, x_column: str, y_columns: List[str], 
                        title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """Create a line plot over time or sequence."""
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        
        for column in y_columns:
            self.ax.plot(self.data[x_column], self.data[column], label=column)
        
        if title:
            self.ax.set_title(title)
        else:
            self.ax.set_title(f'Trends over {x_column}')
        
        self.ax.set_xlabel(x_column)
        self.ax.set_ylabel('Value')
        self.ax.legend()
        
        return self.fig, self.ax
    
    def save_figure(self, output_path: str, dpi: int = 300) -> None:
        """Save the current figure to a file."""
        if self.fig is None:
            raise ValueError("No figure created yet. Call a visualization method first.")
        
        self.fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info("Saved figure to %s", output_path)

def load_and_process_csv(file_path: str) -> pd.DataFrame:
    """Utility function to load and clean a CSV file."""
    processor = DataProcessor(file_path)
    cleaned_data = processor.clean_data()
    return cleaned_data

def generate_summary_report(data: pd.DataFrame, output_path: str) -> None:
    """Generate a comprehensive report with statistics and basic visualizations."""
    processor = DataProcessor(data)
    stats = processor.get_summary_statistics()
    
    # Save statistics to JSON
    with open(output_path + "_stats.json", 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Create visualizations for each numerical column
    visualizer = DataVisualizer(data)
    for column in data.select_dtypes(include=[np.number]).columns:
        visualizer.create_histogram(column)
        visualizer.save_figure(output_path + f"_hist_{column}.png")
    
    logger.info("Generated summary report at %s", output_path)

def main():
    """Demonstrate the use of data processing and visualization functions."""
    # Create sample data
    np.random.seed(42)
    data = {
        'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'metric1': np.random.randint(0, 100, 100),
        'metric2': np.random.exponential(5, 100)
    }
    df = pd.DataFrame(data)
    
    # Process the data
    processor = DataProcessor(df)
    cleaned_data = processor.clean_data()
    stats = processor.get_summary_statistics()
    print("Data Summary:")
    print(json.dumps(stats, indent=2))
    
    # Create visualizations
    visualizer = DataVisualizer(cleaned_data)
    
    # Histogram
    visualizer.create_histogram('value')
    visualizer.save_figure('value_histogram.png')
    
    # Scatter plot
    visualizer.create_scatter_plot('metric1', 'metric2', color_column='value')
    visualizer.save_figure('metrics_scatter.png')
    
    # Line plot
    visualizer.create_line_plot('date', ['value', 'metric1'])
    visualizer.save_figure('trends_line.png')
    
    print("Visualizations saved.")

if __name__ == "__main__":
    main() 