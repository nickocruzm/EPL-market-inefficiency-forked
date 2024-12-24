# main.py
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
import os

# Import our modules
from model_initialization import initialize_and_train_model
from backtest import EPLBacktester
from betting_model import AdvancedBettingNeuralWrapper
from base_algo import BaseAlgorithm
from analysis import analyze_backtest_results

def setup_logging() -> logging.Logger:
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_path = log_dir / f'backtest_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('epl_backtest')

def load_and_prepare_data(matches_path: str, odds_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare the datasets
    
    Args:
        matches_path: Path to matches CSV file
        odds_path: Path to odds CSV file
        
    Returns:
        Tuple of (matches_df, odds_df)
    """
    # Ensure data directory exists
    data_dir = Path('data')
    matches_full_path = data_dir / matches_path
    odds_full_path = data_dir / odds_path
    
    # Check if files exist
    if not matches_full_path.exists():
        raise FileNotFoundError(f"Matches file not found: {matches_full_path}")
    if not odds_full_path.exists():
        raise FileNotFoundError(f"Odds file not found: {odds_full_path}")
    
    # Load data
    try:
        matches_df = pd.read_csv(matches_full_path)
        odds_df = pd.read_csv(odds_full_path)
    except Exception as e:
        raise Exception(f"Error loading data files: {str(e)}")
    
    # Validate required columns
    required_match_cols = ['date', 'home_team', 'away_team', 'result', 'match_id']
    required_odds_cols = ['date', 'match_id', 'home_win_odds', 'draw_odds', 'away_win_odds']
    
    missing_match_cols = [col for col in required_match_cols if col not in matches_df.columns]
    missing_odds_cols = [col for col in required_odds_cols if col not in odds_df.columns]
    
    if missing_match_cols:
        raise ValueError(f"Missing required columns in matches file: {missing_match_cols}")
    if missing_odds_cols:
        raise ValueError(f"Missing required columns in odds file: {missing_odds_cols}")
    
    # Convert dates
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    odds_df['date'] = pd.to_datetime(odds_df['date'])
    
    # Sort chronologically
    matches_df = matches_df.sort_values('date')
    odds_df = odds_df.sort_values('date')
    
    # Ensure match_ids align
    if not set(matches_df['match_id']).intersection(odds_df['match_id']):
        raise ValueError("No matching match_ids between matches and odds data")
    
    return matches_df, odds_df

def validate_model_output(wrapper, sample_data):
    """Validate model output format and ranges"""
    try:
        # Ensure sample data has correct shape
        if sample_data.shape[1] == 20:
            # Duplicate features for home/away teams
            sample_data = np.concatenate([sample_data, sample_data], axis=1)
            
        pred_class, confidence, probabilities = wrapper.predict_with_confidence(sample_data)
        
        # Validate prediction class
        if not isinstance(pred_class, (int, np.integer)) or pred_class not in [0, 1, 2]:
            raise ValueError(f"Invalid prediction class: {pred_class}")
            
        # Validate confidence score
        if not isinstance(confidence, (float, np.floating)) or confidence < 0 or confidence > 1:
            raise ValueError(f"Invalid confidence score: {confidence}")
            
        # Validate probability distribution
        if len(probabilities) != 3 or not np.allclose(sum(probabilities), 1.0):
            raise ValueError(f"Invalid probability distribution: {probabilities}")
            
        return True
        
    except Exception as e:
        raise Exception(f"Model validation failed: {str(e)}")

def save_results(results: Dict, output_dir: str = 'results'):
    """Save backtest results to file"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(output_path / f'backtest_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)

def main():
    """Main execution function"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting EPL backtesting system")
        
        # Load data
        matches_df, odds_df = load_and_prepare_data(
            'epl_matches_2024_25.csv',
            'epl_odds_2024_25.csv'
        )
        logger.info(f"Loaded {len(matches_df)} matches and {len(odds_df)} odds records")
        
        # Initialize algorithm
        algo = BaseAlgorithm(matches_df)
        logger.info("Initialized base algorithm")
        
        # Create neural wrapper
        wrapper = AdvancedBettingNeuralWrapper(
            base_algorithm=algo,
            initial_bankroll=10000,
            min_confidence=0.6,
            max_bet_size=0.1
        )
        logger.info("Created neural wrapper")
        
        # Initialize and train model
        wrapper = initialize_and_train_model(
            matches_df=matches_df,
            odds_df=odds_df,
            wrapper=wrapper,
            test_size=0.2
        )
        logger.info("Model training completed")
        
        # Validate model output
        sample_features = wrapper._calculate_advanced_team_stats("Arsenal", pd.Timestamp("2023-01-01"))
        if validate_model_output(wrapper, sample_features.reshape(1, -1)):
            logger.info("Model validation passed")
        
        # Initialize backtester
        backtester = EPLBacktester(
            betting_model=wrapper,
            initial_bankroll=10000,
        )
        logger.info("Backtester initialized")
        
        # Run backtest
        logger.info("Starting backtest...")
        results = backtester.run_backtest(matches_df, odds_df)
        
        # Analyze and display results
        analyze_backtest_results(results)
        
    except Exception as e:
        logger.error(f"Error during backtest execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()