import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

def initialize_and_train_model(matches_df: pd.DataFrame, odds_df: pd.DataFrame, wrapper, test_size=0.2):
    """Initialize and train the betting model with historical data"""
    logger = logging.getLogger(__name__)
    
    # Sort data chronologically
    matches_df = matches_df.sort_values('date')
    logger.info(f"Processing {len(matches_df)} matches")
    
    features = []
    labels = []
    valid_matches = 0
    

    for idx, match in matches_df.iterrows():
        try:
            match_date = pd.to_datetime(match['date'])
            
            # Get features for both teams
            home_features = wrapper._calculate_advanced_team_stats(match['home_team'], match_date)
            away_features = wrapper._calculate_advanced_team_stats(match['away_team'], match_date)
            
            if np.any(home_features) and np.any(away_features):
                # Combine home and away features
                combined_features = np.concatenate([home_features, away_features])
                features.append(combined_features)
                
                # Determine match result
                home_score = float(match.get('home_score', match.get('score_home', match.get('home_goals', 0))))
                away_score = float(match.get('away_score', match.get('score_away', match.get('away_goals', 0))))
                
                if home_score > away_score:
                    result = 0
                elif home_score == away_score:
                    result = 1
                else:
                    result = 2
                    
                labels.append(result)
                valid_matches += 1
                
        except Exception as e:
            
            logger.warning(f"Error processing match {idx}: {str(e)}")
            continue
    
    
    if valid_matches == 0:
        raise ValueError("No valid features were generated. Check your data and feature calculation logic.")
        
    logger.info(f"Successfully processed {valid_matches} matches")
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Split and process data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    # Scale features
    X_train_scaled = wrapper.scaler.fit_transform(X_train)
    X_test_scaled = wrapper.scaler.transform(X_test)
    
    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=3)
    
    # Build and train model
    input_shape = X_train.shape[1]
    wrapper.model = wrapper.build_enhanced_model(input_shape)
    wrapper.train_enhanced_model(X_train_scaled, y_train_cat, validation_split=0.2)
    
    return wrapper

# Modified main execution
if __name__ == "__main__":
    # Load data
    matches_df = pd.read_csv('epl_matches_2024_25.csv')
    odds_df = pd.read_csv('epl_odds_2024_25.csv')
    
    # Convert dates
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    odds_df['date'] = pd.to_datetime(odds_df['date'])
    
    # Initialize algorithm with historical data
    from base_algo import BaseAlgorithm
    algo = BaseAlgorithm(matches_df)
    
    # Initialize neural wrapper
    from betting_model import AdvancedBettingNeuralWrapper
    wrapper = AdvancedBettingNeuralWrapper(algo)
    
    # Initialize and train the model
    wrapper = initialize_and_train_model(matches_df, odds_df, wrapper)
    
    # Create backtester with trained model
    backtester = EPLBacktester(
        neural_wrapper=wrapper,
        initial_capital=10000.0,
        bet_size_pct=0.02,
        min_confidence=0.4,
        stop_loss_pct=0.2
    )
    
    # Run backtest
    results = backtester.run_backtest(matches_df, odds_df)
    
    # Print results
    print("\nBacktest Results:")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")