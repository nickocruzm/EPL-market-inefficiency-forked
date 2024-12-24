import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple

def analyze_backtest_results(results_df: pd.DataFrame) -> None:
    """
    Analyze and display backtest results
    
    Args:
        results_df: DataFrame containing backtest results
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Check the structure of the results Dat     aFrame
        print(results_df) #previously results_df.head() 
        
        # Ensure 'bankroll' column exists
        if 'bankroll' not in results_df.columns:
            raise ValueError("Results DataFrame does not contain 'bankroll' column.")
        
        final_bankroll = results_df['bankroll'].iloc[-1]
        print(f"Final bankroll: {final_bankroll}")
        
        # Plotting bankroll over time
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['bankroll'], label='Bankroll Over Time')
        plt.title('Bankroll Over Time')
        plt.xlabel('Trade Number')
        plt.ylabel('Bankroll')
        plt.legend()
        plt.grid()
        plt.show()
        
        # Calculate key metrics
        total_trades = len(results_df)
        winning_trades = results_df['won_bet'].sum()
        win_rate = winning_trades / total_trades
        
        final_bankroll = results_df['bankroll'].iloc[-1]
        total_return = (final_bankroll - results_df['bankroll'].iloc[0]) / results_df['bankroll'].iloc[0]
        
        # Calculate Sharpe Ratio
        returns = results_df['bankroll'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Print results
        logger.info("\nBacktest Results:")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Winning Trades: {winning_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results_df['date'], results_df['bankroll'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Bankroll')
        plt.grid(True)
        plt.show()
        
        # Plot win rate by confidence level using custom bins
        plt.figure(figsize=(10, 5))
        
        # Create confidence bins with more precision in labels
        unique_confidences = sorted(results_df['confidence'].unique())
        n_bins = min(10, len(unique_confidences))
        
        if n_bins > 1:
            bin_edges = np.percentile(results_df['confidence'], 
                                    np.linspace(0, 100, n_bins + 1))
            
            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)
            
            # Create unique labels with more precision
            bin_labels = [f'{bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}' 
                         for i in range(len(bin_edges)-1)]
            
            # Assign data to bins
            results_df['confidence_bin'] = pd.cut(
                results_df['confidence'], 
                bins=bin_edges,
                labels=bin_labels,
                include_lowest=True,
                ordered=False  # Allow duplicate labels
            )
            
            # Calculate win rates and counts
            win_rates_df = results_df.groupby('confidence_bin', observed=True).agg({
                'won_bet': ['mean', 'count']
            }).droplevel(0, axis=1)
            
            # Plot
            ax = win_rates_df['mean'].plot(kind='bar', figsize=(12, 6))
            plt.title('Win Rate by Confidence Level')
            plt.xlabel('Confidence Range')
            plt.ylabel('Win Rate')
            
            # Add count labels on top of bars
            for i, (mean, count) in enumerate(zip(win_rates_df['mean'], win_rates_df['count'])):
                ax.text(i, mean, f'n={count}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            logger.warning("Not enough unique confidence values to create meaningful bins")
        
    except Exception as e:
        logger.error(f"Error analyzing backtest results: {str(e)}")
        raise