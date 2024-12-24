import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Optional

# Import the PaperTrader class
from paper_trader import PaperTrader

class EPLBacktester:
    def __init__(self, betting_model, initial_bankroll=10000):
        """
        Initialize the EPL backtester
        
        Args:
            betting_model: AdvancedBettingNeuralWrapper instance
            initial_bankroll: Starting bankroll amount
        """
        self.model = betting_model
        self.trader = PaperTrader(initial_bankroll)  # Initialize PaperTrader
        self.logger = logging.getLogger(__name__)  # Initialize logger

    def load_season_data(self, 
                        matches_file: str,
                        odds_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load EPL 2024-2025 season data
        
        Args:
            matches_file: Path to matches CSV file
            odds_file: Path to odds CSV file
            
        Returns:
            Tuple of matches and odds DataFrames
        """
        try:
            matches_df = pd.read_csv(matches_file)
            odds_df = pd.read_csv(odds_file)
            
            # Convert dates to datetime
            matches_df['date'] = pd.to_datetime(matches_df['date'])
            odds_df['date'] = pd.to_datetime(odds_df['date'])
            
            # Sort by date
            matches_df = matches_df.sort_values('date')
            odds_df = odds_df.sort_values('date')
            
            self.logger.info(f"Loaded {len(matches_df)} matches and {len(odds_df)} odds records")
            
            return matches_df, odds_df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def calculate_bet_size(self, confidence: float) -> float:
        """Calculate position size based on current capital and confidence"""
        if confidence < self.min_confidence:
            return 0.0
            
        # Kelly fraction with confidence adjustment
        fraction = min(self.bet_size_pct * (confidence / self.min_confidence), self.bet_size_pct)
        return self.current_capital * fraction
        
    def execute_trade(self, 
                     match_id: str,
                     prediction: int,
                     confidence: float,
                     odds: float,
                     actual_result: int,
                     match_date: datetime) -> Optional[Dict]:
        """
        Execute a single trade and track results
        
        Args:
            match_id: Unique match identifier
            prediction: Predicted outcome (0=home, 1=draw, 2=away)
            confidence: Model confidence score
            odds: Betting odds for predicted outcome
            actual_result: Actual match result
            match_date: Date of the match
            
        Returns:
            Trade details dictionary if trade executed, None otherwise
        """
        # Calculate bet size
        bet_size = self.trader.calculate_bet_size(confidence)  # Use PaperTrader's method
        
        if bet_size == 0:
            return None
            
        # Track pre-trade capital
        pre_trade_capital = self.trader.current_capital  # Use PaperTrader's current capital
        
        # Determine if prediction was correct
        won = prediction == actual_result
        
        # Update capital
        if won:
            self.trader.current_capital += bet_size * (odds - 1)
        else:
            self.trader.current_capital -= bet_size
            
        # Create trade record
        trade = {
            'match_id': match_id,
            'timestamp': datetime.now(),
            'prediction': prediction,
            'confidence': confidence,
            'odds': odds,
            'bet_size': bet_size,
            'pre_trade_capital': pre_trade_capital,
            'post_trade_capital': self.trader.current_capital,
            'won': won,
            'actual_result': actual_result
        }
        
        self.trader.trades.append(trade)  # Store trade in PaperTrader's trades
        self.logger.info(f"Trade executed - Match: {match_id}, Result: {'Won' if won else 'Lost'}")
        
        return trade
        
    def run_backtest(self, matches_df, odds_df):
        """Run backtest on historical matches"""
        try:
            results = []
            self.trader.current_capital = self.trader.initial_bankroll  # Reset capital
            
            for idx, match in matches_df.iterrows():
                try:
                    # Get match date and teams
                    match_date = pd.to_datetime(match['date'])
                    home_team = match['home_team']
                    away_team = match['away_team']
                    
                    # Get features for prediction
                    home_features = self.model._calculate_advanced_team_stats(home_team, match_date)
                    away_features = self.model._calculate_advanced_team_stats(away_team, match_date)
                    
                    # Combine features
                    match_features = np.concatenate([home_features, away_features]).reshape(1, -1)
                    
                    # Get prediction and confidence
                    pred_class, confidence, probabilities = self.model.predict_with_confidence(match_features)
                    
                    # Get match odds
                    match_odds = odds_df[odds_df['match_id'] == match['match_id']]
                    if len(match_odds) == 0:
                        self.logger.warning(f"No odds found for match {match['match_id']}")
                        continue
                        
                    odds_array = match_odds.iloc[0][['home_win_odds', 'draw_odds', 'away_win_odds']].values
                    selected_odds = odds_array[pred_class]
                    
                    # Get actual result from matches_df
                    actual_result = match['result']  # Assuming 'result' column exists
                    
                    # Execute trade using PaperTrader
                    trade = self.trader.execute_trade(
                        match_id=match['match_id'],
                        prediction=pred_class,
                        confidence=confidence,
                        odds=selected_odds,
                        actual_result=actual_result,
                        match_date=match_date
                    )
                    if trade:
                        results.append(trade)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing match {match['match_id']}: {str(e)}")
                    continue
            
            return self.trader.get_results()  # Return results from PaperTrader
        
        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}")
            raise
        
    def calculate_metrics(self):
        """Calculate and store backtest performance metrics"""
        if not self.trader.trades:
            self.logger.warning("No trades to analyze")
            return
            
        trades_df = pd.DataFrame(self.trader.trades)
        
        # Basic metrics
        self.metrics['total_trades'] = len(self.trader.trades)
        self.metrics['winning_trades'] = len(trades_df[trades_df['won']])
        self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        # Capital metrics
        self.metrics['final_capital'] = self.trader.current_capital
        self.metrics['total_return'] = (self.trader.current_capital - self.trader.initial_bankroll) / self.trader.initial_bankroll
        self.metrics['max_drawdown'] = self.calculate_max_drawdown()
        
        # Risk metrics
        self.metrics['sharpe_ratio'] = self.calculate_sharpe_ratio()
        self.metrics['avg_win'] = trades_df[trades_df['won']]['bet_size'].mean() if len(trades_df[trades_df['won']]) > 0 else 0
        self.metrics['avg_loss'] = trades_df[~trades_df['won']]['bet_size'].mean() if len(trades_df[~trades_df['won']]) > 0 else 0
        
        self.logger.info(f"Backtest completed - Final capital: ${self.trader.current_capital:,.2f}, Return: {self.metrics['total_return']:.2%}")
        
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak"""
        capital_series = pd.DataFrame(self.trader.trades)['post_trade_capital']
        rolling_max = capital_series.expanding().max()
        drawdowns = (capital_series - rolling_max) / rolling_max
        return abs(drawdowns.min())
        
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio of returns"""
        daily_returns = pd.DataFrame(self.trader.trades)['post_trade_capital'].pct_change().dropna()
        excess_returns = daily_returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(excess_returns) > 0 else 0
        
    def plot_results(self, save_path: str = 'backtest_results.png'):
        """Generate and save performance visualization plots"""
        if not self.trader.trades:
            self.logger.warning("No trades to plot")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Equity curve
        plt.subplot(2, 2, 1)
        capital_df = pd.DataFrame(self.trader.trades)
        plt.plot(capital_df['timestamp'], capital_df['post_trade_capital'])
        plt.title('Equity Curve')
        plt.xticks(rotation=45)
        
        # Win rate by confidence
        plt.subplot(2, 2, 2)
        trades_df = pd.DataFrame(self.trader.trades)
        confidence_bins = pd.qcut(trades_df['confidence'], q=5)
        win_rates = trades_df.groupby(confidence_bins)['won'].mean()
        win_rates.plot(kind='bar')
        plt.title('Win Rate by Confidence')
        
        # Monthly returns
        plt.subplot(2, 2, 3)
        monthly_returns = capital_df.set_index('timestamp').resample('M')['post_trade_capital'].last().pct_change()
        monthly_returns.plot(kind='bar')
        plt.title('Monthly Returns')
        plt.xticks(rotation=45)
        
        # Drawdown chart
        plt.subplot(2, 2, 4)
        rolling_max = capital_df['post_trade_capital'].expanding().max()
        drawdowns = (capital_df['post_trade_capital'] - rolling_max) / rolling_max
        plt.plot(capital_df['timestamp'], drawdowns)
        plt.title('Drawdown')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
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
    
    # Initialize neural wrapper with algorithm
    from betting_model import AdvancedBettingNeuralWrapper
    wrapper = AdvancedBettingNeuralWrapper(algo)
    
    # Create backtester
    backtester = EPLBacktester(
        betting_model=wrapper,
        initial_bankroll=10000.0
    )
    
    # Run backtest
    results = backtester.run_backtest(matches_df, odds_df)

    