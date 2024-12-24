# project/paper_trader.py
import logging
import pandas as pd

class PaperTrader:
    def __init__(self, initial_bankroll=10000):
        self.initial_bankroll = initial_bankroll
        self.current_capital = initial_bankroll
        self.trades = []
        self.logger = logging.getLogger(__name__)

    def execute_trade(self, match_id, prediction, confidence, odds, actual_result, match_date):
        bet_size = self.calculate_bet_size(confidence)
        if bet_size == 0:
            return None
        
        pre_trade_capital = self.current_capital
        won = prediction == actual_result
        
        if won:
            self.current_capital += bet_size * (odds - 1)
        else:
            self.current_capital -= bet_size
        
        trade = {
            'match_id': match_id,
            'prediction': prediction,
            'confidence': confidence,
            'odds': odds,
            'bet_size': bet_size,
            'pre_trade_capital': pre_trade_capital,
            'post_trade_capital': self.current_capital,
            'won_bet': won,
            'actual_result': actual_result,
            'bankroll': self.current_capital,
            'date': match_date
        }
        
        self.trades.append(trade)
        self.logger.info(f"Trade executed - Match: {match_id}, Result: {'Won' if won else 'Lost'}")
        return trade

    def calculate_bet_size(self, confidence):
        return self.current_capital * 0.02  # Example: 2% of current capital

    def get_results(self):
        return pd.DataFrame(self.trades)