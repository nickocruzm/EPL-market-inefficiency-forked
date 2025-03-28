import pandas as pd
from datetime import datetime, timedelta

class BaseAlgorithm:
    def __init__(self, matches_df, odds_df = None):
        """
        Initialize algorithm with historical match data
        
        Args:
            matches_df: DataFrame containing historical match results
        """
        self.historical_matches = matches_df
        self.historical_odds = odds_df
        self.match_results = matches_df.copy()
        self.match_results['date'] = pd.to_datetime(self.match_results['date'])
       
    def get_team_stats(self, team_name: str, before_date: datetime, lookback_days: int = 90) -> dict:
        """
        Calculate basic team statistics for given timeframe
        
        Args:
            team_name: Name of the team
            before_date: Calculate stats before this date
            lookback_days: Number of days to look back
        
        Returns:
            Dictionary containing team statistics
        """
        
        start_date = before_date - timedelta(days=lookback_days)
        
        
        # Filter matches for the team within the timeframe
        team_matches = self.match_results[
            (self.match_results['date'] < before_date) &
            (self.match_results['date'] >= start_date) &
            ((self.match_results['home_team'] == team_name) |
             (self.match_results['away_team'] == team_name))
        ]
        
        # Initialize stats
        stats = {
            'matches_played': len(team_matches),
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0
        }
        
        if stats['matches_played'] == 0:
            return stats
            
        # Calculate basic stats
        for _, match in team_matches.iterrows():
            if match['home_team'] == team_name:
                if match['result'] == 0:  # Home win
                    stats['wins'] += 1
                elif match['result'] == 1:  # Draw
                    stats['draws'] += 1
                else:  # Away win
                    stats['losses'] += 1
            else:  # Team played away
                if match['result'] == 2:  # Away win
                    stats['wins'] += 1
                elif match['result'] == 1:  # Draw
                    stats['draws'] += 1
                else:  # Home win
                    stats['losses'] += 1
        
        return stats