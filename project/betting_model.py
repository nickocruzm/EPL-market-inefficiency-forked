import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from datetime import datetime, timedelta
import logging
from logging import handlers  # Add this import
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

class AdvancedBettingNeuralWrapper:
    def __init__(self, base_algorithm, initial_bankroll=10000, min_confidence=0.6, max_bet_size=0.1):
        """
        Initialize the advanced betting neural wrapper
        
        Args:
            base_algorithm: BaseAlgorithm instance for data processing
            initial_bankroll: Starting bankroll amount
            min_confidence: Minimum confidence threshold for placing bets
            max_bet_size: Maximum bet size as fraction of bankroll
        """
        self.algo = base_algorithm
        self.initial_bankroll = initial_bankroll
        self.min_confidence = min_confidence
        self.max_bet_size = max_bet_size
        
        # Initialize model components
        self.model = None
        self.scaler = RobustScaler()
        self.feature_cache = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Enhanced performance tracking
        self.performance_metrics = {
            'predictions': [],
            'actual_results': [],
            'bet_amounts': [],
            'returns': [],
            'bankroll_history': [],
            'confidence_scores': [],
            'validation_metrics': [],
            'feature_importance': {}
        }
        
        # Risk management parameters
        self.current_bankroll = self.initial_bankroll
        self.losing_streak = 0
        self.max_losing_streak = 0
        
        # Market efficiency metrics
        self.market_efficiency = {
            'odds_movement': [],
            'closing_line_value': [],
            'market_volume': []
        }

    def setup_logging(self):
        """Enhanced logging setup with rotation and detailed formatting"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Check if handlers already exist to avoid duplicates
        if not self.logger.handlers:
            # File handler with rotation
            file_handler = handlers.RotatingFileHandler(
                'betting_model.log',
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            
            # Enhanced formatter with more details
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    def _calculate_advanced_team_stats(self, team, date, lookback=10):
        """Enhanced team statistics with advanced metrics"""
        try:
            if (team, date) in self.feature_cache:
                return self.feature_cache[(team, date)]
            
            date = pd.to_datetime(date)
            
            # Debug the data structure
            past_matches = self.algo.match_results[
                (self.algo.match_results['date'] < date) &
                ((self.algo.match_results['home_team'] == team) |
                 (self.algo.match_results['away_team'] == team))
            ].tail(lookback)
            
            # Log available columns
            self.logger.debug(f"Available columns: {past_matches.columns.tolist()}")
            
            # Initialize stats with default values
            stats = {
                'goals_scored': [],
                'goals_conceded': [],
                'points': []
            }
            
            if len(past_matches) == 0:
                return np.zeros(20)
            
            # Process each match with proper column handling
            for _, row in past_matches.iterrows():
                try:
                    is_home = row['home_team'] == team
                    
                    # Get scores based on actual column names
                    if is_home:
                        goals_scored = float(row.get('home_score', row.get('score_home', row.get('home_goals', 0))))
                        goals_conceded = float(row.get('away_score', row.get('score_away', row.get('away_goals', 0))))
                    else:
                        goals_scored = float(row.get('away_score', row.get('score_away', row.get('away_goals', 0))))
                        goals_conceded = float(row.get('home_score', row.get('score_home', row.get('home_goals', 0))))
                    
                    # Calculate points
                    if goals_scored > goals_conceded:
                        points = 3
                    elif goals_scored == goals_conceded:
                        points = 1
                    else:
                        points = 0
                    
                    # Append to stats
                    stats['goals_scored'].append(goals_scored)
                    stats['goals_conceded'].append(goals_conceded)
                    stats['points'].append(points)
                    
                except Exception as e:
                    self.logger.debug(f"Skipping match due to: {str(e)}")
                    continue
            
            # If no valid stats were collected, return zeros
            if not stats['goals_scored']:
                return np.zeros(20)
            
            # Calculate features
            advanced_stats = []
            
            # Scoring ability (5 features)
            advanced_stats.extend([
                np.mean(stats['goals_scored']),
                np.std(stats['goals_scored']) if len(stats['goals_scored']) > 1 else 0,
                np.mean(stats['goals_conceded']),
                np.std(stats['goals_conceded']) if len(stats['goals_conceded']) > 1 else 0,
                sum(stats['goals_scored']) - sum(stats['goals_conceded'])
            ])
            
            # Form metrics (5 features)
            advanced_stats.extend([
                np.mean(stats['points']),
                sum(stats['points'][-3:]) if len(stats['points']) >= 3 else sum(stats['points']),
                sum(stats['points'][-5:]) if len(stats['points']) >= 5 else sum(stats['points']),
                self.calculate_streak(stats['points']),
                len(stats['points']) / lookback
            ])
            
            # Fill remaining features with calculated metrics
            while len(advanced_stats) < 20:
                advanced_stats.append(0.0)
            
            # Ensure exactly 20 features
            advanced_stats = advanced_stats[:20]
            
            # Cache and return
            result = np.array(advanced_stats)
            self.feature_cache[(team, date)] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error in _calculate_advanced_team_stats for {team}: {e}")
            return np.zeros(20)

    def calculate_streak(self, results):
        """Calculate current streak with error handling"""
        if not results:
            return 0
        streak = 0
        for result in reversed(results):
            if result > 0 and streak >= 0:
                streak += 1
            elif result == 0 and streak <= 0:
                streak -= 1
            else:
                break
        return streak

    def calculate_volatility(self, values):
        """Calculate volatility with error handling"""
        if len(values) < 2:
            return 0
        try:
            returns = np.diff(values) / np.array([v if v != 0 else 1 for v in values[:-1]])
            weights = np.exp(np.linspace(-1, 0, len(returns)))
            weights /= weights.sum()
            return np.sqrt(np.sum(weights * returns ** 2))
        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {e}")
            return 0

    def _calculate_enhanced_market_features(self, match_id):
        """Calculate enhanced market features with liquidity and momentum"""
        match_odds = self.algo.historical_odds[
            self.algo.historical_odds['match_id'] == match_id
        ]
        
        if len(match_odds) == 0:
            return np.zeros(15)
            
        features = []
        
        # Basic odds features
        odds_columns = ['home_win_odds', 'draw_odds', 'away_win_odds']
        odds_data = match_odds[odds_columns]
        
        # Opening and closing odds
        opening_odds = odds_data.iloc[0]
        closing_odds = odds_data.iloc[-1]
        
        # Odds movement
        odds_movement = (closing_odds - opening_odds) / opening_odds
        
        # Calculate implied probabilities
        def calculate_implied_probs(odds_row):
            implied_probs = 1 / odds_row
            margin = implied_probs.sum() - 1
            return implied_probs / implied_probs.sum(), margin
        
        opening_probs, opening_margin = calculate_implied_probs(opening_odds)
        closing_probs, closing_margin = calculate_implied_probs(closing_odds)
        
        # Market efficiency metrics
        margin_change = closing_margin - opening_margin
        
        # Odds stability
        odds_volatility = odds_data.std() / odds_data.mean()
        
        # Market consensus
        bookmaker_disagreement = odds_data.std() / odds_data.mean()
        
        # Steam moves detection
        odds_diff = odds_data.diff()
        sharp_moves = (abs(odds_diff) > odds_data.std()).any(axis=1).sum()
        
        # Combine all features
        features.extend([
            *opening_probs,
            *closing_probs,
            *odds_movement,
            opening_margin,
            closing_margin,
            margin_change,
            *odds_volatility,
            *bookmaker_disagreement,
            sharp_moves
        ])
        
        return np.array(features)

    def build_enhanced_model(self, input_shape):
        """Build enhanced neural network with attention mechanism"""
        try:
            # Input validation
            if not input_shape or (isinstance(input_shape, tuple) and 0 in input_shape):
                raise ValueError("Invalid input shape provided")

            # Convert input_shape to tuple if it's a single number
            if isinstance(input_shape, (int, np.integer)):
                input_shape = (input_shape,)

            # Clear any existing backend session
            tf.keras.backend.clear_session()
                
            inputs = layers.Input(shape=input_shape)
            
            # Batch normalization on input
            x = layers.BatchNormalization()(inputs)
            
            # First dense block
            x = layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
            x = layers.LeakyReLU()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Second dense block
            x = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
            x = layers.LeakyReLU()(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
            # Output layer
            outputs = layers.Dense(3, activation='softmax')(x)
            
            model = models.Model(inputs=inputs, outputs=outputs)
            
            # Basic model compilation
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.logger.info(f"Model built successfully with input shape: {input_shape}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise

    def train_enhanced_model(self, features, labels, validation_split=0.2):
        """Enhanced training process with basic error handling"""
        self.logger.info("Starting enhanced model training...")
        
        try:
            # Input validation
            if len(features) == 0 or len(labels) == 0:
                raise ValueError("Empty features or labels provided")
                
            # Prepare data
            features = np.array(features)
            labels = np.array(labels)
            
            # Remove any NaN values
            valid_mask = ~np.isnan(features).any(axis=1)
            features = features[valid_mask]
            labels = labels[valid_mask]
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Build model if not exists
            if self.model is None:
                self.model = self.build_enhanced_model(features.shape[1])
            
            # Simple train-validation split
            train_size = int(len(features_scaled) * (1 - validation_split))
            
            X_train = features_scaled[:train_size]
            X_val = features_scaled[train_size:]
            y_train = labels[:train_size]
            y_val = labels[train_size:]
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
            self.logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
            
            return [history.history]
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise


        # to here**

    def predict_with_confidence(self, features):
        """Predict match outcome with confidence scores"""
        try:
            # Ensure features are in correct shape (40 features total)
            if features.shape[1] == 20:
                features = np.concatenate([features, features], axis=1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions
            predictions = self.model.predict(features_scaled, verbose=0)
            pred_class = np.argmax(predictions, axis=1)
            confidence = np.max(predictions, axis=1)
            
            return int(pred_class[0]), float(confidence[0]), predictions[0]
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def calculate_bet_size(self, confidence, odds):
        """Calculate optimal bet size using Kelly Criterion with risk management"""
        if confidence < self.min_confidence:
            return 0
            
        # Modified Kelly Criterion
        p = confidence
        q = 1 - p
        b = odds - 1  # Convert odds to decimal form
        
        # Kelly fraction
        f = (p * b - q) / b
        
        # Apply fractional Kelly (more conservative)
        f = f * 0.3
        
        # Apply maximum bet constraint
        max_bet = self.current_bankroll * self.max_bet_size
        
        # Reduce bet size if on losing streak
        if self.losing_streak > 0:
            f = f * (0.5 ** (self.losing_streak / 3))
        
        bet_amount = min(f * self.current_bankroll, max_bet)
        return max(0, bet_amount)

    def update_bankroll(self, bet_amount, outcome, odds):
        """Update bankroll and track performance"""
        previous_bankroll = self.current_bankroll
        
        if outcome:
            winnings = bet_amount * (odds - 1)
            self.current_bankroll += winnings
            self.losing_streak = 0
        else:
            self.current_bankroll -= bet_amount
            self.losing_streak += 1
            self.max_losing_streak = max(self.max_losing_streak, self.losing_streak)
        
        # Track performance
        self.performance_metrics['bankroll_history'].append({
            'timestamp': datetime.now(),
            'previous_bankroll': previous_bankroll,
            'current_bankroll': self.current_bankroll,
            'bet_amount': bet_amount,
            'outcome': outcome,
            'odds': odds
        })
        
        # Calculate ROI
        total_bets = len(self.performance_metrics['bankroll_history'])
        if total_bets > 0:
            roi = (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll
            self.logger.info(f"Current ROI: {roi:.2%}")

    def analyze_performance(self):
        """Generate comprehensive performance analysis"""
        if not self.performance_metrics['bankroll_history']:
            return "No betting history available."
            
        # Calculate key metrics
        total_bets = len(self.performance_metrics['bankroll_history'])
        winning_bets = sum(1 for bet in self.performance_metrics['bankroll_history'] if bet['outcome'])
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        total_profit = self.current_bankroll - self.initial_bankroll
        roi = total_profit / self.initial_bankroll
        
        # Create performance plots
        plt.figure(figsize=(15, 10))
        
        # Bankroll curve
        plt.subplot(2, 2, 1)
        bankroll_history = [bet['current_bankroll'] for bet in self.performance_metrics['bankroll_history']]
        plt.plot(bankroll_history)
        plt.title('Bankroll Over Time')
        plt.xlabel('Bet Number')
        plt.ylabel('Bankroll')
        
        # ROI histogram
        plt.subplot(2, 2, 2)
        returns = [(bet['current_bankroll'] - bet['previous_bankroll']) / bet['previous_bankroll'] 
                  for bet in self.performance_metrics['bankroll_history']]
        plt.hist(returns, bins=50)
        plt.title('Distribution of Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        
        # Confidence vs. Outcome
        plt.subplot(2, 2, 3)
        confidence_scores = self.performance_metrics['confidence_scores']
        outcomes = [bet['outcome'] for bet in self.performance_metrics['bankroll_history']]
        plt.scatter(confidence_scores, outcomes, alpha=0.5)
        plt.title('Confidence vs. Outcome')
        plt.xlabel('Confidence Score')
        plt.ylabel('Outcome (Win/Loss)')
        
        # Save plots
        plt.tight_layout()
        plt.savefig('performance_analysis.png')
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'max_losing_streak': self.max_losing_streak,
            'current_bankroll': self.current_bankroll,
            'total_profit': total_profit
        }