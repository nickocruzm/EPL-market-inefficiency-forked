# Explanation:

## main.py
1. Data is loaded into `matches_df` and `odds_df`
2. BaseAlgorithm class is initialized 
3. AdvancedBettingNeuralWrapper object initialized using the base algorithm class and the following arguments:
   - bankroll: 10,000
   - minimum confidence: 60%
   - maximum bet size: 10% of bankroll

4. Model is initialized and trained on:
   - matches dataframe
   - odds dataframe
   - NeuralWrapper is passed into object as the wrapper????
     - The wrapper inside of the training and init is also strange.

   - test size is set to 0.2.
     - Unsure if this is the amount of data being used to train on...

5. Validate model output

6. Init backtester
    - We pass a new bankroll into the backtester...?

7. Backtest START
    - passes in matches and odds dataframe

8. Backtest results analyzed.


- data is split between two seperate files:
  - matches
  - odd

- matches_df and odds_df are pandas dataframes.
  - both dataframes are sorted by date.


- Are there columns that aren't required?
- Code below should probably use assert to ensure these columns exist within the dataframe.

```python
# Validate required columns
    required_match_cols = ['date', 'home_team', 'away_team', 'result', 'match_id']
    required_odds_cols = ['date', 'match_id', 'home_win_odds', 'draw_odds', 'away_win_odds']

```

## BaseAlgorithm

### Initialization

- arguments passed in are 
  - matches_df: pandas.dataframe
  - odds_df: pandas.dataframe
    - Default value is set to None.


- The two attributes below, both are assigned the matches_df argument.

```python

self.historical_matches
self.match_results

```

- during initialization the matches 'date' column is converted to datetime object.

### get_team_stats()

- args:
  - team_name: str
  - before_date: datetime
  - lookback_days: int

- return: dictionary 

- looks at team's data up until specifed date (`before_date`)

- starting at the given date and going back the number of   `lookback_days` from that specified date.

- filters match results such that:
  - match_results are in between the start and ending date.
  - the match_result's are associated with the team name given as an argument.

- only could take in one team name at a time.

## AdvancedBettingNeuralWrapper

- uses `RobustScaler()`
  - maybe playing around with this later would be interesting.

## model_initialization

### initialize and train model

- args
  - matches_df: pandas.DataFrame
  - odds_df: pandas.DataFrame


1. Match Dataframe is sorted chronologically by 'date'.
2. Iterate through all the matches in the dataframe.
  - create a `match_date` list (series?) that holds the dates of matches as datetime objects.
  - `_calculate_advanced_team_stats()` is called twice, first time is called to return features for the home team. The second time it is called is to return features for the away team.

  - if there are features inside of home AND away then:
    - both home and away sets of features are combined and placed into a features list
    - home_score and away_score are assigned float values.
    - home_score and away_score are then used to determine the results of the match.
      - 0,1,2
      - 0: Home win
      - 1: Draw
      - 2: Away win
    - labels list is appened with the result.
    - number of valid matches is incremented by 1.
    - Start the second iteration.

  3. Check if any of the matches are valid.
  4. Convert features: list -> np.array
  5. convert labels: list -> np.array
  6. Set up data to build and train model
   - Split up arrays into training and testing datasets.
   - X-train features are scaled 
   - X-test are scaled
   - y train and test are converted to categorical data

  7. Build then Train enhanced model, return trained model.














## Notes

### BaseAlgorithm.init()

- if match_results['date'] is used as a pd.to_datetime anywhere else it is best to only have one location where it gets converted to a datetime object.

### BaseAlgorithm.get_team_stats()


#### possible code updates?

- Why use lookback_days and not just give a start and end date?

- should probably clean up the code below, setting the filter condition to its own variable.

```python

  team_matches = self.match_results[
      (self.match_results['date'] < before_date) &
      (self.match_results['date'] >= start_date) &
      ((self.match_results['home_team'] == team_name) |
        (self.match_results['away_team'] == team_name))
  ]

```