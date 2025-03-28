import torch
import numpy as np
import pandas as pd

def getDraws(y_train_res, y_test):
    train_draw_count = y_train_res.value_counts()[1]
    test_draw_count = y_test.value_counts()[1]

    training_total_draws = y_train.shape[0]
    testing_total_draws = y_test.shape[0]

    training_draw_percentage = round( (train_draw_count / training_total_draws) * 100 )
    testing_draw_percentage = round( (test_draw_count / testing_total_draws) * 100 )


    training_output = f"""TRAINING:
        \n\t Features shape: {X_train_res.shape} 
        \t Test shape : {y_train_res.shape} 
        \t Draw-Count: { y_train_res.value_counts()[1]}
        \t Draw-Percentage: {training_draw_percentage}% \n
    """
    
    testing_output = f"""TESTING:
        \n\t Features shape: {X_test.shape} 
        \t Test shape: {y_test.shape}
        \t Draw-Count: {y_test.value_counts()[1]}
        \t  Draw-Percentage: {testing_draw_percentage}% \n
    """
    
    print(training_output)
    print(testing_output)
    
    return training_draw_percentage, testing_draw_percentage


def create_favorites(df):
    favorite = []
    i = 0
    for odd in df['home_win_odds']:
        if(odd < 2.0): favorite.append(df['home_team'][i])
        
        elif (odd > 2.0): favorite.append(df['away_team'][i])
        
        else:
            favorite.append(-1)

        i += 1
        
    df['favorite'] = favorite
    return df


def construct_dataset(DataPath):
    matches = pd.read_csv(DataPath + 'matches.csv')
    odds = pd.read_csv(DataPath + 'odds.csv')
    df = pd.merge(matches, odds, on='match_id', how='inner')
    df.columns = df.columns.str.strip().str.replace(' ', '')
    df = create_favorites(df)
    print(df)
    return df


def prepare_dataset(df):
    df = df.drop(columns=['date_y']).rename(columns={'date_x': 'date'})
    
    # Get all the teams and construct a unique id's
    unique_teams = set(df['home_team']).union(set(df['away_team']))
    
    # map each team to a unique integer
    team_to_id = {team: idx for idx, team in enumerate(unique_teams)}
    
    # create new columns of integer id's
    df['home_team_id'] = df['home_team'].map(team_to_id)
    df['away_team_id'] = df['away_team'].map(team_to_id)
    df['favorite_id'] = df['favorite'].map(team_to_id)

    # create a new column to hold 1's and 0's.
    # 1: Draw
    # 0: Not a Draw

    df['is_draw'] = (df['result'] == 1).astype(int)
    
    return df


def min_max_normalization(matrix):

    # get min and max for each feature (column)
        # dim=0: Go down the columns; dim=1: Go aross the rows.
        
        # ft_mins: tensor of size that is equal to the amount of columns in the matrix.
        #          Contains the minimum values of each column in the matrix.
    
    ft_mins, _ = torch.min(matrix, dim=0)
    ft_max, _ = torch.max(matrix, dim=0)
    # print(ft_mins, ft_max)
    
    # norm will be updated later and ultimately returned.
    norm = matrix
    
    # number of rows
    rows = matrix.shape[0]
    # number of cols
    cols = matrix.shape[1]
    
    # go to column (i), then iterate down column with (j)
    for i in range(0,cols):
        x_min = ft_mins[i]
        x_max = ft_max[i]
        dx = x_max - x_min
        for j in range(0,rows):
            x = norm[j][i]
            norm[j][i] = (x - x_min) / dx
    
    
    return norm
    
