import requests

def fetch_epl_odds(api_key: str, region: str = 'uk', sport: str = 'soccer_epl') -> dict:
    """
    Fetch EPL betting odds using The Odds API.

    Args:
        api_key (str): Your API key from The Odds API.
        region (str): Region for bookmakers (default is 'uk').
        sport (str): Sport key for EPL (default is 'soccer_epl').

    Returns:
        dict: JSON response with odds data.
    """
    url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds'
    params = {
        'apiKey': api_key,
        'regions': region,
        'markets': 'h2h',  # Head-to-head odds
        'oddsFormat': 'decimal'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Example usage
api_key = '21e983fbd0d7e3e13688c7258d12db2f'
odds_data = fetch_epl_odds(api_key)
print(odds_data)
