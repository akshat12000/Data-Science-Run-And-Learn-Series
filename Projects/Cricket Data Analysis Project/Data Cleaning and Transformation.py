import json
import pandas as pd

# Read the data from the JSON file
with open("./Data/t20_json_files/t20_wc_match_results.json") as f:
    data = json.load(f)

# Create a dataframe from the data
df_match = pd.DataFrame(data[0]['matchSummary'])

# Creating Id
df_match.rename({'scorecard':'match_id'}, axis=1, inplace=True)

match_id_dict = {}
for index , row in df_match.iterrows():
    key1 = row['team1']+' Vs '+row['team2']
    key2 = row['team2']+' Vs '+row['team1']
    match_id_dict[key1] = row['match_id']
    match_id_dict[key2] = row['match_id']

df_match.to_csv('./Data/t20_csv_files/dim_match_summary.csv', index = False)

with open('./Data/t20_json_files/t20_wc_batting_summary.json') as f:
    data = json.load(f)
    all_records = []
    for record in data:
        all_records.extend(record['battingSummary'])

# Create a dataframe from the data
df_batting = pd.DataFrame(all_records)

# Cleaning the data
df_batting["out/not_out"] = df_batting.dismissal.apply(lambda x: "out" if len(x)>0 else "not_out")
df_batting.drop("dismissal", axis=1, inplace=True)
df_batting["batsmanName"] = df_batting["batsmanName"].apply(lambda x: x.replace("â€",""))
df_batting["batsmanName"] = df_batting["batsmanName"].apply(lambda x: x.replace("\xa0",""))
df_batting["match_id"] = df_batting['match'].map(match_id_dict)

df_batting.to_csv("./Data/t20_csv_files/t20_wc_batting_summary.csv", index=False)

with open("./Data/t20_json_files/t20_wc_bowling_summary.json") as f:
    data = json.load(f)
    all_records = []
    for record in data:
        all_records.extend(record['bowlingSummary'])

# Create a dataframe from the data
df_bowling = pd.DataFrame(all_records)

# Cleaning the data
df_bowling['match_id'] = df_bowling['match'].map(match_id_dict)

df_bowling.to_csv('./Data/t20_csv_files/fact_bowling_summary.csv', index = False)

with open('./Data/t20_json_files/t20_wc_player_info.json') as f:
    data = json.load(f)

# Create a dataframe from the data
df_players = pd.DataFrame(data)

# Cleaning the data
df_players['name'] = df_players['name'].apply(lambda x: x.replace('â€', ''))
df_players['name'] = df_players['name'].apply(lambda x: x.replace('†', ''))
df_players['name'] = df_players['name'].apply(lambda x: x.replace('\xa0', ''))

df_players.to_csv('./Data/t20_csv_files/dim_players_no_images.csv', index = False)