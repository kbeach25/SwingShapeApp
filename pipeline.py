from pybaseball import statcast, playerid_reverse_lookup
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pandasql import sqldf
import os

warnings.filterwarnings('ignore')

### Create folder for the data
os.makedirs("data", exist_ok = True)

### Data gathering function
# - Using 2025 MLB data
def getData(start_date, end_date):
    # raw data straight from statcast
    statcast_df = statcast(start_date, end_date)

    # add the names using the player ids
    ids = statcast_df['batter'].dropna().unique()
    id_df = playerid_reverse_lookup(ids, key_type = 'mlbam')

    # capitalize names 
    id_name_dict = {}
    for each, row in id_df.iterrows():
        first_name = row['name_first'].title()
        last_name = row['name_last'].title()
        id_name_dict[row['key_mlbam']] = f'{first_name} {last_name}'
    
    statcast_df['batter_name'] = statcast_df['batter'].map(id_name_dict)

    return statcast_df

### Time to contact calclation
# - Adapted from Driveline article
def calculate_ttc(bat_speed, swing_length):

    if pd.isna(bat_speed) or pd.isna(swing_length):
        return float('nan')
    
    ttc = round(1.3636 * (swing_length / bat_speed), 4)

    return ttc

### Vertical Bat Angle calculation
def calculate_vba(tilt):
    return tilt * -1

### Turn hard hit balls into a binary variable
def encode_hard_hit(exit_velocity):
    if pd.isna(exit_velocity):
        return None
    
    # hard hit ball is 95+
    if exit_velocity >= 95.0:
        return 1
    else:
        return 0

### Turn contact into a binary variable
# - Considering swings only
def encode_contact(contact):
    contact_descriptions = {'hit_into_play', 'foul'}
    whiff_descriptions = {'swinging_strike', 'foul_tip', 'striking_strike_blocked'}

    if contact in contact_descriptions:
        return 1
    elif contact in whiff_descriptions:
        return 0
    else:
        return None

### Generalize pitch type and encode
def encode_pitch_type(pitch_type):
    fastballs = {"FF", "FT", "SI", "FC"}
    breaking = {"SL", "CU", "KC", "SC", "KN", "ST"}
    offspeed = {"CH", "FS", "FO", "EP"}

    if pitch_type in fastballs:
        return 0
    elif pitch_type in breaking:
        return 1
    elif pitch_type in offspeed:
        return 2
    else:
        return None

### Encode specific pitch types, excude screwball (SC), knuckleball (KN), forkball (FO), eephus (EP)
def encode_pitch_type_specific(pitch_type):
    # four seam
    if pitch_type == "FF":
        return 0
    # two seam/sinker
    elif pitch_type == "SI" or pitch_type == "FT":
        return 1
    # cutter 
    elif pitch_type == "FC":
        return 2
    # slider
    elif pitch_type == "SL":
        return 3
    # curveball
    elif pitch_type == "CU" or pitch_type == "KC":
        return 4
    # changeup
    elif pitch_type == "CH":
        return 5
    # splitter
    elif pitch_type == "FS":
        return 6
    # sweeper
    elif pitch_type == "ST":
        return 7
    else:
        return 8

### Encode pitcher and batter handedness
def encode_side(side):
    if side == "R":
        return 0
    elif side == "L":
        return 1
    else:
        return None
    
### Average swing shape metric function, competitive swings only
def individual_swing_shape(batter_df, player_id):
    batter_df = batter_df[batter_df['batter'] == player_id]

    # No swings below 10th percentile of bat speed
    threshold = batter_df['bat_speed'].quantile(0.10)
    batter_df = batter_df[batter_df['bat_speed'] >= threshold]

    return {
        'name': batter_df['batter_name'].iloc[0],
        'batter': player_id,
        'bat_speed': round(batter_df['bat_speed'].mean(), 3),
        'attack_angle': round(batter_df['attack_angle'].mean(), 3),
        'vba': round(batter_df['vba'].mean(), 3),
        'ttc': round(batter_df['ttc'].mean(), 3),
        'zone_top': round(batter_df['sz_top'].mean(), 3),
        'zone_bot': round(batter_df['sz_bot'].mean(), 3),
        'hba': round(batter_df['attack_direction'].mean(), 3),
        'side': batter_df['stand'].iloc[0]
    }

### Function for assigning pitch zones, using an 8x8 grid
def ZoneSections(df, sections):
    df['ZoneHeight'] = df['ZoneTop'] - df['ZoneBot']
    sec_height = df['ZoneHeight'] / sections
    sec_width = 17 / sections

    # start at the top of the strike zone and on the outer left (-8.5 inches)
    current_height = df['ZoneTop']
    current_width = -8.5

    pitch_side = df['PitchX']
    pitch_height = df['PitchZ']

    height_coord = 0
    width_coord = 0
    # if the pitch's height means it could be a strike, assign it a general height location
    if pitch_height < df['ZoneTop'] and pitch_height > df['ZoneBot']:
        height_coord = 1

        # pitch is in a particular zone if it's between the top coordinate of that zone (current_height)
        # and the bottom coordinate of that zone (current_height - sec_height)
        while current_height - sec_height > pitch_height:
            # end the loop if you reach the bottom of the zone; the pitch is not a strike
            if height_coord == sections:
                break
            
            height_coord += 1
            current_height -= sec_height

    if pitch_side > -8.5 and pitch_side < 8.5:
        width_coord = 1
        
        # same logic for inside/outside pitches
        while current_width + sec_width < pitch_side:
            if width_coord == sections:
                break

            width_coord += 1
            current_width += sec_width

    return f'{height_coord}{width_coord}'

### Function to assign batted ball type (ground ball, fly ball, etc.)
def hit_class(df):
    df = df.copy()

    # using mlb definitions for batted ball types
    conditions = [df['LaunchAngle'] < 10,
                  (df['LaunchAngle'] >= 10) & (df['LaunchAngle'] <= 25),
                  (df['LaunchAngle'] > 25) & (df['LaunchAngle'] <= 50),
                  df['LaunchAngle'] > 50]
    
    labels = ['isGB', 'isLD', 'isFB', 'isPU']

    for label, cond in zip(labels, conditions):
        df[label] = np.where(cond, 1, 0)

    return df

### Get the raw Statcast data
mlb_data = getData('2025-03-27', '2025-09-28')

### Apply the functions to add the necessary columns
mlb_data['ttc'] = mlb_data.apply(lambda row: calculate_ttc(row['bat_speed'], row['swing_length']), axis = 1)
mlb_data['vba'] = calculate_vba(mlb_data['swing_path_tilt'])
mlb_data['hard_hit_encoded'] = mlb_data['launch_speed'].apply(encode_hard_hit)
mlb_data['contact_encoded'] = mlb_data['description'].apply(encode_contact)
mlb_data['pitch_type_encoded'] = mlb_data['pitch_type'].apply(encode_pitch_type)
mlb_data['pitch_type_specific_encoded'] = mlb_data['pitch_type'].apply(encode_pitch_type_specific)
mlb_data['pitcher_hand'] = mlb_data['p_throws'].apply(encode_side)
mlb_data['batter_hand'] = mlb_data['stand'].apply(encode_side)

### MLB requires at least 3.1 plate appearances per scheduled game to be a qualified batter
# - For 3.1 PA/G, about 3.85 P/PA, 162 games, that's about 1933 pitches to qualify
# - That's pretty high, so I'll use about half of that. Minimum of 900 pitches to qualify
qualified_batters = mlb_data['batter'].value_counts()
qualified_batters = qualified_batters[qualified_batters >= 900].index

results = []
for player_id in qualified_batters:
    results.append(individual_swing_shape(mlb_data, player_id))

### Convert data to a dataframe
swing_shape_df = pd.DataFrame(results)

### Need to join batting-stance data from BaseballSavant to swing shape data
stance_data = pd.read_csv("batting-stance.csv")

swingshape_query = """
                   SELECT
                   ss.name AS name,
                   ss.batter AS batter,
                   ss.bat_speed AS bat_speed,
                   ss.attack_angle AS attack_angle,
                   ss.vba AS vba,
                   ss.ttc AS ttc,
                   bs.bat_side AS side,
                   bs.avg_batter_y_position AS batter_depth,
                   bs.avg_batter_x_position AS batter_distance,
                   avg_intercept_y_vs_plate AS contact_depth

                   FROM swing_shape_df AS ss

                   INNER JOIN stance_data AS bs
                   ON ss.batter = bs.id
                   """

pysql = lambda q: sqldf(q, globals())

swing_shape_data_full = pysql(swingshape_query)
### This is the first output, the dataframe with individual players and swing shapes
swing_shape_filename = "SwingShapeData.csv"
swing_shape_data_full.to_csv(f"data/{swing_shape_filename}", index = False)
print(f'Saved swing shape data as {swing_shape_filename}')

### mlb_data is pitch-by-pitch data, swing_shape_df is individual player swing shape data
# - These need to be joined so the individual's swing shape profile is assigned to each pitch they saw
# - Join using batter, their ID
swing_shape_df['batter'] = swing_shape_df['batter'].astype(int)
mlb_data['batter'] = mlb_data['batter'].astype('int64')

query = """
        SELECT
        shape.name AS Name,
        shape.batter AS BatterID, 
        shape.bat_speed AS BatSpeed,
        shape.attack_angle AS AttackAngle, 
        shape.vba AS VBA,
        shape.ttc AS TTC, 
        mlb.release_speed AS ReleaseSpeed,
        mlb.pitch_type_encoded AS PitchType,
        mlb.pitch_type_specific_encoded AS PitchTypeSpecific,
        mlb.pitcher_hand AS PitcherHand,
        mlb.batter_hand AS BatterHand,
        mlb.plate_x AS PitchX,
        mlb.plate_z AS PitchZ,
        mlb.description AS Outcome,
        mlb.sz_top AS ZoneTop,
        mlb.sz_bot AS ZoneBot,

        -- Outcomes
        mlb.hard_hit_encoded AS HardHit,
        mlb.contact_encoded AS Contact,
        mlb.launch_speed AS ExitVelocity,
        mlb.launch_angle AS LaunchAngle        

        FROM swing_shape_df AS shape

        INNER JOIN mlb_data AS mlb
        ON shape.batter = mlb.batter

        -- Only want swings
        WHERE mlb.description IN ('hit_into_play', 'foul', 'swinging_strike', 
        'foul_tip', 'swinging_strike_blocked')
        """

pitch_swing_df = pysql(query)

### Remove rows with missing values that are needed and invalid specific pitch types
pitch_swing_df = pitch_swing_df.dropna(subset = ['PitchType', 'ReleaseSpeed', 'PitchX', 'PitchZ'])
pitch_swing_df = pitch_swing_df[pitch_swing_df['PitchTypeSpecific'] != 8]
pitch_swing_df.loc[pitch_swing_df['Outcome'] == 'foul', ['ExitVelocity', 'LaunchAngle']] = None

### Convert pitch x and z distances from feet to inches
pitch_swing_df['PitchX'] = pitch_swing_df['PitchX'] * 12
pitch_swing_df['PitchZ'] = pitch_swing_df['PitchZ'] * 12
pitch_swing_df['ZoneTop'] = pitch_swing_df['ZoneTop'] * 12
pitch_swing_df['ZoneBot'] = pitch_swing_df['ZoneBot'] * 12

### Assign pitch zones using coordinate data
pitch_swing_df['PitchZone'] = pitch_swing_df.apply(lambda row: ZoneSections(row, 8), axis = 1)

### Assign batted ball types
pitch_swing_df = hit_class(pitch_swing_df)

### Save output
pitch_swing_filename = "ModelData.csv"
pitch_swing_df.to_csv(f"data/{pitch_swing_filename}", index = False)
print(f'Full modeling data saved as {pitch_swing_filename}')