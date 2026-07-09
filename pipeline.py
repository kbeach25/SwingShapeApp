from pybaseball import statcast, playerid_reverse_lookup
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandasql

warnings.filterwarnings('ignore')

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

df = getData('2025-03-27', '2025-09-28')