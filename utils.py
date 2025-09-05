"""
Utility and help functions
"""
import os
from math import exp
import numpy as np
import json
import bz2
import _pickle as cPickle
import pandas as pd
from geopy import distance
import random

# preference

# functions for computing travel probabilities


def f_urban(d):
    return 1.0204453821123711 * exp(-0.09293721688764371 * d ** 1.1143658747407779)


def f_rural(d):
    return 1.026050886854989 * exp(-0.168984216127075 * d ** 1.080541705731028)


def get_tier_saving_dict():
    """
    Return a dictionary mapping each tier level to its estimated saving (£)
    if a facility of that tier is closed.
    These values are based on known cost estimates from Hampshire HWRC reports.
    """
    return {
        1: 271000,     # Tier 1: rough estimate
        2: 214000,     # Tier 2
        3: 157000,     # Tier 3: from document, avg of 7 facilities = £1.1M / 7
        4: 100000      # Tier 4: from document, avg of 5 facilities = £500k / 5
    }

# Distance dict

def create_distance_dict(users_and_facs_df, users, facs):
    """
    Create a dictionary of straight-line distances (in miles) between users and facilities.

    :param users_and_facs_df: DataFrame containing user & facility spatial info
    :param users: list of user IDs
    :param facs: list of facility IDs
    :return: distance_dict[i][j] = distance in miles between user i and facility j
    """
    distance_dict = {i: {} for i in users}
    
    for i in users:
        print('user', i)
        lat_1 = users_and_facs_df.at[i, 'centroid_lat']
        lon_1 = users_and_facs_df.at[i, 'centroid_lon']
        
        for j in facs:
            lat_2 = users_and_facs_df.at[j, 'rc_centroid_lat']
            lon_2 = users_and_facs_df.at[j, 'rc_centroid_lon']
            dist_mi = distance.distance((lat_1, lon_1), (lat_2, lon_2)).miles
            distance_dict[i][j] = dist_mi

    with open('distance_dict.json', 'w') as outfile:
        json.dump(distance_dict, outfile)

    return distance_dict


def save_distance_dict(distance_dict, distance_dict_filename, abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "\\data"
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    with bz2.BZ2File(abs_path + "\\" + distance_dict_filename, "w") as f:
        cPickle.dump(distance_dict, f)
    json_path = os.path.join(os.getcwd(), 'distance_dict.json')
    if os.path.exists(json_path):
        os.remove(json_path)
        print(f"Deleted temporary file: {json_path}")
    print(f"Compressed file saved to: {abs_path}")

# Travel dict

def create_travel_dict(users_and_facs_df, users, facs):
    """
    create the travel dictionary that specifies the probabilities for each travel combination
    :param users_and_facs_df: dataframe of the user and facility related input data
    :param users: list of the users used in the instance
    :param facs: list of the facilities used in the instance
    :return: travel dictionary that specifies the probabilities for each travel combination
    """
    travel_dict = {i: {} for i in users}
    for i in users:
        print('user', i)
        regiotype = users_and_facs_df.at[i, 'regional spatial type']
        lat_1 = users_and_facs_df.at[i, 'centroid_lat']
        lon_1 = users_and_facs_df.at[i, 'centroid_lon']
        for j in facs:
            lat_2 = users_and_facs_df.at[j, 'rc_centroid_lat']
            lon_2 = users_and_facs_df.at[j, 'rc_centroid_lon']
            dist = distance.distance((lat_1, lon_1), (lat_2, lon_2)).miles
            if regiotype == "urban":
                travel_dict[i][j] = f_urban(dist)
            else:
                travel_dict[i][j] = f_rural(dist)
    with open('travel_dict.json', 'w') as outfile:
        json.dump(travel_dict, outfile)

    return travel_dict

# functions for loading and saving data files

def save_travel_dict(travel_dict, travel_dict_filename, abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "\\data"
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)
    with bz2.BZ2File(abs_path + "\\" + travel_dict_filename, "w") as f:
        cPickle.dump(travel_dict, f)
    json_path = os.path.join(os.getcwd(), 'travel_dict.json')
    if os.path.exists(json_path):
        os.remove(json_path)
        print(f"Deleted temporary file: {json_path}")
    print(f"Compressed file saved to: {abs_path}")

# the prepareation function for load_input_data
def load_users_and_facs(users_and_facs_filename="users_and_facilities.xlsx", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "\\data"
    users_and_facs_df = pd.read_excel(abs_path + "\\" + users_and_facs_filename)
    return users_and_facs_df
def load_distance_dict(distance_dict_filename="distance_dict.json.pbz2", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "\\data"
    data = bz2.BZ2File(abs_path + "\\" + distance_dict_filename, 'rb')
    distance_dict = cPickle.load(data)
    distance_dict = {int(i): {int(j): distance_dict[i][j] for j in distance_dict[i]} for i in distance_dict}
    return distance_dict

def load_travel_dict(travel_dict_filename="travel_dict.json.pbz2", abs_path=None):
    if not abs_path:
        abs_path = os.getcwd() + "\\data"
    data = bz2.BZ2File(abs_path + "\\" + travel_dict_filename, 'rb')
    travel_dict = cPickle.load(data)
    travel_dict = {int(i): {int(j): travel_dict[i][j] for j in travel_dict[i]} for i in travel_dict}
    return travel_dict
    
def load_users_with_facs(users_and_facs_df):
    users = [int(i) for i in users_and_facs_df.index]
    facs = [i for i in users_and_facs_df.index if users_and_facs_df.at[i, 'capacity'] > 0]
    return users, facs

# the function used to input users_and_facs_df and travel_dict
def load_input_data(users_and_facs_filename="users_and_facilities.xlsx", travel_dict_filename="travel_dict.json.pbz2",
                    distance_dict_filename="distance_dict.json.pbz2",abs_path=None):
    users_and_facs_df = load_users_and_facs(users_and_facs_filename, abs_path)
    travel_dict = load_travel_dict(travel_dict_filename, abs_path)
    distance_dict = load_distance_dict(distance_dict_filename="distance_dict.json.pbz2", abs_path=None)
    
    return users_and_facs_df, travel_dict, distance_dict


def load_complete_data(users_and_facs_filename="users_and_facilities.xlsx", travel_dict_filename="travel_dict.json.pbz2", 
                       distance_dict_filename="distance_dict.json.pbz2", abs_path=None):
    users_and_facs_df, travel_dict, distance_dict = load_input_data(users_and_facs_filename, travel_dict_filename, abs_path)
    users, facs = load_users_with_facs(users_and_facs_df)
    return users, facs, users_and_facs_df, travel_dict, distance_dict

# Orderly delete facilities to find the effect of optimally choosing closed facilities
def closure_sequence(lst, num_to_remove):
    mapping = {
        1: 236,  2: 59,  3: 210,   4: 137,  5: 198,
        6: 215,  7: 16,  8: 200,   9: 38
    }
    if num_to_remove not in mapping:
        raise ValueError("No more facilities allowed")
    lst.remove(mapping[num_to_remove])
    return lst
