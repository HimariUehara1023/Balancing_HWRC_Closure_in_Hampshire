"""
module containing command for running assignment results
"""

from utils import *
from model import *
from results import *

# Input basic data
users_and_facs_df = load_users_and_facs(users_and_facs_filename="Sector dataset.xlsx", abs_path=None)
users, facs = load_users_with_facs(users_and_facs_df)

# Output the Distance_dict
distance_dict = create_distance_dict(users_and_facs_df, users, facs)
save_distance_dict(distance_dict, distance_dict_filename="distance_dict.json.pbz2")

# Output the Distance_dict
travel_dict = create_travel_dict(users_and_facs_df, users, facs)
save_travel_dict(travel_dict, travel_dict_filename="travel_dict.json.pbz2")

users, facs, users_and_facs_df, travel_dict, distance_dict = load_complete_data(users_and_facs_filename="Sector dataset.xlsx", 
                                                                   travel_dict_filename="travel_dict.json.pbz2", 
                                                                   distance_dict_filename="distance_dict.json.pbz2")

if __name__ == "__main__":

    print("\n====== Start Running Optimal Results ======")
    optimal_results(
        users_and_facs_df=users_and_facs_df,
        travel_dict=travel_dict,
        users=users,
        facs=facs,
        distance_dict=distance_dict, 
        cap_factor=1.0
    )

    print("\n====== Start Running Non-Optimal Results ======")
    non_optimal_results(
        users_and_facs_df=users_and_facs_df,
        travel_dict=travel_dict,
        users=users,
        facs=facs,
        distance_dict=distance_dict, 
        cap_factor=1.0
    )

    print("\n Finish Success")