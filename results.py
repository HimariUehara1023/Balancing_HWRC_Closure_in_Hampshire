"""
module containing functions for creating excel tables from a given results file
and functions for creating excel tables after creating the corresponding results
"""

from model import *
from utils import *
import os
import shutil


def Assignment_excel(users_and_facs_df, travel_dict, users, facs, distance_dict,
                        output_filename="AssignmentResults.xlsx", budget_factor=1.0, strict_assign_to_one=False,
                        cap_factor=1.0, cutoff=0, max_access=False, main_threads=1, main_tolerance=5e-3,
                        main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                        post_threads=1, post_tolerance=0.0, post_print_sol=False, post_log_file=None,
                        post_preqlinearize=-1):

    # Step 1: Solve model
    is_feasible, results = solve_model_naively(users_and_facs_df, travel_dict, users, facs, budget_factor, strict_assign_to_one,
                                                cap_factor, cutoff, max_access, main_threads, main_tolerance,
                                                main_time_limit, main_print_sol, main_log_file, main_preqlinearize,
                                                post_threads, post_tolerance, post_print_sol, post_log_file,
                                                post_preqlinearize)

    if not is_feasible:
        print("Model infeasible. No output generated.")
        return

    # Step 2: Build assignment dataframe
    assignment_records = []
    for i, j in results["solution_details"]["assignment"].items():
        if j is None:
            print("Exist user who is not assigned to any facility")
            continue
        user_type = users_and_facs_df.loc[i, 'regional spatial type']
        fac_type = users_and_facs_df.loc[j, 'regional spatial type']
        distance = distance_dict[i][j]
        assignment_records.append({
            'user': i,
            'facility': j,
            'distance': distance,
            'user_type': user_type,
            'facility_type': fac_type
        })

    assignment_df = pd.DataFrame(assignment_records)

    # Step 3: Compute medians for user-to-assigned-facility distances
    # Compute medians
    medians = {
        "rural_users_to_facilities": assignment_df[assignment_df['user_type'] == 'rural']['distance'].median(),
        "urban_users_to_facilities": assignment_df[assignment_df['user_type'] == 'urban']['distance'].median(),
        "users_to_rural_facilities": assignment_df[assignment_df['facility_type'] == 'rural']['distance'].median(),
        "users_to_urban_facilities": assignment_df[assignment_df['facility_type'] == 'urban']['distance'].median()
    }
    results["median_analysis"] = medians

    # Optional: save to Excel
    output_dir = os.path.join(os.getcwd(), "own_results")
    os.makedirs(output_dir, exist_ok=True)
    full_output_path = os.path.join(output_dir, output_filename)

    summary_df = pd.DataFrame({
        "Median distance": [
            "rural users - facilities",
            "urban users - facilities",
            "- rural facilities",
            "- urban facilities"
        ],
        "Distance (miles)": [
            medians["rural_users_to_facilities"],
            medians["urban_users_to_facilities"],
            medians["users_to_rural_facilities"],
            medians["users_to_urban_facilities"]
        ]
    })
    
    # Step 3.1: Add percentiles (25th, 50th, 75th) based on all distance values
    percentiles_needed = [25, 50, 75]
    distances_all = assignment_df['distance']
    percentile_values = [np.percentile(distances_all, p) for p in percentiles_needed]

    percentile_rows = pd.DataFrame({
        "Median distance": [f"All users: {p}th percentile" for p in percentiles_needed],
        "Distance (miles)": percentile_values
    })

    # Append percentile rows to summary_df
    summary_df = pd.concat([summary_df, percentile_rows], ignore_index=True)

    # Step 4: Compute utilization for each assigned facility
    utilization_records = []
    for j in assignment_df['facility'].unique():
        assigned_users = assignment_df[assignment_df['facility'] == j]['user']
        numerator = sum(
            users_and_facs_df.loc[i, 'population'] * travel_dict[i][j]
            for i in assigned_users
        )
        denominator = users_and_facs_df.loc[j, 'capacity']
        utilization = numerator / denominator if denominator > 0 else None
        utilization_records.append({
            'facility': j,
            'utilization': utilization,
            'assigned_user_count': len(assigned_users),
            'capacity': denominator
        })

    utilization_df = pd.DataFrame(utilization_records)

    # Step X1: Create summary DataFrame
    util_stats_df = pd.DataFrame({
        'Statistic': ['Variance', 'Max', 'Min', 'Median'],
        'Utilization': [
            utilization_df['utilization'].var(),
            utilization_df['utilization'].max(),
            utilization_df['utilization'].min(),
            utilization_df['utilization'].median()
        ]
    })


    # Step Y: Compute savings from closed facilities
    tier_saving_dict = get_tier_saving_dict()
    all_facs = set(users_and_facs_df.index)
    open_facs = set(results["solution_details"]["open_facs"])
    closed_facs = list(all_facs - open_facs)
    
    closed_fac_records = []
    total_saving = 0
    
    for j in closed_facs:
        tier = users_and_facs_df.loc[j, 'Tier']
        saving = tier_saving_dict.get(tier, 0)
        closed_fac_records.append({
            'facility_id': j,
            'tier': tier,
            'saving (£)': saving
        })
        total_saving += saving
    
    closed_fac_records.append({'facility_id': 'Total', 'tier': '', 'saving (£)': total_saving})
    saving_df = pd.DataFrame(closed_fac_records)

    # Step Z: List of closed facilities with names
    closed_fac_df = pd.DataFrame({
        'facility_id': closed_facs,
        'facility_name': [users_and_facs_df.loc[j, 'Facility_name'] for j in closed_facs],
        'tier': [users_and_facs_df.loc[j, 'Tier'] for j in closed_facs]
    })

    with pd.ExcelWriter(full_output_path) as writer:
        assignment_df.to_excel(writer, sheet_name="Assignments", index=False)
        summary_df.to_excel(writer, sheet_name="Medians Summary", index=False)
        utilization_df.to_excel(writer, sheet_name="Utilization Summary", index=False)
        util_stats_df.to_excel(writer, sheet_name="Utilization Stats", index=False)
        saving_df.to_excel(writer, sheet_name="Saving Summary", index=False)
        closed_fac_df.to_excel(writer, sheet_name="Closed Facilities", index=False)

    print("Success. Results saved to:", full_output_path)

def optimal_results(users_and_facs_df, travel_dict, users, facs, distance_dict,
                    total_facility_count=26,
                    strict_assign_to_one=False, cap_factor=1.0,
                    max_access=False, main_threads=1, main_tolerance=5e-3,
                    main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                    post_threads=1, post_tolerance=0.0, post_print_sol=False, post_log_file=None,
                    post_preqlinearize=-1):
    """
    Runs Assignment_excel for x in 0 to 9, each time reducing the number of facilities by x.
    Output files are named as Result_O_{x}.xlsx.
    """
    base_dir = os.getcwd()
    dirs_to_save = [
        os.path.join(base_dir, "own_results", "All Assignment")
    ]
    for d in dirs_to_save:
        os.makedirs(d, exist_ok=True)

    for x in range(0, 10):
        budget_factor = (total_facility_count - x) / total_facility_count
        filename = f"Result_O_{x}.xlsx"

        # Primary output path: data/All Assignment
        main_path = os.path.join(dirs_to_save[0], filename)

        Assignment_excel(users_and_facs_df=users_and_facs_df, travel_dict=travel_dict, users=users, facs=facs,
                         budget_factor=budget_factor, cutoff=0.0, distance_dict=distance_dict,
                         output_filename=main_path)

def non_optimal_results(users_and_facs_df, travel_dict, users, facs, distance_dict,
                        strict_assign_to_one=False, cap_factor=1.0,
                        max_access=False, main_threads=1, main_tolerance=5e-3,
                        main_time_limit=20000, main_print_sol=False, main_log_file=None, main_preqlinearize=-1,
                        post_threads=1, post_tolerance=0.0, post_print_sol=False, post_log_file=None,
                        post_preqlinearize=-1):
    """
    For x in 1 to 9, remove x Tier 3, 4 facilities and solve the model with full budget (budget_factor=1.0).
    Each run saves result to Result_NO_{x}.xlsx
    """
    base_dir = os.getcwd()
    dirs_to_save = [
        os.path.join(base_dir, "own_results", "All Assignment")
    ]
    for d in dirs_to_save:
        os.makedirs(d, exist_ok=True)

    for x in range(1, 10):
        facs_reduced = closure_sequence(facs, x)
        filename = f"Result_NO_{x}.xlsx"

        # Primary output path: data/All Assignment
        main_path = os.path.join(dirs_to_save[0], filename)

        Assignment_excel(users_and_facs_df=users_and_facs_df, travel_dict=travel_dict, users=users, facs=facs_reduced,
                         budget_factor=1, cutoff=0.0, distance_dict=distance_dict,
                         output_filename=main_path)