"""
module for creating the exact figures and table included in the study
"""

from plotting import *

plot_figure1(
    folder_path="./own_results/All Assignment",
    output_path="./own_results/tables and figures/figure1.png",
    show_legend=False
)

plot_figure2(
    sector_xlsx="./data/Sector dataset.xlsx",
    geojson_path="./data/map_data/all_sectors.geojson",
    boundary_path="./data/Counties_and_Unitary_Authorities_April_2019_Ultra_Generalised_Boundaries_EW_2022_1790725369940793023.geojson",
    result_o9="./data/All Assignment/Result_O_9.xlsx",
    result_no9="./data/All Assignment/Result_NO_9.xlsx",
    out_dir="./own_results/tables and figures",
    use_basemap=True,
    dpi=200
)

plot_figure3(
    sector_path="./data/Sector dataset.xlsx",
    results_dir="./own_results/All Assignment",
    output_path="./own_results/tables and figures/figure3.png",
    require_urban=True,
    inland_prefixes=("RG", "GU", "SP"),
    assign_sheet="Assignments"
)

plot_figure4(
    folder_path="./own_results/All Assignment",
    output_path="./own_results/tables and figures/figure4.png",
    show_legend=False
)

build_table5(
    assign_dir="./own_results/All Assignment",
    tables_dir="./own_results/tables and figures",
    sector_path="./data/Sector dataset.xlsx",
    travel_dict_path="./data/travel_dict.json.pbz2",
    output_filename="table5.xlsx"
)
