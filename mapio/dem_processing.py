import os
from whitebox import WhiteboxTools
from tqdm import tqdm
wbt = WhiteboxTools()

wbt.verbose = False

base_path = "./data/05m_chips"
elevation_dir = "elevation"
slope_dir = "slope"
slope_deg_dir = "slope_deg"
fill_dep_dir = "fill_dep"
flow_acc_dir = "flow_acc"
twi_dir = "twi"

dem_base_path = os.path.join(base_path, elevation_dir)
slope_base_path = os.path.join(base_path, slope_dir)
slope_deg_base_path = os.path.join(base_path, slope_deg_dir)
fill_base_dep_path = os.path.join(base_path, fill_dep_dir)
flow_base_acc_path = os.path.join(base_path, flow_acc_dir)
twi_base_path = os.path.join(base_path, twi_dir)

os.makedirs(slope_deg_base_path, exist_ok=True)
os.makedirs(fill_base_dep_path, exist_ok=True)
os.makedirs(flow_base_acc_path, exist_ok=True)
os.makedirs(twi_base_path, exist_ok=True)

for image in tqdm(os.listdir(slope_base_path)):

    dem_path = os.path.abspath(os.path.join(dem_base_path, image))
    slope_deg_path = os.path.abspath(os.path.join(slope_deg_base_path, image))
    fill_dep_path = os.path.abspath(os.path.join(fill_base_dep_path, image))
    flow_acc_path = os.path.abspath(os.path.join(flow_base_acc_path, image))
    twi_path = os.path.abspath(os.path.join(twi_base_path, image))

    wbt.fill_depressions(dem_path, fill_dep_path, fix_flats=True)
    wbt.fd8_flow_accumulation(fill_dep_path, flow_acc_path)
    wbt.slope(dem_path, slope_deg_path)
    wbt.wetness_index(flow_acc_path, slope_deg_path, twi_path)
