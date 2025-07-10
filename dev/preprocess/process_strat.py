import yaml
import sys
import time
import pandas as pd
import os
from glob import glob
import gc
from multiprocessing import Pool, cpu_count
import shutil

# Update your sys.path to include the project directory
sys.path.append('/home/cheddarjackk/Developer')

# Import the functions from the modules
from VAmodel.dev.preprocess.strat import process_tick_data
from VAmodel.dev.preprocess.dat_handler import load_data, save_data
# from f__modules.valuemarker import value_marker

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def clear_marked_dir(dir_path):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            shutil.rmtree(item_path)

def process_single_session(file_path):
    print(f'Processing file: {file_path}')

    # Load tick data
    tick_data = load_data(file_path)

    # Process tick data
    tick_data = process_tick_data(tick_data, config['parameters'])
    # tick_data = generate_signals(tick_data, config['parameters'])
    # tick_data = simulate_trades(tick_data, config['parameters'])

    # trades_data = extract_trades(tick_data)

    # Optionally, save processed tick data for this chunk
    indicatored_data_dir = os.path.join(config['data']['indicatored_data_dir'], os.path.basename(file_path))
    save_data(tick_data, indicatored_data_dir)

    # # Free up memory
    # del tick_data
    # gc.collect()

    # return trades_data
    return tick_data

if __name__ == "__main__":
    start_time = time.time()
    print('Program started')

    # Load paths
    config = load_config('/home/cheddarjackk/Developer/VAmodel/dev/config.yaml')
    data_dir = '/home/cheddarjackk/Developer/VAmodel/data/data_edit/3_day_data_preprocessed_ext'

    # List all parquet files in the directory
    all_files = glob(os.path.join(data_dir, 'session_*.parquet'))

    # Define the start and end session IDs
    start_session_id = 130
    end_session_id = 190
    # Clear old files
    clear_marked_dir(config['data']['indicatored_data_dir'])
    
    # Filter files based on session IDs in file names
    selected_files = []
    for file in all_files:
        file_name = os.path.basename(file)
        parts = file_name.replace('.parquet', '').split('_')

        try:
            session_id = int(float(parts[1]))
        except (IndexError, ValueError) as e:
            print(f'Filename {file_name} does not match expected pattern or contains invalid session ID. Skipping.')
            continue

        # Check if the session_id is within the desired range
        if start_session_id <= session_id <= end_session_id:
            selected_files.append(file)

    # Sort the files by session ID for consistent processing
    selected_files.sort(key=lambda x: int(float(os.path.basename(x).split('_')[1])))

    # Ensure the config is pickleable (avoid issues with multiprocessing)
    import copy
    config_for_multiprocessing = copy.deepcopy(config)

    num_workers = 8 #cpu_count()  # Use the number of CPU cores available
    print(f'Using {num_workers} worker processes.')

    with Pool(processes=num_workers) as pool:
        # Map the processing function to the list of selected files
        all_trades_data = pool.map(process_single_session, selected_files)

    # Concatenate all trades data into a single DataFrame
    if all_trades_data:
        trades_data_combined = pd.concat(all_trades_data, ignore_index=True)
    else:
        trades_data_combined = pd.DataFrame()   

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Program ended. Total elapsed time: {elapsed_time:.2f} seconds')
