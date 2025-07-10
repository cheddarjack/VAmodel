import yaml
import pandas as pd
import os
import glob
import shutil

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clear_marked_dir(dir_path):
    """Remove all files and directories in the given path."""
    if not os.path.exists(dir_path):
        return
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            shutil.rmtree(item_path)

def load_marker_mappings(excel_file):
    """
    Load marker mappings from an Excel file.
    Expects columns: 'session', 'start', and 'end'.
    """
    df = pd.read_excel(excel_file)
    mappings = []
    for idx, row in df.iterrows():
        session = row.get('session')
        start_marker = row.get('Start')
        end_marker = row.get('End')
        if pd.isna(session):
            print(f"Row {idx}: Session is NaN, skipping.")
            continue
        if pd.isna(start_marker) or pd.isna(end_marker):
            print(f"Row {idx}: Start or end marker is NaN, skipping.")
            continue
        # Convert session to string (assumed to be integer-like)
        session = str(int(session))
        mappings.append({
            'session': session,
            'start_marker': start_marker,
            'end_marker': end_marker
        })
    return mappings



def process_session(mapping, data_dir, output_dir):
    """
    For a given mapping (session, start_marker, end_marker):
    - Finds the session parquet file from data_dir matching the pattern "session_{session}_*.parquet".
    - Marks rows where 'bar_id' is between start_marker and end_marker (inclusive).
    - Saves the marked rows as a new parquet file in output_dir.
    """
    session = mapping['session']
    start_marker = mapping['start_marker']
    end_marker = mapping['end_marker']
    
    # Use glob to find file that matches the session pattern.
    file_pattern = os.path.join(data_dir, f"session_{session}_*.parquet")
    file_list = glob.glob(file_pattern)
    if not file_list:
        print(f"File for session {session} not found with pattern {file_pattern}. Skipping.")
        return
    # Use the first match
    file_path = file_list[0]
    
    df = pd.read_parquet(file_path)
    if 'bar_id' not in df.columns:
        print(f"'bar_id' column not found in file for session {session} ({file_path}). Skipping.")
        return

    # Select rows where 'bar_id' is between start_marker and end_marker (inclusive)
    marked_subset = df[(df['bar_id'] >= start_marker) & (df['bar_id'] <= end_marker)].copy()
    
    if marked_subset.empty:
        print(f"No rows found in session {session} between markers {start_marker} and {end_marker}.")
        return
        
    # Mark these rows with a new column 'marker' set to 1
    marked_subset['marker'] = 1

    # drop all all rows that have the same value in the 'Last' column as the previous row
    marked_subset = marked_subset.loc[marked_subset['Last'].ne(marked_subset['Last'].shift())].reset_index(drop=True)

    print(f"Max value in 'Last' column: {marked_subset['Last'].max()}")
    print(f"Min value in 'Last' column: {marked_subset['Last'].min()}")

    num_rows = len(marked_subset)
    output_filename = f"session_{session}_{start_marker}_{end_marker}_{num_rows}.parquet"
    output_path = os.path.join(output_dir, output_filename)
    marked_subset.to_parquet(output_path, index=False)
    print(f"Saved {num_rows} rows for session {session} (markers: {start_marker}-{end_marker}) as {output_path}")

def main():
    config_path = "/home/cheddarjackk/Developer/VAmodel/dev/config.yaml"
    config = load_config(config_path)
    
    data_dir = config['data']['indicatored_data_dir']
    output_dir = config['data']['inference_data_dir']
    excel_file = '/home/cheddarjackk/Developer/VAmodel/dev/Inference_marker.xlsx'
    
    # Clear and recreate the output directory.
    clear_marked_dir(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    mappings = load_marker_mappings(excel_file)
    if not mappings:
        print("No valid marker mappings found. Exiting.")
        return
    
    for mapping in mappings:
        process_session(mapping, data_dir, output_dir)

if __name__ == "__main__":
    main()