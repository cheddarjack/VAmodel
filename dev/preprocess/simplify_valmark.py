import os
import pandas as pd
import shutil

def clear_marked_dir(dir_path):

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            shutil.rmtree(item_path)

def main():
# Define the source and destination directories
    source_dir = '/home/cheddarjackk/Developer/VAmodel/data/data_training/marked-final'
    dest_dir = '/home/cheddarjackk/Developer/VAmodel/data/data_training/marked-final'

    # Create the destination directory if it does not exist
    os.makedirs(dest_dir, exist_ok=True)

    # Define the allowed absolute values
    allowed_values = {1, 2, 3, 4, 5}

    # Iterate through each file in the source directory
    for file_name in os.listdir(source_dir):
        # Process only parquet files
        if file_name.endswith('.parquet'):
            file_path = os.path.join(source_dir, file_name)
            try:
                # Read the parquet file into a DataFrame
                df = pd.read_parquet(file_path)
                
                # Filter rows where the absolute value of 'Last' is in the allowed set
                filtered_df = df[df['valuemarker'].abs().isin(allowed_values)]
                
                output_file_name = file_name.replace('.parquet', '_altered.parquet')
                output_path = os.path.join(dest_dir, output_file_name)
                
                # Save the filtered DataFrame back to a parquet file and
                filtered_df.to_parquet(output_path)
                print(f"Processed and saved: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == "__main__":
    main()
