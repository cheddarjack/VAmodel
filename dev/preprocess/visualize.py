import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.backends.backend_pdf import PdfPages
import shutil
import multiprocessing as mp
from PyPDF2 import PdfMerger

def clear_marked_dir(dir_path):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            shutil.rmtree(item_path)

def process_file(file_path, visual_dir):
    try:
        df = pd.read_parquet(file_path)
        if 'Last' not in df.columns or 'valuemarker' not in df.columns:
            print(f"Skipping {file_path}: Missing columns")
            return None
        # Constants
        LAST_COL = (0, 0, 0.01)
        VAL_COL = (0, 0.6, 0)
        FLAG_COL = (0.5, 0, 0)
        PRE_COL = (1, 0.38, 0.32)
        LAST_WIDTH = 3
        PRE_WIDTH = 2
        VAL_WIDTH = 1
        LAST_STYLE = '-'
        PRE_STYLE = ':'
        VAL_STYLE = '-'
        

        x_values = list(range(len(df)))
        fig, ax_left = plt.subplots(figsize=(30, 10))
        ax_left.add_line(plt.Line2D(x_values, df['Last'], color=LAST_COL , alpha=1, label='Last', lw=LAST_WIDTH, ls=LAST_STYLE))
        ax_left.set_xlabel('Data Points')
        ax_left.set_ylabel('Last')
        min_last, max_last = df['Last'].min(), df['Last'].max()
        ax_left.set_ylim([min_last, max_last])
        tick_interval = max(1, len(x_values) // 10)
        ax_left.set_xticks(range(0, len(x_values), tick_interval))
        ax_left.set_xticklabels(range(0, len(x_values), tick_interval), rotation=45)
        ax_left.tick_params(axis='x', labelsize=10, pad=5)

        
        ax_right = ax_left.twinx()
        ax_right.add_line(plt.Line2D(x_values, df['valuemarker'], color=VAL_COL, alpha=1, label='ValueMarker', lw=VAL_WIDTH, ls=VAL_STYLE))
        ax_right.set_ylabel('ValueMarker')
        ax_right.set_ylim([-5, 5])
        for y_val in [1,2,3,4,5,-1,-2,-3,-4,-5]:
            ax_right.axhline(y=y_val, linestyle=':', color='gray', linewidth=0.5)

        if 'predicted_value_0' in df.columns and df['predicted_value_0'].notna().any():
            ax_right.add_line(plt.Line2D(x_values, df['predicted_value_0'], color=PRE_COL, alpha=0.6, label='Predicted Value', lw=PRE_WIDTH, ls=PRE_STYLE))

        if 'predicted_value' in df.columns and df['predicted_value'].notna().any():
            ax_right.add_line(plt.Line2D(x_values, df['predicted_value'], color=PRE_COL, alpha=0.6, label='Predicted Value', lw=PRE_WIDTH, ls=PRE_STYLE))

        flag_mask = df['flag'] == 1
        if flag_mask.any():
            flag_x = [i for i, flag in enumerate(df['flag']) if flag == 1]
            flag_y = df.loc[flag_mask, 'valuemarker']
            flag_last_y = df.loc[flag_mask, 'Last']
        
            # Determine point sizes based on valuemarker values
            point_sizes = flag_y.abs().map({
                1: 100,  # Small
                2: 200,  # Medium
                3: 400, # Large
                4: 800,  # Even larger
                5: 1600  # Largest
            }).fillna(10)  # Default size for other values
        
            # Scatter points on the 'valuemarker' line
            # ax_right.scatter(flag_x, flag_y, color=FLAG_COL, s=point_sizes, zorder=5, label='Flagged (ValueMarker)', alpha=0.5, edgecolors='black')
        
            # Scatter points on the 'Last' line
            ax_left.scatter(flag_x, flag_last_y, color=FLAG_COL, s=point_sizes, zorder=5, label='Flagged (Last)', alpha=0.5, edgecolors='black')
            
        y_range = max_last - min_last
        if y_range > 0:
            ax_left.yaxis.set_major_locator(MultipleLocator(y_range / 10))
            ax_left.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_right.yaxis.set_major_locator(MultipleLocator(1))
        ax_right.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        ax_left.legend(loc='upper left')
        ax_right.legend(loc='upper right')
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        fig.suptitle(base_name, fontsize=16)
        fig.tight_layout()
        
        output_pdf = os.path.join(visual_dir, f"{base_name}.pdf")
        fig.savefig(output_pdf, dpi=200, bbox_inches='tight')
        # print(f"Created visualization PDF: {output_pdf}")

        output_png = os.path.join(visual_dir, f"{base_name}.png")
        plt.savefig(output_png, dpi=200, bbox_inches='tight')
        plt.close(fig)
        # print(f"Created visualization: {output_png}")

        return output_pdf
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    
def create_visuals(marked_dir, inf_dir, visual_dir):
    os.makedirs(visual_dir, exist_ok=True)
    all_parquet_paths = glob.glob(os.path.join(marked_dir, '**', '*.parquet'), recursive=True) + \
                        glob.glob(os.path.join(inf_dir, '**', '*.parquet'), recursive=True)
    
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.starmap(process_file, [(file_path, visual_dir) for file_path in all_parquet_paths])
    pool.close()
    pool.join()
    
    pdf_files = [pdf for pdf in results if pdf is not None]
    return pdf_files



def merge_pdfs(pdf_list, output_pdf):
    merger = PdfMerger()
    for pdf in pdf_list:
        merger.append(pdf)
    merger.write(output_pdf)
    merger.close()
    print(f"Created merged PDF: {output_pdf}")

def main():
    MARKED_DIR = '/home/cheddarjackk/Developer/VAmodel/data/data_training/marked-final/'
    INF_DIR = '/home/cheddarjackk/Developer/VAmodel/data/data_training/run-inference/'
    VISUAL_DIR = '/home/cheddarjackk/Developer/VAmodel/data/data_training/visuals/'
    
    # Clear the output directory if needed.
    clear_marked_dir(VISUAL_DIR)
    
    pdf_files = create_visuals(MARKED_DIR, INF_DIR, VISUAL_DIR)
    master_pdf = os.path.join(VISUAL_DIR, "master_visual.pdf")
    merge_pdfs(pdf_files, master_pdf)

if __name__ == '__main__':
    main()