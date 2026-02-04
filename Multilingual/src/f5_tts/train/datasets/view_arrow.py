import argparse
from pathlib import Path
import pyarrow.ipc
import pandas as pd
import os

# 设置 pandas 显示选项，防止长文本被截断
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

def inspect_arrow_file(arrow_path: str, num_rows: int = 5):
    """
    读取并打印 Arrow 文件的前 N 行内容。
    """
    arrow_file = Path(arrow_path)
    if not arrow_file.exists():
        print(f"❌ Error: File not found at '{arrow_file}'")
        return

    print(f"==============================================")
    print(f"🔎 Inspecting Arrow file: {arrow_file.name}")
    print(f"==============================================")

    try:
        # 使用 memory_map 高效读取，不消耗大量内存
        with pyarrow.memory_map(str(arrow_file), 'r') as source:
            # 打开 IPC 文件读取器
            reader = pyarrow.ipc.RecordBatchFileReader(source)
            
            # 计算总行数
            total_rows = reader.num_rows
            print(f"Total rows in file: {total_rows}\n")
            
            if total_rows == 0:
                print("File is empty.")
                return

            # 读取第一个 RecordBatch
            first_batch = reader.get_batch(0)
            
            # 将 Batch 转换为 pandas DataFrame 以便漂亮地打印
            df = first_batch.to_pandas()
            
            # 只显示前 N 行
            display_df = df.head(num_rows)
            
            print(display_df.to_string())
            
            # 额外检查第一条数据的音频路径是否存在
            if not display_df.empty:
                first_path = display_df.iloc[0]['audio_path']
                print(f"\nChecking first audio path:")
                print(f"  Path: {first_path}")
                print(f"  Exists: {'✅ Yes' if os.path.exists(first_path) else '❌ No'}")


    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the first few rows of an Arrow IPC file.")
    
    parser.add_argument("--arrow_file", type=str,  help="Path to the .arrow file to inspect.")
    parser.add_argument("-n", "--num_rows", type=int, default=5, help="Number of rows to display.")
    
    args = parser.parse_args()
    
    inspect_arrow_file(args.arrow_file, args.num_rows)