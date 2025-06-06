import os
import re
import csv
import argparse
import fnmatch # <-- 新增导入

all_deduped = set()  # 用于存储所有去重后的结果

def process_onednn_log(file_path, output_dir):
    """
    Extracts lines containing "exec,gpu" from onednn.verbose.log.
    """
    output_filename = os.path.join(output_dir, os.path.basename(file_path) + ".filtered.log")
    extracted_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                if "exec,gpu" in line:
                    extracted_lines.append(line)
        
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for line in extracted_lines:
                outfile.write(line)
        print(f"Processed '{file_path}'. Filtered log saved to '{output_filename}'")
        return True
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return False
    except Exception as e:
        print(f"Error processing '{file_path}': {e}")
        return False

def process_torch_profile(file_path, output_dir):
    output_filename = os.path.join(output_dir, os.path.basename(file_path) + ".parsed.csv")
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Step 1: 找到 header 和其对应字段位置
    header_line = None
    split_line = None
    split_line_idx = None

    for i, line in enumerate(lines):
        if 'Name' in line and 'Input Shapes' in line:
            header_line = line
            header_line_idx = i
            break
    
    if header_line is None:
        print(f"Header not found in {file_path}")
        return
    
    for j in range(header_line_idx + 1, len(lines)):
        if re.match(r'^----+\s+', lines[j]):
            split_line = lines[j]
            split_line_idx = j
            break

    if split_line_idx is None or split_line is None:
        print(f"Split line not found in {file_path}")
        return

    # 记录每一列的起始位置（通过空格判断）
    col_starts = [m.start() for m in re.finditer(r'\S+', split_line)]
    col_ends = col_starts[1:] + [len(split_line)]
    col_names = [header_line[start:end].strip() for start, end in zip(col_starts, col_ends)]

    # Step 2: 解析有效数据区段（直到下一个分隔线或统计信息）
    data_lines = []
    for line in lines[split_line_idx + 1:]:
        if re.match(r'^----+\s+', line):  # 第二个分割线出现，结束
            break
        if not line.strip():
            continue
        data_lines.append(line.rstrip('\n'))

    results = []
    for line in data_lines:
        fields = []
        for i, (start, end) in enumerate(zip(col_starts, col_ends)):
            fields.append(line[start:end].strip())
        row = dict(zip(col_names, fields))

        # 处理 Name（去除 <...> 模板部分）
        row['Name'] = re.sub(r'<.*', '', row['Name']).strip()

        results.append({
            'Name': row.get('Name', ''),
            'Input Shapes': row.get('Input Shapes', '[]')
        })

    # 去重 + 排序
    global all_deduped
    deduped = sorted(set((r['Name'], r['Input Shapes']) for r in results))
    all_deduped.update(deduped)  # 更新全局去重集合

    # 输出
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Input Shapes'])
        writer.writerows(deduped)

    print(f"Saved processed CSV to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Process log files in a specified folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the log files.")
    
    args = parser.parse_args()
    
    folder_path = args.folder_path
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found or is not a directory.")
        return

    # Create an output sub-directory
    output_dir = os.path.join(folder_path, "processed_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in: '{output_dir}'")

    processed_count = 0
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isfile(file_path): # Ensure it's a file
            if filename == "onednn.verbose.log":
                if process_onednn_log(file_path, output_dir):
                    processed_count += 1
            elif fnmatch.fnmatch(filename, "token_*_profile.txt"):
                if process_torch_profile(file_path, output_dir):
                    processed_count += 1
            # Add more conditions here if you have other file types to process
            # elif "some_other_pattern" in filename:
            #     process_other_file(file_path, output_dir)
    
    global all_deduped
    all_deduped_output_filename = os.path.join(output_dir, "all_deduped_results.csv")
    all_deduped = sorted(all_deduped)  # 将全局去重集合转换为列表并排序
    with open(all_deduped_output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Input Shapes'])
        writer.writerows(all_deduped)

    if processed_count == 0:
        print(f"No target files ('onednn.verbose.log', 'token_x_profile.txt') found in '{folder_path}'.")
    else:
        print(f"Finished processing. Processed {processed_count} file(s).")

if __name__ == "__main__":
    main()