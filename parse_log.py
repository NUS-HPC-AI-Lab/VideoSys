import re
import argparse
import os
import pandas as pd


def parse_log(file_path):
    imbalance_pattern = r'Total imbalance for this epoch:.*?\((\d+\.\d+)%\)'
    throughput_pattern = r'token throughput: (\d+\.\d+) token/s'
    flops_pattern = r'Final FLOPS: (\d+\.\d+)'  # Extracts the first number after 'Final FLOPS:'

    with open(file_path, 'r') as file:
        log_lines = file.read()

    imbalance_match = re.search(imbalance_pattern, log_lines)
    throughput_match = re.search(throughput_pattern, log_lines)
    flops_match = re.search(flops_pattern, log_lines)

    imbalance_percent = float(imbalance_match.group(1)) if imbalance_match else None
    token_throughput = float(throughput_match.group(1)) if throughput_match else None
    flops = float(flops_match.group(1)) if flops_match else None

    return imbalance_percent, token_throughput, flops


def main():
    parser = argparse.ArgumentParser(description='Parse log file for imbalance and throughput metrics.')
    parser.add_argument("--log_dir", type=str, required=True)
    args = parser.parse_args()
    results = []
    for dirpath, dirnames, filenames in os.walk(args.log_dir):
        if 'log.txt' in filenames:
            log_path = os.path.join(dirpath, 'log.txt')
            try:
                imbalance_percent, token_throughput, flops = parse_log(log_path)

                relative_path = os.path.relpath(dirpath, args.log_dir)

                results.append({
                    'experiment': relative_path.split('/')[0],  # baseline, dcp_inter, etc.
                    'run': relative_path.split('/')[-1],        # 000-OpenSora, etc.
                    'log_path': relative_path,
                    'imbalance_percent': imbalance_percent,
                    'token_throughput': token_throughput,
                    'flops': flops
                })

            except Exception as e:
                print(f"Error reading {log_path}: {e}")
    
    df = pd.DataFrame(results)
    df = df.sort_values(by=['run', 'experiment']).reset_index(drop=True)
    # print(df)
    save_path = os.path.join(args.log_dir, 'summary.csv')
    df.to_csv(save_path, index=False)
    print(f"Parsed results saved to '{save_path}'")


if __name__ == "__main__":
    main()
