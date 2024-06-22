import csv


def load_eval_prompts(csv_file_path):
    prompts_dict = {}
    # Read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            prompts_dict[row['id']] = row['text']
    return prompts_dict

