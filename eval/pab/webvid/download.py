import csv
import os
import time

import requests
import tqdm


def read_csv(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    data = data[1:]
    print(f"Read {len(data)} rows from {csv_file}")
    return data


def select_csv(data, min_text_len, min_vid_len, select_num):
    results = []
    assert 0 <= min_vid_len <= 60
    min_vid_len_str = f"PT00H00M{min_vid_len:02d}S"
    for d in data:
        # [id, link, duration, page, text]
        if d[2] < min_vid_len_str:
            continue
        token_num = len(d[4].split(" "))
        if token_num < min_text_len:
            continue
        results.append(d)
        if len(results) == select_num:
            break
    return results


def save_data_list(data, save_path):
    with open(save_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "link", "duration", "page", "text"])
        for d in data:
            writer.writerow(d)


def download_video(data, save_path):
    os.makedirs(save_path, exist_ok=True)
    for d in tqdm.tqdm(data):
        url = d[1]
        video_path = os.path.join(save_path, f"{d[0]}.mp4")
        while True:
            try:
                r = requests.get(url, stream=True)
                with open(video_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                break
            except ConnectionError:
                time.sleep(1)
                print(f"Failed to download {url}, retrying...")
                continue
        time.sleep(0.1)


if __name__ == "__main__":
    data = read_csv("./datasets/webvid.csv")
    selected_data = select_csv(data, 20, 5, 500)
    save_data_list(selected_data, "./datasets/webvid_selected.csv")
    download_video(selected_data, "./datasets/webvid")
