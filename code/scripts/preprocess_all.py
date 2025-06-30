#!/usr/bin/env python3
import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1) Metadata mapping
folder_metadata = {
    "LOC1":   {"Location": "Lausanne",   "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop"},
    "LOC2":   {"Location": "Leuven",     "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop"},
    "LOC3":   {"Location": "Singapore",  "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop (AWS)"},
    # "OW":     {"Location": "Lausanne",   "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop"},
    "RPI":    {"Location": "Lausanne",   "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Raspberry Pi"},
    "CL-FF":  {"Location": "Leuven",     "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop"},
    "GOOGLE": {"Location": "Leuven",     "Resolver": "Google",     "Client": "Firefox",    "Platform": "Desktop"},
    "CLOUD":  {"Location": "Leuven",     "Resolver": "Cloudflare", "Client": "Firefox",    "Platform": "Desktop"},
}


def json_to_rows(json_data, metadata):
    """
    Turn each JSON trace into one row dict with a fixed-length packet list.
    """
    rows = []
    for pcap_file, pcap_data in json_data.items():
        sent_i = recv_i = 0
        counts = []
        for order in pcap_data["order"]:
            if order == 1:
                counts.append(pcap_data["sent"][sent_i])
                sent_i += 1
            else:
                counts.append(-pcap_data["received"][recv_i])
                recv_i += 1
        # Truncate or pad to exactly 128
        counts = (counts[:128] + [0]*128)[:128]
        if not counts:
            continue

        rows.append({
            "Website":        int(pcap_file[:-5]),
            "Location":       metadata["Location"],
            "Resolver":       metadata["Resolver"],
            "Client":         metadata["Client"],
            "Platform":       metadata["Platform"],
            "packet_counts":  counts
        })
    return rows


def main():
    all_rows = []

    # 2) Load every JSON across all folders
    for folder, meta in folder_metadata.items():
        for path in glob.glob(f"../../dataset/{folder}/*.json"):
            with open(path) as f:
                jd = json.load(f)
            all_rows.extend(json_to_rows(jd, meta))

    # 3) Build initial DataFrame
    df = pd.DataFrame(all_rows)
    print("Columns after load:", df.columns.tolist())

    # 4) Expand packet_counts → 128 columns
    packet_df = pd.DataFrame(
        df["packet_counts"].tolist(),
        columns=[str(i) for i in range(128)]
    )
    df_expanded = pd.concat(
        [df.drop(columns=["packet_counts"]), packet_df], axis=1)
    print("Columns after expand:", df_expanded.columns.tolist())

    # 5) Scale the 0–127 columns
    scaler = StandardScaler()
    cols_to_scale = [str(i) for i in range(128)]
    df_expanded[cols_to_scale] = scaler.fit_transform(
        df_expanded[cols_to_scale])

    # 6) Reset index and save
    df_final = df_expanded.reset_index(drop=True)
    df_final.to_csv(
        "../../dataset/processed/LOC1-LOC2-LOC3-RPI-CL-GOOGLE-CLOUD-processed_dataset.csv", index_label="index")
    print("Saved processed_dataset.csv with shape", df_final.shape)


if __name__ == "__main__":
    main()
