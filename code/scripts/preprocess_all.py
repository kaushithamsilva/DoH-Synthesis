import glob
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1) Metadata mapping for each folder
folder_metadata = {
    "LOC1":   {"Location": "Lausanne",   "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop"},
    "LOC2":   {"Location": "Leuven",     "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop"},
    "LOC3":   {"Location": "Singapore",  "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop (AWS)"},
    "OW":     {"Location": "Lausanne",   "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop"},
    "RPI":    {"Location": "Lausanne",   "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Raspberry Pi"},
    "CL-FF":  {"Location": "Leuven",     "Resolver": "Cloudflare", "Client": "Cloudflare", "Platform": "Desktop"},
    "GOOGLE": {"Location": "Leuven",     "Resolver": "Google",     "Client": "Firefox",    "Platform": "Desktop"},
    "CLOUD":  {"Location": "Leuven",     "Resolver": "Cloudflare", "Client": "Firefox",    "Platform": "Desktop"},
}

# 2) JSON → flat dict, truncating/padding to exactly 128 packets


def json_to_rows(json_data, metadata, date, id_):
    rows = []
    for pcap_file, pcap_data in json_data.items():
        sent_i, recv_i = 0, 0
        counts = []
        for order in pcap_data["order"]:
            if order == 1:
                counts.append(pcap_data["sent"][sent_i])
                sent_i += 1
            else:
                counts.append(-pcap_data["received"][recv_i])
                recv_i += 1

        # truncate or pad to exactly 128
        counts = (counts[:128] + [0]*128)[:128]
        if not counts:
            continue

        rows.append({
            "Website":      int(pcap_file[:-5]),
            "Location":     metadata["Location"],
            "Resolver":     metadata["Resolver"],
            "Client":       metadata["Client"],
            "Platform":     metadata["Platform"],
            # we no longer store Folder, ID, Date if you don't need them
            "Packet Counts": counts
        })
    return rows


# 3) Loop folders → collect all rows
all_rows = []
for folder, meta in folder_metadata.items():
    for path in glob.glob(f"../dataset/{folder}/*.json"):
        fname = path.rsplit("/", 1)[-1]
        date_str, id_str = fname.split("_")[0][:8], fname.split("_")[
            1].split(".")[0]
        # parsed if you ever need it
        date = datetime.strptime(date_str, "%d-%m-%y")
        # id_ = int(id_str)

        with open(path) as f:
            jd = json.load(f)
        all_rows.extend(json_to_rows(jd, meta, date, id_str))

# 4) Build DataFrame
df = pd.DataFrame(all_rows)

# 5) Expand Packet Counts into 128 separate columns named '0'..'127'
pc_expanded = pd.DataFrame(df["Packet Counts"].tolist(),
                           columns=[str(i) for i in range(128)])
df = pd.concat([df.drop(columns=["Packet Counts"]), pc_expanded], axis=1)

# 6) Standard-scale the packet-count columns
scaler = StandardScaler()
df[[str(i) for i in range(128)]] = scaler.fit_transform(
    df[[str(i) for i in range(128)]])

# 7) (Optionally) reset index so you get a new integer index
df = df.reset_index(drop=True)

# 8) Save to CSV
df.to_csv("../../dataset/processed/processed_dataset.csv", index_label="index")

print("Processed dataset saved to processed_dataset.csv")
