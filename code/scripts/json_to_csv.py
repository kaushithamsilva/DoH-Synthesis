import json
import pandas as pd
import glob
import numpy as np


def json_to_dic(json_data: dict, location: str):
    """
    Converts a JSON object containing PCAP data into a Pandas DataFrame.

    Args:
        json_data (dict): The JSON object representing the PCAP data.

    Returns:
        pandas.DataFrame: The DataFrame containing the extracted data.
    """

    data = []
    for pcap_file, pcap_data in json_data.items():
        packet_counts = []
        sent_count = 0
        received_count = 0
        for order in (pcap_data["order"]):
            if order == 1:
                # Positive for sent
                packet_counts.append(pcap_data['sent'][sent_count])
                sent_count += 1
            else:
                # Negative for Received
                packet_counts.append(-pcap_data['received'][received_count])
                received_count += 1

        data.append({
            "Location": location,
            "Website": int(pcap_file[:-5]),
            "Packet Counts": packet_counts,

        })

    return data


def fix_packet_count_length(df):
    # Calculate the lengths of each "Packet Counts"
    lengths = df["Packet Counts"].apply(len)

    # Find the length at the 80th percentile
    length_80th_percentile = int(np.percentile(lengths, 80))

    print(f"length: {length_80th_percentile}")
    # Pad the "Packet Counts" with zeros to the 80th percentile length
    df["Packet Counts"] = df["Packet Counts"].apply(
        lambda counts: counts + [0] * (length_80th_percentile - len(counts)) if len(
            counts) < length_80th_percentile else counts[:length_80th_percentile]
    )

    return df


def main():
    folders = ['LOC1', 'LOC2', 'LOC3']
    concat_data: list[dict] = []

    for folder in folders:
        for file in glob.glob(f"../../dataset/{folder}/*.json"):
            with open(file, "r") as f:
                json_data = json.load(f)
                concat_data.extend(json_to_dic(json_data, folder))

    df = fix_packet_count_length(pd.DataFrame(concat_data))

    # Expanding the Packet Counts column into a DataFrame
    packet_counts_expanded = pd.DataFrame(df['Packet Counts'].tolist())

    # Renaming the columns with a range from 0 to the max length of the lists in Packet Counts
    packet_counts_expanded.columns = [
        str(i) for i in range(packet_counts_expanded.shape[1])]

    # Concatenating the expanded packet counts back to the original DataFrame
    df_expanded = pd.concat(
        [df.drop(columns=['Packet Counts']), packet_counts_expanded], axis=1)

    df_expanded.sort_values(by=['Location', 'Website'], inplace=True)
    df_expanded.to_csv('../../dataset/processed/Locations.csv', index=False)


if __name__ == '__main__':
    main()
