import pandas as pd
import json
import random
import torch
import os
    

def main():
    chain = 'polygon'

    labels = pd.read_csv('../../_data/data/labels.csv').query('Chain == @chain')

    # Use three-class as an example.
    n = 3
    category_counts = labels['Category'].value_counts()
    select_class = list(category_counts.head(n).index)
    category_to_label = {category: i for i, category in enumerate(select_class)}
    labels['Category'] = labels['Category'].map(category_to_label)

    labels_select = list(labels.query('Category < @n').Category.values)
    labels_select_df = labels.query('Category < @n').reset_index(drop = True)

    # create timestamps
    stats = []
    for addr in labels_select_df.Contract.values:
        tx = pd.read_csv(f'../../_data/data/transactions/{chain}/{addr}.csv')
        first_timestamp = tx['timestamp'].min()
        stats.append({'address': addr, 'first_timestamp': first_timestamp})
        
    timestamps = pd.DataFrame(stats)

    # create index mapping
    all_address = list(labels_select_df.Contract.values)
    index_mapping = {addr: idx for idx, addr in enumerate(all_address)}

    timestamps['addr_index'] = timestamps['address'].apply(lambda x: index_mapping[x])
    timestamps = timestamps.sort_values(by = 'first_timestamp')

    train_num = int(len(list(timestamps['addr_index'].values)) * 0.8)
    test_num = len(list(timestamps['addr_index'].values)) - train_num

    train_index = list(timestamps['addr_index'].values)[:train_num]
    test_index = list(timestamps['addr_index'].values)[train_num:]

    file_name = f'../../_data/GoG/node/{chain}_train_index_{n}.txt'
    with open(file_name, 'w') as file:
        for item in train_index:
            file.write(f"{item}\n")

    file_name = f'../../_data/GoG/node/{chain}_test_index_{n}.txt'
    with open(file_name, 'w') as file:
        for item in test_index:
            file.write(f"{item}\n")

if __name__ == "__main__":
    main()