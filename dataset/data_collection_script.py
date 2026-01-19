import requests
import csv
from itertools import filterfalse, tee
from collections import defaultdict
import time
import os
from tqdm import tqdm
import pandas as pd

def unique_everseen(iterable, key=None):
    "List unique elements, preserving order. Remember all elements ever seen."
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

def remove_duplicates(input_list):
    key = lambda d: tuple(d.items())
    return list(unique_everseen(input_list, key=key))

def get_token_transactions(token_address, api_key, start_block=0, end_block=99999999): # as an example 
    base_url = "https://api.etherscan.io/api"

    # base_url = "https://api.bscscan.com/api" for bscscan
    # base_url = "https://api.polygonscan.com/api" for polygonscan

    module = "account"
    action = "tokentx"

    params = {
        "module": module,
        "action": action,
        "contractaddress": token_address,
        "startblock": start_block,
        "endblock": end_block,
        "sort": "asc",
        "apikey": api_key
    }
    success = False
    count = 0
    while not success and count < 10:
        try: 
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            success = True
        except Exception as e:
            print(f'Error: {e}, retrying...')

    if not success:
        print(f'Failed to get transactoins for token {token_address} from block {start_block} to {end_block} after 10 tries.')
        return []

    if data.get("status") == "1":
        return data.get("result")
    else:
        return []

def save_to_csv(transactions, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = transactions[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)

def main():
    # read in the address 
    chain = 'ethereum'
    address = list(pd.read_csv('../../_data/dataset/labels.csv').query('Chain == @chain').Contract.values)
    
    api_key = "Your EtherScan/PolygonScan/BscScan API"

    for token_address in tqdm(address):
        newpath = rf'{token_address}' 
        if os.path.exists(newpath):
            continue
        else:
            os.makedirs(newpath)

        transactions = []
        start_block = 0
        call_count = 0
        while True:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f'Currently getting transactions from block {start_block}, current time is {current_time}.')
            transactions_split = get_token_transactions(token_address, api_key, start_block=start_block)
            if not transactions_split:
                print(f'No more transactions found for starting from {start_block}, or there\'s an error.')
                break
            
            # remove all the transactions in the final block found, since they may not be complete
            first_block = int(transactions_split[0]['blockNumber'])
            final_block = int(transactions_split[-1]['blockNumber'])
            if first_block != final_block:
                end_index = len(transactions_split)
                for i in range(len(transactions_split) - 1, 0, -1):
                    if int(transactions_split[i]['blockNumber']) != final_block:
                        end_index = i + 1
                        break
                transactions_split = transactions_split[:end_index]
                final_block -= 1
            
            # append current transactions to the list
            transactions.extend(transactions_split)
            call_count += 1
            start_block = final_block + 1

            if call_count >= 10 or not transactions_split:
                if transactions:
                    first_block = transactions[0]['blockNumber']
                    last_block = transactions[-1]['blockNumber']
                    filename = f"{token_address}/transactions_{first_block}_{last_block}.csv"
                    print(f'Writing transactions from {first_block} to {last_block} to {filename}, current time is {current_time}, {len(transactions)} transactions in total.')
                    save_to_csv(transactions, filename)
                    print(f'Finished writing transactions from {first_block} to {last_block} to {filename}, current time is {current_time}.')
                    transactions = []
                    call_count = 0 
       
        if transactions:
            first_block = transactions[0]['blockNumber']
            last_block = transactions[-1]['blockNumber']
            filename = f"{token_address}/transactions_{first_block}_{last_block}.csv"
            print(f'Writing transactions from {first_block} to {last_block} to {filename}, current time is {current_time}, {len(transactions)} transactions in total.')
            save_to_csv(transactions, filename)   
            print(f'Finished writing transactions from {first_block} to {last_block} to {filename}, current time is {current_time}.')

if __name__ == "__main__":
    main()
