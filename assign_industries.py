# #region imports
# from AlgorithmImports import *
# #endregion
from sec_api import MappingApi
import os
import pandas as pd
import json

API_KEY = "a403d5c05f8508aeba59443977a060e318f5feb778ff5416701f3b36e0609ef9"
EXCHANGES = ["NASDAQ", "NYSE", "AMEX"]
SICCODES_FILE = "siccodes49.txt"
OUTPUT_DIR = "industry_stocks_data"
MAPPING_FILE = "morningstar_to_ff_mapping.txt"
MORNINGSTAR_TO_FF_FILE = "morningstar_to_ff.json"
FF_IND_TO_SIC_FILE = "ind_siccodes.json"

def map_morningstar_to_ff(mapping_file, output_file_name):
    """
    Setup one-to-many mapping between morningstar and ff industries
    """
    with open(mapping_file, "r") as f:
        mapping = f.readlines()
    f.close()

    morningstar_to_ff = {}
    for line in mapping:
        morningstar, _, ff = line.strip().split()[0:3]
        if morningstar not in morningstar_to_ff:
            morningstar_to_ff[morningstar] = []
        morningstar_to_ff[morningstar].append(ff)

    # save dict to file for future use
    with open(output_file_name, "w") as f:
        json.dump(morningstar_to_ff, f)

def create_ff_ind_to_siccodes(siccodes_data):
    """
    Takes in input of `load_siccodes_data`
    """
    ind_siccodes = {}
    for sic, ind_idx in siccodes_data['siccode_to_idx'].items():
        ind_abbr = siccodes_data['industry_abbrs'][ind_idx]
        if ind_abbr not in ind_siccodes:
            ind_siccodes[ind_abbr] = []
        ind_siccodes[ind_abbr].append(sic)

    # save to json
    with open(IND_TO_SIC_FILE, "w") as f:
        json.dump(ind_siccodes, f)
    return ind_siccodes

def load_exhanges_data(api_key, exchanges):
    mapping_api = MappingApi(api_key=api_key)

    data = {}

    for exchange in exchanges:
        resp = mapping_api.resolve("exchange", exchange)
        data[exchange] = [x for x in resp if x["isDelisted"] == False]

    return data


def load_siccodes_data(siccodes_file):
    with open(siccodes_file, "r") as f:
        siccodes = f.readlines()
    f.close()

    industries_abbr = []
    industries_name = []
    industries_siccodes = {}

    idx = 0
    while idx < len(siccodes):
        curr_line = siccodes[idx]
        industry_idx = int(curr_line[0:2].strip()) - 1
        industry_abbr = curr_line[2:10].strip()
        industry_name = curr_line[10:].strip()

        industries_abbr.append(industry_abbr)
        industries_name.append(industry_name)

        idx += 1

        curr_line = siccodes[idx].strip()
        while curr_line != "" and idx + 1 < len(siccodes):
            low = int(curr_line[:4])
            high = int(curr_line[5:9])
            for code in range(low, high + 1):
                industries_siccodes[code] = industry_idx
            idx += 1
            curr_line = siccodes[idx].strip()

        if idx + 1 == len(siccodes):
            curr_line = siccodes[idx].strip()
            low = int(curr_line[:4])
            high = int(curr_line[5:9])
            for code in range(low, high + 1):
                industries_siccodes[code] = industry_idx
            break

        idx += 1

    siccodes_data = {
        "industry_abbrs": industries_abbr,
        "industry_names": industries_name,
        "siccode_to_idx": industries_siccodes,
    }

    return siccodes_data


def assign_stocks_to_industries(exchanges_data, siccodes_data):
    industry_not_found = 0
    industries_found = 0

    industry_stocks = {abbr: {} for abbr in siccodes_data["industry_abbrs"]}

    for exchange, stocks in exchanges_data.items():
        for idx, stock in enumerate(stocks):
            try:
                siccode = int(stock["sic"])
            except ValueError:
                industry_not_found += 1
                continue

            industry = siccodes_data["siccode_to_idx"].get(siccode, None)

            if industry is None:
                industry_not_found += 1
                continue

            industry_stocks[siccodes_data["industry_abbrs"][industry]][
                stock["ticker"]
            ] = stock
            industries_found += 1

    print(f"Matched industries for {industries_found} stocks")
    print(f"Couldn't match industries for {industry_not_found} stocks")

    return industry_stocks


def save_industry_stocks(industry_stocks, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for industry, stocks in industry_stocks.items():
        try:
            df = pd.DataFrame.from_dict(stocks, orient="index")
            df = df.drop(columns=["ticker"])
            df.index.name = "ticker"
            df.to_csv(f"{output_dir}/{industry}.csv")
        except Exception as e:
            print(f"Couldn't save {industry} stocks to file: {e}")


def assign_industries_main(api_key, exchanges, siccodes_file, output_dir):
    exchanges_data = load_exhanges_data(api_key, exchanges)
    for exchange, stocks in exchanges_data.items():
        print(f"Loaded {len(stocks)} stocks for {exchange} exchange")

    siccodes_data = load_siccodes_data(siccodes_file)
    print(f"Loaded siccodes data for {len(siccodes_data['industry_abbrs'])} industries")

    industry_stocks = assign_stocks_to_industries(exchanges_data, siccodes_data)

    save_industry_stocks(industry_stocks, output_dir)
    print(f"Saved industry stocks to {output_dir}")


if __name__ == "__main__":
    assign_industries_main(API_KEY, EXCHANGES, SICCODES_FILE, OUTPUT_DIR)

