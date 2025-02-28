import csv

# Define file paths
csv_file_path = "stock_list.csv"
txt_file_path = "sp500_data.txt"

# Dictionary to store companies and their associated ticker symbols
sp500_dict = {}

# Read the CSV file
with open(csv_file_path, mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)  # Default comma-separated CSV
    for row in reader:
        if len(row) < 3:  # Skip rows with missing data
            continue
        ticker_symbol = row[1].strip()  # Column B: Ticker Symbol
        company_name = row[2].strip()  # Column C: Company Name

        if company_name in sp500_dict:
            sp500_dict[company_name].append(ticker_symbol)
        else:
            sp500_dict[company_name] = [ticker_symbol]

# Write formatted output to txt file
with open(txt_file_path, mode="w", encoding="utf-8") as file:
    file.write("sp500_data = [\n")
    for company, symbols in sp500_dict.items():
        file.write(f'    {{"symbols": {symbols}, "company": "{company}"}},\n')
    file.write("]\n")

print(f"Formatted data saved to {txt_file_path}")
