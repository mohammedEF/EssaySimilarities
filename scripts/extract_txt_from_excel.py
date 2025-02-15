import pandas as pd

# File path
file_path = "/data/ESP 2024.xlsx"

# Load Excel file
xls = pd.ExcelFile(file_path)

# Read all sheets into a dictionary of DataFrames
sheets_data = {sheet: xls.parse(sheet, dtype=str) for sheet in xls.sheet_names}

# Convert all sheets to text format
txt_content = ""

for sheet_name, df in sheets_data.items():
    txt_content += f"Sheet: {sheet_name}\n"
    txt_content += df.to_string(index=False, header=True)  # Keep exact formatting
    txt_content += "\n\n"

# Define output file path
output_txt_path = "/data/ESP_2024.txt"

# Save to a text file
with open(output_txt_path, "w", encoding="utf-8") as txt_file:
    txt_file.write(txt_content)
