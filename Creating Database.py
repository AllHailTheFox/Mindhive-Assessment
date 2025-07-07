import sqlite3
import pandas as pd

# Load Excel file
df = pd.read_excel(r"C:\Users\Ervyn\Downloads\Mindhive\SQL.xlsx", index_col=None)

# Step 2: Select only the relevant columns (ensure exact match)
expected_cols = [
    "Shop Name",
    "Address",
    "City",
    "State",
    "Opening Days",
    "Opening Time",
    "Closing Time",
    "Phone Number"
]

df = df[expected_cols]

# Optional: print to verify structure
print(df)
print("Columns:", df.columns)
print("Shape:", df.shape)

# Step 3: Insert into SQLite
conn = sqlite3.connect("zus_outlets.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS outlets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_name TEXT,
    address TEXT,
    city TEXT,
    state TEXT,
    opening_days TEXT,
    opening_time TEXT,
    closing_time TEXT,
    phone_number TEXT
)
""")

# Step 4: Insert data
outlet_data = df.values.tolist()  # Convert to list of tuples
cursor.executemany("""
    INSERT INTO outlets (
        shop_name, address, city, state, opening_days, opening_time, closing_time, phone_number
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", outlet_data)

conn.commit()
conn.close()

print("✅ Successfully inserted data into zus_outlets.db")