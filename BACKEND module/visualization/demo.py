"""
demo.py
End-to-end example: seed a SQLite table via db_crud, then visualize the
results via viz_helpers. Run with: python demo.py
"""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "database"))
from db_crud import DatabaseManager
from viz_helpers import set_style, plot_bar, plot_line, show

set_style()

# --- 1. CRUD: set up a small "sales" table -----------------------------
db = DatabaseManager("demo.db")
db.drop_table("sales")
db.create_table("sales", {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "month": "TEXT NOT NULL",
    "region": "TEXT NOT NULL",
    "revenue": "REAL NOT NULL",
})

db.insert_many("sales", [
    {"month": "Jan", "region": "East", "revenue": 12500},
    {"month": "Feb", "region": "East", "revenue": 13800},
    {"month": "Mar", "region": "East", "revenue": 15200},
    {"month": "Jan", "region": "West", "revenue": 9800},
    {"month": "Feb", "region": "West", "revenue": 11100},
    {"month": "Mar", "region": "West", "revenue": 10400},
])

# Read it back
rows = db.read("sales", order_by="month")
print(f"Inserted rows, read back {len(rows)}:")
for r in rows:
    print(" ", r)

# Update example
updated = db.update("sales", {"revenue": 16000}, where={"month": "Mar", "region": "East"})
print(f"\nUpdated {updated} row(s)")

# Count / read_one examples
print("East region count:", db.count("sales", where={"region": "East"}))

# --- 2. Visualization: chart the CRUD results ---------------------------
df = pd.DataFrame(db.read("sales"))

plot_bar(df[df.region == "East"], x="month", y="revenue",
          title="East Region Revenue by Month", save_path="bar_chart.png")

pivot = df.pivot(index="month", columns="region", values="revenue").reset_index()
import matplotlib.pyplot as plt
plt.figure()
plot_line(pivot, x="month", y=["East", "West"],
          title="Revenue by Region Over Time", ylabel="Revenue ($)",
          save_path="line_chart.png")

db.close()
print("\nCharts saved to the visualization folder.")
