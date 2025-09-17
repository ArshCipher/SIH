import sqlite3

# Connect to database
conn = sqlite3.connect('health_chatbot.db')
cursor = conn.cursor()

# Check schema
cursor.execute("SELECT sql FROM sqlite_master WHERE name='disease_info'")
schema = cursor.fetchone()
print("Schema:")
print(schema[0] if schema else "Table not found")

# Check current data
cursor.execute("SELECT COUNT(*) FROM disease_info")
count = cursor.fetchone()[0]
print(f"\nCurrent disease count: {count}")

# Show sample data
cursor.execute("SELECT disease_name, symptoms FROM disease_info LIMIT 5")
diseases = cursor.fetchall()
print("\nSample diseases:")
for disease in diseases:
    print(f"- {disease[0]}: {disease[1][:100]}...")

conn.close()