import sqlite3

db_path = 'data/data.db'

conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute(
    "SELECT DISTINCT username FROM info WHERE username != ?",
    ("shivss",)
)
result = c.fetchone()
conn.commit()
conn.close()

print(f"result\n{result}\n")