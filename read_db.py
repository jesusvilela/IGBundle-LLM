import sqlite3
import json

conn = sqlite3.connect('iacs/data/messages.db')
cur = conn.cursor()
cur.execute('SELECT timestamp, sender, payload FROM messages ORDER BY created_at DESC LIMIT 20;')
for row in cur.fetchall():
    print(f"[{row[0]}] {row[1]}: {row[2][:200]}")
