import sqlite3
import json

conn = sqlite3.connect('iacs/data/messages.db')
cur = conn.cursor()
cur.execute('SELECT timestamp, sender, payload FROM messages WHERE sender != "gemini" ORDER BY created_at DESC LIMIT 1;')
row = cur.fetchone()
if row:
    print(f"[{row[0]}] {row[1]}: {row[2]}")
else:
    print("No messages found.")
