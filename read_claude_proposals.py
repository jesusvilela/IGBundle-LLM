import sqlite3
import json

conn = sqlite3.connect('iacs/data/messages.db')
cur = conn.cursor()
cur.execute('SELECT timestamp, sender, payload FROM messages WHERE sender != "gemini" AND timestamp > "2026-03-22T17:50" ORDER BY created_at ASC;')

for row in cur.fetchall():
    data = json.loads(row[2])
    print(f"\n{'='*80}\n[{row[0]}] FROM {row[1]}\n{'='*80}")
    print(data.get("content", ""))
