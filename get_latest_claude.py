import sqlite3
import json

conn = sqlite3.connect('iacs/data/messages.db')
cur = conn.cursor()
cur.execute('SELECT timestamp, sender, payload FROM messages WHERE sender != "gemini" ORDER BY created_at DESC LIMIT 3;')

for row in reversed(cur.fetchall()):
    data = json.loads(row[2])
    print(f"\n{'='*80}\n[{row[0]}] FROM {row[1]}\n{'='*80}")
    print(data.get("content", data))
