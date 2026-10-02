import json

with open("users.json") as f:
    users = json.load(f)

print(f"Total accounts: {len(users)}")
for username in users:
    marker = "  <-- BLANK USERNAME" if username.strip() == "" else ""
    print(f"  {username!r}{marker}")
