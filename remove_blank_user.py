import json

with open("users.json") as f:
    users = json.load(f)

if "" in users:
    del users[""]
    with open("users.json", "w") as f:
        json.dump(users, f, indent=2)
    print("Removed the blank-username account.")
else:
    print("No blank-username account found -- nothing to remove.")
