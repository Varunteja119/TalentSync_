import json
import os

USER_DB = "users.json"


def load_users():
    if not os.path.exists(USER_DB):
        return {}
    try:
        with open(USER_DB, "r") as f:
            data = json.load(f)
            return data if data else {}
    except (json.JSONDecodeError, IOError):
        return {}


def save_users(users):
    with open(USER_DB, "w") as f:
        json.dump(users, f, indent=2)


def signup(username, password, profile):
    users = load_users()

    if username in users:
        return False

    users[username] = {
        "password": password,
        "profile": profile
    }

    save_users(users)
    return True


def login(username, password):
    users = load_users()

    if username in users and users[username]["password"] == password:
        return users[username]["profile"]

    return None


def update_profile(username, profile):
    users = load_users()

    if username not in users:
        return False

    users[username]["profile"] = profile
    save_users(users)
    return True