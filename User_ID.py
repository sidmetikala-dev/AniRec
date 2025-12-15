import json, os

def get_user_id(username):
    user_map_file = "user_map.json"
    if os.path.exists(user_map_file):
        with open(user_map_file, "r") as f:
            user_map = json.load(f)
    else:
        user_map = {}
    if username in user_map:
        return user_map[username]
    # assign new ID
    new_id = max(user_map.values(), default=0) + 1
    user_map[username] = new_id
    with open(user_map_file, "w") as f:
        json.dump(user_map, f, indent=2)
    return new_id

