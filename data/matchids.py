import apis
import json
import time

def main():
    puuids = []
    for entry in apis.get_dplus_entries("challenger"):
        puuids.append(entry['puuid'])
    
    for entry in apis.get_dplus_entries("grandmaster"):
        puuids.append(entry['puuid'])
    
    match_ids = set()
    iteration = 0
    for puuid in puuids:
        try:
            for id in apis.get_match_ids(puuid):
                match_ids.add(id)
            print(f"Iteration: {iteration}/{len(puuids)}")
            iteration += 1
        except Exception as e:
            print(e)
    print(f"Total match IDs: {len(match_ids)}")
    with open("match_ids.json", "w") as f:
        json.dump(list(match_ids), f)

if __name__ == "__main__":
    main()