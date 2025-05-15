import apis
from itertools import product
import pandas as pd
import json

LANE_POSITIONS = {
    "TOP_LANE": {
        "LANE": {"x": 981, "y": 13875, "radius": 3300},
        100: {"x":981,"y":10441, "radius": 1000},
        200: {"x":4318,"y":13875, "radius": 1000},
    },
    "MID_LANE": {
        "LANE": {"x": 7400, "y": 7450, "radius": 1700},
        100: {"x":5846,"y":6396, "radius": 1000},
        200: {"x":8955,"y":8510, "radius": 1000},
    },
    "BOT_LANE": {
        "LANE": {"x": 13866, "y": 1029, "radius": 3300},
        100: {"x":10504,"y":1029, "radius": 1000},
        200: {"x":13866,"y":4505, "radius": 1000},
    },
}

ROLE_TO_OFFSET = {
    "TOP": 0,
    "JUNGLE": 1,
    "MIDDLE": 2,
    "BOTTOM": 3,
    "UTILITY": 4,
}

TEAM_TO_OFFSET = {
    100: 0,
    200: 5,
}

TEAMS = (100, 200)

ACTIONS = [
    "GANK","OBJECTIVE","RESET","INVADE",
    "CLEAR","PUSH","VISION_CONTROL",
    "PATH_BOT","PATH_TOP"
]

ACTION_INDEX = {
    (team, act): i
    for i, (team, act) in enumerate(product(TEAMS, ACTIONS))
}

WPE_STATE_COLS = (
    ["TIMESTAMP"] +
    [f"{'BLUE' if team==100 else 'RED'}_{role}_{feat}"
    for team, (role, feat) in product(TEAMS, product(["TOP", "JG", "MID", "BOT", "SUP"], ["XP", "TOTAL_GOLD"]))] +
    [f"{'BLUE' if team==100 else 'RED'}_{monster}_KILLS"
    for team, monster in product(TEAMS, ["HORDE", "AIR_DRAGON", "CHEMTECH_DRAGON", "EARTH_DRAGON", "FIRE_DRAGON", "HEXTECH_DRAGON", "WATER_DRAGON", "RIFTHERALD"])] +
    [f"{'BLUE' if team==100 else 'RED'}_{vision}"
    for team, vision in product(TEAMS, ["WARDS_PLACED", "WARDS_KILLED"])] + 
    [f"{'BLUE' if team==100 else 'RED'}_{lane}_{turret}"
    for team, (lane, turret) in product(TEAMS, product(("TOP", "MID", "BOT"), ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET")))] +
    [f"{'BLUE' if team==100 else 'RED'}_{lane}_INHIBITOR"
    for team, lane in product(TEAMS, ("TOP", "MID", "BOT"))] +
    ["BLUE_FEATS", "RED_FEATS"]
)

JPO_STATE_COLS = (
    ["TIMESTAMP"] +
    [f"{'BLUE' if team==100 else 'RED'}_{role}_{feat}"
    for team, (role, feat) in product(TEAMS, product(["TOP", "JG", "MID", "BOT", "SUP"], ["CHAMPION", "X", "Y", "LEVEL", "CURRENT_GOLD", "TOTAL_GOLD"]))] +
    ["BLUE_JG_JUNGLE_MINIONS_KILLED", "RED_JG_JUNGLE_MINIONS_KILLED"] +
    [f"{'BLUE' if team==100 else 'RED'}_{monster}_KILLS"
    for team, monster in product(TEAMS, ["HORDE", "AIR_DRAGON", "CHEMTECH_DRAGON", "EARTH_DRAGON", "FIRE_DRAGON", "HEXTECH_DRAGON", "WATER_DRAGON", "RIFTHERALD"])] +
    ["DRAGON_ALIVE", "HORDE_ALIVE", "RIFTHERALD_ALIVE"] +
    [f"{'BLUE' if team==100 else 'RED'}_{lane}_{turret}"
    for team, (lane, turret) in product(TEAMS, product(("TOP", "MID", "BOT"), ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET")))] +
    [f"{'BLUE' if team==100 else 'RED'}_{lane}_INHIBITOR"
    for team, lane in product(TEAMS, ("TOP", "MID", "BOT"))]
)

JPO_ACTION_COLS = [
    f"{'BLUE' if team==100 else 'RED'}_{act}"
    for team, act in product(TEAMS, ACTIONS)
]

def other_team(teamId):
    return 200 - (teamId-100)

def in_lane(teamId, x, y):
    otherTeamId = other_team(teamId)
    for lane, cfg in LANE_POSITIONS.items():
        for region in ["LANE", otherTeamId]:
            cx, cy, r = cfg[region]["x"], cfg[region]["y"], cfg[region]["radius"]
            if (x - cx)**2 + (y - cy)**2 <= r**2:
                return True
    return False

def in_opposing_jungle(teamId, x, y):
    return (y > -x + 16000 and teamId == 100) or (y < -x + 14000 and teamId == 200)

def get_side(x, y):
    if x > y:
        return "PATH_BOT"
    else:
        return "PATH_TOP"

def process_match_info(data):
    info = data["info"]
    if info["gameDuration"] < 900:
        return None
    
    ret = {
        "participants": {},
    }
    team = info["teams"]
    for t in team:
        if t["win"]:
            ret["winTeamId"] = t["teamId"]
    
    particpants = info["participants"]
    for p in particpants:
        ret["participants"][p["participantId"]] = {
            "participantId": p["participantId"],
            "teamId": p["teamId"],
            "position": p["teamPosition"],
            "championId": p["championId"],
        }
    return ret

def process_match_timeline(data, match_info):
    objectiveState = {
        "DRAGON": {
            "ALIVE": 0,
            "ALIVE_VAL": 1,
            "LASTKILLED": 0,
            "SPAWNTIMER": 300000,
            "RESPAWNTIMER": 300000
        },
        "HORDE": {
            "ALIVE": 0,
            "ALIVE_VAL": 3,
            "LASTKILLED": 0,
            "SPAWNTIMER": 360000, 
            "RESPAWNTIMER": 240000,
            "DESPAWNTIMER": 825000,
        },
        "RIFTHERALD": {
            "ALIVE": 0,
            "ALIVE_VAL": 1,
            "LASTKILLED": 0,
            "SPAWNTIMER": 960000,
        }
    }
    teamStates = {
        100: {
            "TOWER_BUILDING": {
                "TOP_LANE": {"OUTER_TURRET": True, "INNER_TURRET": True, "BASE_TURRET": True},
                "MID_LANE": {"OUTER_TURRET": True, "INNER_TURRET": True, "BASE_TURRET": True},
                "BOT_LANE": {"OUTER_TURRET": True, "INNER_TURRET": True, "BASE_TURRET": True},
            },
            "INHIBITOR_BUILDING": {
                "TOP_LANE": True,
                "MID_LANE": True,
                "BOT_LANE": True,
            },
            "HORDE": 0,
            "DRAGON": {
                "AIR_DRAGON": 0,
                "CHEMTECH_DRAGON": 0,
                "EARTH_DRAGON": 0,
                "FIRE_DRAGON": 0,
                "HEXTECH_DRAGON": 0,
                "WATER_DRAGON": 0,
            },
            "RIFTHERALD": 0,
            "CHAMPION_KILL": 0,
            "ELITE_MONSTER_KILL": 0,
            "WARD_PLACED": 0,
            "WARD_KILL": 0,
            "FEATS": False,
        },
        200: {
            "TOWER_BUILDING": {
                "TOP_LANE": {"OUTER_TURRET": True, "INNER_TURRET": True, "BASE_TURRET": True},
                "MID_LANE": {"OUTER_TURRET": True, "INNER_TURRET": True, "BASE_TURRET": True},
                "BOT_LANE": {"OUTER_TURRET": True, "INNER_TURRET": True, "BASE_TURRET": True},
            },
            "INHIBITOR_BUILDING": {
                "TOP_LANE": True,
                "MID_LANE": True,
                "BOT_LANE": True,
            },
            "HORDE": 0,
            "DRAGON": {
                "AIR_DRAGON": 0,
                "CHEMTECH_DRAGON": 0,
                "EARTH_DRAGON": 0,
                "FIRE_DRAGON": 0,
                "HEXTECH_DRAGON": 0,
                "WATER_DRAGON": 0,
            },
            "RIFTHERALD": 0,
            "CHAMPION_KILL": 0,
            "ELITE_MONSTER_KILL": 0,
            "WARD_PLACED": 0,
            "WARD_KILL": 0,
            "FEATS": False,
        },
    }
    feats = {
        "CHAMPION_KILL": 0,
        "BUILDING_KILL": 0,
        "ELITE_MONSTER_KILL": 0,
    }

    info = data["info"]
    frames = info["frames"]
    roleToId = {}
    for participant in match_info["participants"].values():
        pos = 0
        pos += TEAM_TO_OFFSET[participant["teamId"]]
        pos += ROLE_TO_OFFSET[participant["position"]]
        roleToId[pos] = participant
    
    jpo_X = []
    jpo_Y = []
    wpe_X = []
    wpe_Y = []
    junglers = {
        100: roleToId[1],
        200: roleToId[6],
    }
    prevParticipantFrames = None
    for i, f in enumerate(frames[:min(len(frames), 21)]):
        timestamp = f["timestamp"]

        for monsterType, state in objectiveState.items():
            if state.get("DESPAWNTIMER", -1) != -1 and timestamp >= state["DESPAWNTIMER"]:
                state["ALIVE"] = 0
            elif state["ALIVE"] == 0 and ((state["LASTKILLED"] == 0 and timestamp >= state["SPAWNTIMER"]) or (state.get("RESPAWNTIMER", -1) != -1 and (state["LASTKILLED"] != 0 and timestamp >= state["LASTKILLED"] + state["RESPAWNTIMER"]))):
                state["ALIVE"] = state["ALIVE_VAL"]

        jpo_state = [timestamp]
        wpe_state = [timestamp]
        jpo_action = [0] * (len(TEAMS) * len(ACTIONS))
        participantFrames = f["participantFrames"]
        for role in range(10):
            participant = roleToId[role]
            jpo_state.append(participant["championId"])
            participantId = participant["participantId"]
            participantFrame = participantFrames[str(participantId)]
            jpo_state.append(participantFrame["position"]["x"])
            jpo_state.append(participantFrame["position"]["y"])
            jpo_state.append(participantFrame["level"])
            jpo_state.append(participantFrame["currentGold"])
            jpo_state.append(participantFrame["totalGold"])
            wpe_state.append(participantFrame["xp"])
            wpe_state.append(participantFrame["totalGold"])

        for role in [1, 6]:
            participant = roleToId[role]
            participantId = participant["participantId"]
            participantFrame = participantFrames[str(participantId)]
            jpo_state.append(participantFrame["jungleMinionsKilled"])
            teamId = participant["teamId"]
            if prevParticipantFrames and participantFrame["jungleMinionsKilled"] > prevParticipantFrames[str(participantId)]["jungleMinionsKilled"]:
                if in_opposing_jungle(teamId, participantFrame["position"]["x"], participantFrame["position"]["y"]):
                    jpo_action[ACTION_INDEX[(teamId, "INVADE")]] = 1
                else:
                    jpo_action[ACTION_INDEX[(teamId, "CLEAR")]] = 1
        
            if prevParticipantFrames and in_lane(teamId, participantFrame["position"]["x"], participantFrame["position"]["y"]):
                jpo_action[ACTION_INDEX[(teamId, "GANK")]] = 1
            
            jpo_action[ACTION_INDEX[(teamId, get_side(participantFrame["position"]["x"], participantFrame["position"]["y"]))]] = 1

        events = f["events"]
        for e in events:
            e_type = e["type"]

            if e_type == "WARD_PLACED":
                creatorId = e["creatorId"]
                if creatorId == 0:
                    continue
                teamId = match_info["participants"][creatorId]["teamId"]
                teamStates[teamId][e_type] += 1

                if junglers[teamId]["participantId"] == creatorId:
                    jpo_action[ACTION_INDEX[(teamId, "VISION_CONTROL")]] = 1
            
            if e_type == "WARD_KILL":
                killerId = e["killerId"]
                if killerId == 0:
                    continue
                teamId = match_info["participants"][killerId]["teamId"]
                teamStates[teamId][e_type] += 1 

                if junglers[teamId]["participantId"] == killerId:
                    jpo_action[ACTION_INDEX[(teamId, "VISION_CONTROL")]] = 1

            if e_type == "ELITE_MONSTER_KILL":
                teamId = e["killerTeamId"]
                if teamId == 300:
                    continue
                monsterType = e["monsterType"]
                if monsterType == "DRAGON":
                    monsterSubType = e["monsterSubType"]
                    teamStates[teamId][monsterType][monsterSubType] += 1
                else:
                    teamStates[teamId][monsterType] += 1
                timestamp = e["timestamp"]
                objective = objectiveState[monsterType]
                objective["ALIVE"] -= 1
                if objective["ALIVE"] == 0:
                    objective["LASTKILLED"] = timestamp
                    if monsterType != "HORDE" or (monsterType == "HORDE" and (teamStates[teamId][monsterType] == 3 or teamStates[teamId][monsterType] == 6)):
                        teamStates[teamId][e_type] += 1
                        if teamStates[teamId][e_type] == 3 and feats[e_type] == 0:
                            feats[e_type] = teamId
                jpo_action[ACTION_INDEX[(teamId, "OBJECTIVE")]] = 1
            
            if e_type == "BUILDING_KILL":
                killerId = e["killerId"]
                if killerId == 0:
                    continue
                teamId = e["teamId"]
                otherTeamId = other_team(teamId)
                buildingType = e["buildingType"]
                laneType = e["laneType"]
                if buildingType == "TOWER_BUILDING":
                    towerType = e["towerType"]
                    teamStates[teamId][buildingType][laneType][towerType] = False
                if buildingType == "INHIBITOR_BUILDING":
                    teamStates[teamId][buildingType][laneType] = False
                if feats[e_type] == 0:
                    feats[e_type] = otherTeamId
                
                if junglers[otherTeamId]["participantId"] == killerId or junglers[otherTeamId]["participantId"] in e.get("assistingParticipantIds", []):
                    jpo_action[ACTION_INDEX[(otherTeamId, "PUSH")]] = 1

            if e_type == "TURRET_PLATE_DESTROYED":
                killerId = e["killerId"]
                teamId = e["teamId"]

                if junglers[teamId]["participantId"] == killerId:
                    jpo_action[ACTION_INDEX[(teamId, "PUSH")]] = 1

            if e_type == "CHAMPION_KILL":
                killerId = e["killerId"]
                if killerId == 0:
                    continue
                killerTeamId = match_info["participants"][killerId]["teamId"]
                victimId = e["victimId"]
                victimTeamId = match_info["participants"][victimId]["teamId"]
                teamStates[killerTeamId][e_type] += 1
                if teamStates[killerTeamId][e_type] == 3 and feats[e_type] == 0:
                    feats[e_type] = killerTeamId
                
                if (junglers[killerTeamId]["participantId"] == killerId or junglers[killerTeamId]["participantId"] in e.get("assistingParticipantIds", [])):
                    if in_lane(killerTeamId, e["position"]["x"], e["position"]["y"]):
                        jpo_action[ACTION_INDEX[(killerTeamId, "GANK")]] = 1
                    if in_opposing_jungle(killerTeamId, e["position"]["x"], e["position"]["y"]):
                        jpo_action[ACTION_INDEX[(killerTeamId, "INVADE")]] = 1

                if junglers[victimTeamId]["participantId"] == victimId:
                    if in_lane(victimTeamId, e["position"]["x"], e["position"]["y"]):
                        jpo_action[ACTION_INDEX[(victimTeamId, "GANK")]] = 1
                    if in_opposing_jungle(victimTeamId, e["position"]["x"], e["position"]["y"]):
                        jpo_action[ACTION_INDEX[(victimTeamId, "INVADE")]] = 1

            if e_type == "ITEM_PURCHASED" or e_type == "ITEM_SOLD" or e_type == "ITEM_UNDO":
                participantId = e["participantId"]
                if participantId == junglers[100]["participantId"]:
                    jpo_action[ACTION_INDEX[(100, "RESET")]] = 1
                elif participantId == junglers[200]["participantId"]:
                    jpo_action[ACTION_INDEX[(200, "RESET")]] = 1

        for team in TEAMS:
            wpe_state.append(teamStates[team]["HORDE"])
            jpo_state.append(teamStates[team]["HORDE"])
            for subtype in ("AIR_DRAGON",
                            "CHEMTECH_DRAGON",
                            "EARTH_DRAGON",
                            "FIRE_DRAGON",
                            "HEXTECH_DRAGON",
                            "WATER_DRAGON"):
                wpe_state.append(teamStates[team]["DRAGON"][subtype])
                jpo_state.append(teamStates[team]["DRAGON"][subtype])
            wpe_state.append(teamStates[team]["RIFTHERALD"])
            jpo_state.append(teamStates[team]["RIFTHERALD"])

        jpo_state.append(objectiveState["DRAGON"]["ALIVE"])
        jpo_state.append(objectiveState["HORDE"]["ALIVE"])
        jpo_state.append(objectiveState["RIFTHERALD"]["ALIVE"])

        for team in TEAMS:
            wpe_state.append(teamStates[team]["WARD_PLACED"])
            wpe_state.append(teamStates[team]["WARD_KILL"])

        for team in TEAMS:
            tb = teamStates[team]["TOWER_BUILDING"]
            for lane_key in ("TOP_LANE", "MID_LANE", "BOT_LANE"):
                for turret in ("OUTER_TURRET", "INNER_TURRET", "BASE_TURRET"):
                    wpe_state.append(int(tb[lane_key][turret]))
                    jpo_state.append(int(tb[lane_key][turret]))

        for team in TEAMS:
            ib = teamStates[team]["INHIBITOR_BUILDING"]
            for lane_key in ("TOP_LANE", "MID_LANE", "BOT_LANE"):
                wpe_state.append(int(ib[lane_key]))
                jpo_state.append(int(ib[lane_key]))

        for team in TEAMS:
            count = sum(1 for v in feats.values() if v == team)
            wpe_state.append(int(count >= 2))
        
        if i != min(len(frames), 21) - 1:
            jpo_X.append(jpo_state)
        if i != 0:
            jpo_Y.append(jpo_action)
        wpe_X.append(wpe_state)
        wpe_Y.append([match_info["winTeamId"]])
        prevParticipantFrames = participantFrames

    return wpe_X, wpe_Y, jpo_X, jpo_Y

def main():
    with open("match_ids.json", "r") as f:
        match_ids = json.load(f)

    CHUNK_SIZE = 100
    all_wpe_X, all_wpe_Y = [], []
    jpo_sequences = []

    for idx, match_id in enumerate(match_ids[46600:], 46600):
        print(f"Processing {idx + 1}/{len(match_ids)}: {match_id}")
        try:
            info_raw = apis.get_match_info(match_id)
            match_info = process_match_info(info_raw)
            if match_info is None:
                continue

            timeline_raw = apis.get_match_timeline(match_id)
            wpe_X_batch, wpe_Y_batch, jpo_X_batch, jpo_Y_batch = (
                process_match_timeline(timeline_raw, match_info)
            )

            wpe_X_df = pd.DataFrame(wpe_X_batch, columns=WPE_STATE_COLS)
            wpe_X_df["MATCH_ID"] = match_id
            wpe_Y_df = pd.DataFrame(wpe_Y_batch, columns=["WIN_TEAMID"])
            wpe_Y_df["MATCH_ID"] = match_id
            jpo_X_df = pd.DataFrame(jpo_X_batch, columns=JPO_STATE_COLS)
            jpo_X_df["MATCH_ID"] = match_id
            jpo_Y_df = pd.DataFrame(jpo_Y_batch, columns=JPO_ACTION_COLS)
            jpo_Y_df["MATCH_ID"] = match_id

            all_wpe_X.append(wpe_X_df)
            all_wpe_Y.append(wpe_Y_df)
            jpo_sequences.append((jpo_X_df, jpo_Y_df))

            if (idx + 1) % CHUNK_SIZE == 0:
                part = (idx + 1) // CHUNK_SIZE
                wpe_chunk = pd.concat(all_wpe_X, ignore_index=True)
                y_chunk   = pd.concat(all_wpe_Y, ignore_index=True)
                wpe_chunk.to_csv(f"/scratch/network/ly4431/wpe/wpe_X_part{part}.csv", index=False)
                y_chunk.to_csv(f"/scratch/network/ly4431/wpe/wpe_Y_part{part}.csv", index=False)
                for seq_i, (jx, jy) in enumerate(jpo_sequences, start=(part - 1) * CHUNK_SIZE):
                    jx.to_csv(f"/scratch/network/ly4431/jpo/jpo_X_{seq_i}.csv", index=False)
                    jy.to_csv(f"/scratch/network/ly4431/jpo/jpo_Y_{seq_i}.csv", index=False)
                all_wpe_X.clear()
                all_wpe_Y.clear()
                jpo_sequences.clear()
        except Exception as e:
            print(f"[Warning] Error processing match {match_id} at index {idx}: {e}")
            continue

    if all_wpe_X:
        part = (len(match_ids) // CHUNK_SIZE) + 1
        wpe_chunk = pd.concat(all_wpe_X, ignore_index=True)
        y_chunk   = pd.concat(all_wpe_Y, ignore_index=True)
        wpe_chunk.to_csv(f"/scratch/network/ly4431/wpe_X_part{part}.csv", index=False)
        y_chunk.to_csv(f"/scratch/network/ly4431/wpe_Y_part{part}.csv", index=False)
        for seq_i, (jx, jy) in enumerate(jpo_sequences, start=(part - 1) * CHUNK_SIZE):
            jx.to_csv(f"/scratch/network/ly4431/jpo/jpo_X_{seq_i}.csv", index=False)
            jy.to_csv(f"/scratch/network/ly4431/jpo/jpo_Y_{seq_i}.csv", index=False)

if __name__ == "__main__":
    main()