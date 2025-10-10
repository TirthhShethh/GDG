#part A
def extract_zone(forest_map, center, m):
    rows = len(forest_map)
    cols = len(forest_map[0])
    r, c = center

    half = m // 2

    start_row = max(0, r - half)
    end_row = min(rows, r + half + 1)
    start_col = max(0, c - half)
    end_col = min(cols, c + half + 1)

    zone = [row[start_col:end_col] for row in forest_map[start_row:end_row]]

    total_trees = sum(sum(row) for row in zone)

    print(f"Center of Operations: ({r}, {c})")
    print(f"Extraction Zone ({m}x{m}):")
    for row in zone:
        print(row)
    print(f"Total Lal Chandan Trees in Zone: {total_trees}")


forest_map = [
    [1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 0, 1, 1, 0],
    [0, 1, 0, 1, 1]
]

m = 3
center = (2, 3)

extract_zone(forest_map, center, m)

#Prt B
from itertools import combinations

players = {
    "Kohli": {
        "strengths": ["chase_master", "fast_bowling_destroyer", "fielding"],
        "weaknesses": ["left_arm_spin"]
    },
    "Rahul": {
        "strengths": ["opener", "power_play", "wicketkeeping"],
        "weaknesses": ["pressure", "death_bowling"]
    },
    "Bumrah": {
        "strengths": ["death_bowling", "yorkers", "economy"],
        "weaknesses": ["batting"]
    },
    "Jadeja": {
        "strengths": ["power_hitting", "off_spin", "fielding"],
        "weaknesses": []
    },
    "Maxwell": {
        "strengths": ["spin_bowling", "fielding", "finisher"],
        "weaknesses": ["pace_bounce", "consistency"]
    },
    "Siraj": {
        "strengths": ["swing_bowling", "new_ball"],
        "weaknesses": ["batting"]
    },
    "Shreyas": {
        "strengths": ["middle_order", "spin_hitter"],
        "weaknesses": ["express_pace", "short_ball"]
    },
    "Chahal": {
        "strengths": ["leg_spin", "wicket_taker"],
        "weaknesses": ["fielding", "batting", "expensive"]
    },
    "DK": {
        "strengths": ["finisher", "wicketkeeping", "experience"],
        "weaknesses": ["poor_wicketkeeping"]
    },
    "Faf": {
        "strengths": ["opener", "experience", "fielding"],
        "weaknesses": ["slow_starter"]
    }
}
def calculate_team_score(team):
    strengths = set()
    weaknesses = set()
    
    for player in team:
        strengths.update(players[player]["strengths"])
        weaknesses.update(players[player]["weaknesses"])
    
    score = len(strengths) - len(weaknesses)
    if "opener" in strengths:
        score += 1
    if "finisher" in strengths:
        score += 1
    if "death_bowling" in strengths:
        score += 1

    overlap_penalty = len(strengths.intersection(weaknesses))
    score -= overlap_penalty

    return score, len(strengths), len(weaknesses)

def find_best_team(k):
    best_score = float('-inf')
    best_team = None
    best_details = None

    for combo in combinations(players.keys(), k):
        score, s_count, w_count = calculate_team_score(combo)
        if score > best_score:
            best_score = score
            best_team = combo
            best_details = (s_count, w_count)

    print(f"\n Best Team Found for k = {k}")
    print("=======================================")
    print("Players:", ', '.join(best_team))
    print(f"Net Score: {best_score}  (Unique Strengths = {best_details[0]}, Unique Weaknesses = {best_details[1]})")
    print("=======================================")

if __name__ == "__main__":
    print("=== IPL Dream Team Challenge ===")
    k = int(input("Enter number of players : "))
    find_best_team(k)