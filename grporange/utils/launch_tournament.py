import hacka.games.moveit as moveit
import json
from models import multiplayerbot, VIPBot, MultiDangerPathMCBot, MonteCarloBot,VIPAvoidPathBot

with open("./config-7x10.json") as file:
    config = json.load(file)

n_games = 100000

def create_multiplayer_bot():
    return multiplayerbot.MultiPlayerBot()

def create_avoid_path_bot():
    vip = VIPBot.VIPBot()
    enemy = multiplayerbot.MultiPlayerBot()
    return VIPAvoidPathBot.VIPAvoidPathMCBot(vip, enemy, debug=False)

def create_multi_danger_path_bot():
    vip = VIPBot.VIPBot()
    enemy = multiplayerbot.MultiPlayerBot()
    return MultiDangerPathMCBot.MultiDangerPathMCBot(vip, enemy, debug=False)

def create_monte_carlo_bot():
    vip = VIPBot.VIPBot()
    enemy = multiplayerbot.MultiPlayerBot()
    return MonteCarloBot.PathMCBot(vip, enemy, debug=False)

bots = [
    ("MultiPlayerBot", create_multiplayer_bot),
    ("AvoidPathBot", create_avoid_path_bot),
    ("VIPMoveTrackerBot", create_multi_danger_path_bot),
    ("MonteCarloBot", create_monte_carlo_bot)
]

results = {name: {"total_score": 0.0, "total_diff": 0.0, "games": 0} for name, _ in bots}

# Round-robin tournament: each pair of bots plays two matches (swapping roles)
for i in range(len(bots)):
    for j in range(i + 1, len(bots)):
        bot1_name, bot1_creator = bots[i]
        bot2_name, bot2_creator = bots[j]
        for match in range(2):
            gameEngine = moveit.GameEngine(
                matrix=config['matrix'],
                tic=config['tic'],
                numberOfPlayers=config['numberOfPlayers'],
                numberOfRobots=config['numberOfRobots'],
                numberOfPVips=config['numberOfPVips']
            )
            gameMaster = moveit.GameMaster(gameEngine, randomMission=10)
            if match == 0:
                players = [bot1_creator(), bot2_creator()]
            else:
                players = [bot2_creator(), bot1_creator()]
            gameMaster.launch(players, gameEngine.numberOfPlayers())
            score1 = gameEngine.score(1)
            score2 = gameEngine.score(2)
            if match == 0:
                results[bot1_name]["total_score"] += score1
                results[bot2_name]["total_score"] += score2
                results[bot1_name]["total_diff"] += (score1 - score2)
                results[bot2_name]["total_diff"] += (score2 - score1)
            else:
                results[bot2_name]["total_score"] += score1
                results[bot1_name]["total_score"] += score2
                results[bot2_name]["total_diff"] += (score1 - score2)
                results[bot1_name]["total_diff"] += (score2 - score1)
            results[bot1_name]["games"] += 1
            results[bot2_name]["games"] += 1
            print(f"Match between {bot1_name} and {bot2_name} (swap={match}): scores: {score1} vs {score2}")

print("----- Tournament Results -----")
for name, data in results.items():
    avg_score = data["total_score"] / data["games"] if data["games"] > 0 else 0.0
    avg_diff = data["total_diff"] / data["games"] if data["games"] > 0 else 0.0
    print(f"{name}: Average Score = {avg_score}, Average Diff (our score - opponent score) = {avg_diff}")