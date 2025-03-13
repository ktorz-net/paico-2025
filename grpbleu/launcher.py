import hacka.games.moveit as moveit
from src import Bot

import os
import json

# Select the map
map = "map_3"

config_path = os.path.join(os.getcwd(), "PAICO","data", "map.json")
with open(config_path, 'r') as file:
    data = json.load(file)
matrix = data[map]["matrix"]
numberOfPlayers = data[map]["numberOfPlayers"]
numberOfRobots = data[map]["numberOfRobots"]
tic = data[map]["tic"]
missions = data[map]["missions"]
random_mission = len(missions)
numberOfVips = data[map]["numberOfVips"]

gameEngine = moveit.GameEngine(
    matrix=matrix,
    numberOfPlayers=numberOfPlayers,
    numberOfRobots=numberOfRobots,
    numberOfPVips=numberOfVips,
    tic=tic,
    missions=missions)

gameMaster = moveit.GameMaster(gameEngine)
players = []
for i in range(numberOfPlayers):
    players.append(Bot())

gameMaster.launch(players, gameEngine.numberOfPlayers())