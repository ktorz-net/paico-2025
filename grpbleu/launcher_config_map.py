#!env python3
import json, sys
import os

import hacka.games.moveit as moveit

from src.team import bots

# Build the list of bots.
bots= bots()

print( "Start: " + " ".join(sys.argv) )
config_name = "medium-2.json"
config_path = os.path.join(os.getcwd(), "..", "data", "configs", config_name)
with open(config_path, 'r') as file:
    config = json.load(file)

# Configure the game:
gameEngine= moveit.GameEngine(
    matrix= config['matrix'],
    tic= config['tic'],
    numberOfPlayers= config['numberOfPlayers'],
    numberOfRobots= config['numberOfRobots'],
    numberOfPVips= config['numberOfPVips'],
)

# Then Go...
gameMaster= moveit.GameMaster(
    gameEngine,
    randomMission= config['numberOfMissions'],
    vipZones= config['vipZones']
)

if config['numberOfPlayers'] == 2 :
    gameMaster.launch( [bots[0], bots[1]] )
else :
    gameMaster.launch( [bots[0]] )
