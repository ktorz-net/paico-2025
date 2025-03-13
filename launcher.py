#!env python3
import json, sys
import hacka.games.moveit as moveit

from grpred.team import bots as botGenerator
#from grpbleu.src.team import bots as bbots

bots= botGenerator()

print( "Start: " + " ".join(sys.argv) )
config= {
    "matrix": [
        [ 0,  0,  0,  0],
        [ 0, -1,  0, -1],
        [ 0,  0,  0,  0]
    ],
    "tic": 10,
    "numberOfMissions": 2,
    "numberOfPlayers": 1,
    "numberOfRobots": 1,
    "numberOfPVips": 1,
    "vipZones": [4, 10, 7, 1]
}

# Open a configuration file:
nbArgs= len(sys.argv)
if nbArgs > 1 :
    with open( sys.argv[1] ) as file:
        config= json.load(file)

# number of player:
if nbArgs > 2 :
    config["numberOfPlayers"]= int(sys.argv[2])

# number of robot:
if nbArgs > 3 :
    config["numberOfRobots"]= int(sys.argv[3])

# VIP:
if nbArgs > 4 :
    config["numberOfPVips"]= max( int(sys.argv[4]), 1 )

# Configure the game:
gameEngine= moveit.GameEngine(
    matrix= config['matrix'],
    tic= config['tic'],
    numberOfPlayers= config['numberOfPlayers'],
    numberOfRobots= config['numberOfRobots'],
    numberOfPVips= config['numberOfPVips']
)

# Then Go...
gameMaster= moveit.GameMaster(
    gameEngine,
    randomMission= config['numberOfMissions'],
    vipZones= config['vipZones']
)

if config['numberOfPlayers'] == 2 :
    gameMaster.launch( [bots[0], bots[1]], 10 )
else :
    gameMaster.launch( [bots[0]], 10 )
