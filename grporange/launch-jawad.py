import hacka.games.moveit as moveit
import os
import json

from models import SemiCompleteBot, TspBot, CompleteBot, MultiBot

map = "large-1.json"
matrix = None
with open(f"./configs/{map}") as file:
    config= json.load(file)
    # On génère des données sur 10 parties
    matrix = config["matrix"]

gameEngine= moveit.GameEngine(
    matrix= matrix,
    numberOfPlayers=2, numberOfRobots=2, tic=100,
    missions= [(20, 32), (16, 24), (6, 7)],
    numberOfPVips=1
)
player= CompleteBot()
player2= MultiBot()
gameMaster= moveit.GameMaster(gameEngine, randomMission=10)
gameMaster.launch([player, player2], gameEngine.numberOfPlayers())

print("Terminé")
