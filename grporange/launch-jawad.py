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
    numberOfPlayers=1, numberOfRobots=3, tic=100,
    missions= [(1, 32), (16, 24), (6, 7)],
    numberOfPVips=1
)
player= CompleteBot(debug=True)
player2= MultiBot()
gameMaster= moveit.GameMaster(gameEngine, randomMission=10)
gameMaster.launch([player], gameEngine.numberOfPlayers())

print("Terminé")
