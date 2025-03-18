import hacka.games.moveit as moveit
import os
import json

from models import SpacialTimeGenBot

# Liste des configurations
maps = os.listdir("./configs")

# Nombre de partie par config
n_parties_per_config = 100

for map in maps:
    with open(f"./configs/{map}") as file:
        config= json.load(file)

    gameEngine= moveit.GameEngine(
        matrix= config['matrix'],
        numberOfPlayers=1, numberOfRobots=0, tic=100,
        missions= [(20, 32), (16, 24), (6, 7)],
        numberOfPVips=1
    )
    player= SpacialTimeGenBot()
    gameMaster= moveit.GameMaster(gameEngine)
    gameMaster.launch([player], numberOfGames=n_parties_per_config)

print("Terminé")
