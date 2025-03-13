import hacka.games.moveit as moveit
import os
import json

from models import BetterMultiPlayerBot, SpacialTimeGenBot

# Liste des configurations
maps = os.listdir("./configs")

# Nombre de partie par config
n_parties_per_config = 10

# Pour chaque map
for i, map in enumerate(maps):
    with open(f"./configs/{map}") as file:
        config= json.load(file)
        # On génère des données sur 10 parties
        for j in range(n_parties_per_config):
            gameEngine= moveit.GameEngine(
                matrix= config['matrix'],
                numberOfPlayers=1, numberOfRobots=0, tic=100,
                missions= [(20, 32), (16, 24), (6, 7)],
                numberOfPVips=1
            )
            player= SpacialTimeGenBot()
            gameMaster= moveit.GameMaster(gameEngine, randomMission=10)
            print(f"partie{n_parties_per_config * i + j}/{n_parties_per_config * len(maps)}")
            gameMaster.launch([player], gameEngine.numberOfPlayers())

print("Terminé")
