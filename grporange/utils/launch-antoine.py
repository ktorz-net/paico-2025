import playground
from models import firstbot
from models import multibot
from models import multiplayerbot
from models import multiplayerbotennemy
from models import bottest
from models import multiplay
import hacka.games.moveit as moveit
import json

# Configure the game:
gameEngine= moveit.GameEngine(
    matrix= [ 
              [00, 00, 00, 00, 00, 00, 00, 00],
              [00, 00, -1, -1, -1, -1, 00, 00],
              [00, 00, 00, 00, 00, 00, 00, 00],
            ],
    numberOfPlayers=1, numberOfRobot=2, tic=100,
    missions= [(1, 29), (16, 24), (4, 21)]
)

# Then Go...
gameMaster= moveit.GameMaster( gameEngine, randomMission=10 )
player= multiplayerbotennemy.MultiPlayerBotEnnemy()
#player2= multiplay.MultiPlayerBot()
gameMaster.launch( [player], gameEngine.numberOfPlayers() )
