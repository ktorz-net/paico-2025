import sys
from initializer import initialize

matrix, numberOfPlayers, numberOfRobots, numberOfVips, tic, missions = initialize("map_multipath_testing")

workdir= __file__.split('tests')[0]
sys.path.insert( 1, workdir )

from src.game.game import Game

def test_recursivePath():
    game = Game()
    game.buildModel(matrix, numberOfPlayers, numberOfRobots, numberOfVips, tic, missions)
    game._model._map.teleport(1,3)
    game.initPlayersAndRobots(numberOfPlayers, 1)
    game.computeAllDistances()
    game._model.render()
    a = game.path(1,4)
    print(f"This is daway: {a}")
    assert 1 == 2
