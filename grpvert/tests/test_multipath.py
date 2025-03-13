from hacka.games.moveit import GameEngine
from bot import SoloBot

def test_multipath1():

    gameEngine= GameEngine(
        matrix= [
        [ -1,  -1,  0,  -1,  -1],
        [ 0,  0,  0,  0,  0],
        [0,  -1,  0,  -1,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  -1,  0,  0],
        [ 0,  0,  0,  0,  0]
        ],
        tic= 20,
        numberOfPlayers=1, numberOfRobots=1)

    player = SoloBot()
    player._id = 1
    player._model = gameEngine
    player.initGame()
    player._model.render()
    assert player._model.mobilePosition(1,1) == 1
    assert player.path(1,21,5) == [([6, 6, 6, 9, 6, 6, 3], [4, 8, 12, 11, 16, 20, 21]), ([6, 6, 6, 3, 6, 6, 9], [4, 8, 12, 13, 17, 22, 21]), ([6, 6, 6, 9, 6, 9, 6, 3, 3], [4, 8, 12, 11, 16, 15, 19, 20, 21]), ([6, 6, 6, 9, 9, 6, 3, 6, 3], [4, 8, 12, 11, 10, 15, 16, 20, 21]), ([6, 6, 6, 9, 9, 6, 6, 3, 3], [4, 8, 12, 11, 10, 15, 19, 20, 21])]
    assert player.path(3,15,5) == [([9, 6, 6, 6], [2, 7, 10, 15]), ([9, 6, 6, 3, 6, 9], [2, 7, 10, 11, 16, 15]), ([3, 6, 6, 9, 9, 6], [4, 8, 12, 11, 10, 15]), ([3, 6, 6, 9, 6, 9], [4, 8, 12, 11, 16, 15]), ([9, 6, 6, 3, 6, 6, 9, 12], [2, 7, 10, 11, 16, 20, 19, 15])]
    assert player.detectCollision([1,3,6,9,7],[5,5,6,7]) == True
    assert player.detectCollision([1,3,6,9,7],[5,6,3,7]) == True