import multiplayerbot
import hacka.games.moveit as moveit

# def test_calc_probability():
#     gameEngine= moveit.GameEngine(
#         matrix= [ [00, 00, 00, 00, 00, 00, 00, 00],
#                 [00, 00, -1, -1, -1, -1, 00, 00],
#                 [00, 00, 00, 00, -1, 00, 00, 00],
#                 [00, 00, 00, 00, 00, 00, 00, 00],
#                 [00, 00, -1, -1, 00, 00, -1, 00],
#                 [00, 00, -1, 00, 00, 00, -1, 00],
#                 [00, 00, 00, 00, 00, 00, 00, 00],
#                 ],
#         numberOfPlayers=2, numberOfRobots=3, tic=100,
#         missions= [(1, 29), (16, 24), (4, 21)]
#     )
#     bot = multiplayerbot.BetterMultiPlayerBot()
#     bot._model = moveit.GameEngine()
#     bot._model.fromPod(gameEngine.asPod())

#     bot.wakeUp(self, 1, )