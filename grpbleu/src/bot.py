import time
from .path.path_manager import PathManager
from .game.game import Game
from .utils.action_builder import ActionBuilder

class Bot:
    def __init__(self):
        self._id = None
        self.game = Game()
        self.all_robots = []

    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self._id = playerId

        self.game.initModel(gameConfiguration)
        self.game.computeAllDistances()
        self.game.initVIP()
        self.game.initPlayers(numberOfPlayers, playerId)
        self.game.initRobots()
        self.game.initRobotPriority(len(self.game.getAllRobots()))
        self.game.getMissionManager().initMissions(self.game.getModel())
        self.game.getModel().render()

        self.all_robots = self.game.getAllRobots()

    def decide(self):
        self.game.getMissionManager().assignMissions(self.game, self._id)
        self.game.getPlayer(self._id).updateMissionBoolean()

        PathManager.assignPaths(self.game, self.all_robots)
        self.game.getPlayer(self._id).updateRobotBlockerAndPriorities(self.game.getDistances(), self.all_robots)

        result_actions = ""
        for robot in self.all_robots:
            action = ActionBuilder.action_decide(self.game, robot)
            robot.setLastAction(action)
            result_actions += action
            result_actions += " "

        result_actions_checked = ActionBuilder.checkResultAction(self.game, result_actions)

        # print(f"Player {self._id}: Without Check: {result_actions}")
        # print(f"Player {self._id}: With check:    {result_actions_checked}")

        final_action = ActionBuilder.format_actions(result_actions_checked)
        return final_action

    def perceive(self, state):
        """Met à jour l'état du jeu et enregistre les cases visitées"""
        self.game.getModel().setOnState(state)
        # print(f"Score: {self.game.getModel().setOnState(state)._scores}")

        self.game.getVIP().updateVipPosition(self.game.getModel())
        for player in self.game.getPlayers():
            player.updateRobotsPosition(self.game.getModel())
            for robot in player.getRobots():
                self.game.addTile(robot.getPosition())

        self.game.getMissionManager().updateAllMissions(self.game.getModel())

        self.game.getModel().render()
        # time.sleep(0.3)

    def sleep(self, result):
        """Gestion de fin de partie avec logging des statistiques"""
        # self.logger.info("\n=== Fin de partie ===")
        # self.logger.info(f"Score final: {result}")
        # self.logger.info(f"Nombre total de cases visitées: {len(self._visited_tiles)}")
        # self.logger.info(f"Cases visitées: {sorted(list(self._visited_tiles))}")

        # Statistiques additionnelles
        # if hasattr(self, '_vip_stuck_counter'):
        #     self.logger.info(f"Nombre de fois où le VIP est resté bloqué: {self._vip_stuck_counter}")
        pass