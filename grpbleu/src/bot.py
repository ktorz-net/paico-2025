import time
from .path.path_manager import PathManager
from .game.game import Game
from .utils.action_builder import ActionBuilder

class Bot:
    def __init__(self):
        self._id = None
        self.game = None
        self.all_robots = []

    def wakeUp(self, playerId, numberOfPlayers, gameConfiguration):
        self._id = playerId
        self.game = Game()
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
        # 1. Assign missions to robots
        # This step determines which missions are assigned to which robots based on the game state
        self.game.getMissionManager().assignMissions(self.game, self._id)

        # 2. Assign paths and moves to robots
        # After missions are assigned, this step computes the optimal paths and movements for each robot
        PathManager.assignPaths(self.game, self.all_robots)

        # 3. Add priority rankings to robots
        # This step updates priorities based on blocking robots, closer to common cells and mission importance
        self.game.updateRobotBlockerAndPriorities()

        # 4. Select an action for each robot
        # Iterates through all robots and determines their next action
        result_actions = ""
        for robot in self.all_robots:
            action = ActionBuilder.action_decide(self.game, robot)
            result_actions += action + " "  # Append the action with a space separator

        # 5. Check the result actions and generate new ones if collisions/penalties are detected
        result_actions_checked = ActionBuilder.checkResultAction(self.game, result_actions)

        # Debugging: Uncomment to print the actions before and after validation
        # print(f"Player {self._id}: Without Check: {result_actions}")
        # print(f"Player {self._id}: With check:    {result_actions_checked}")

        # 6. Format the actions to match the expected game format
        return ActionBuilder.format_actions(result_actions_checked)

    def perceive(self, state):
        # "Update State of the game
        self.game.getModel().setOnState(state)
        # Debug: print the score of each player
        # print(f"Score: {self.game.getModel().setOnState(state)._scores}")
        # Update vip position
        self.game.getVIP().updateVipPosition(self.game.getModel())
        # Update robot position of each player
        for player in self.game.getPlayers():
            player.updateRobotsPosition(self.game.getModel())
            for robot in player.getRobots():
                self.game.addTile(robot.getPosition())

        # Update missions
        self.game.getMissionManager().updateAllMissions(self.game.getModel())

        self.game.getModel().render()
        # time.sleep(0.3)

    def sleep(self, result):
        pass