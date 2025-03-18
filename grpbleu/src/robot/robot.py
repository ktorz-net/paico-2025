import random

class Robot:
    def __init__(self, id, player_id, position):
        self.id = id
        self.player_id = player_id
        self.position = position

        # Mission ------------------------------------------------------------------------------------------------------
        # Mission to execute: Represents the mission that has been selected and assigned to the robot.
        # Example on the ui: .1 -4 to 12.   (Team-1)
        self.mission_to_execute = None
        # Selected Missions: Represents the best missions for the robot to pursue
        # based on the current state of the game (not yet assigned).
        self.selected_mission = []
        # Score associated with the selected mission (used for determining priority).
        self.score_selected_mission = float('-inf')
        # Boolean indicating whether the mission is in the process of being executed or has been selected.
        self.has_execute_mission = False
        # 'has_mission' is a more general parameter to ensure that the robot either has a selected mission
        # or a mission to execute.
        self.has_mission = False

        # Path ---------------------------------------------------------------------------------------------------------
        self.path = []
        self.move = []

        # Priority -----------------------------------------------------------------------------------------------------
        # Priority is a list of Boolean values (initialized later).
        # The idea is to set 'True' for each robot's index, and by counting the number of 'True' values,
        # we can determine which robot has the highest priority.
        self.priority = []
        # Robots present on the path that can block the movement of the robot.
        self.block_robots = []
        # Common cells where a 'blocking robot' can be found in the robot's path.
        # The indices of 'block_robots' and 'common_cells' are aligned.
        # Example: To access the first blocking robot, use index [0] in 'block_robots',
        # and you can refer to the corresponding cell in 'common_cells' using the same index [0].
        self.common_cells = []
        # Count Wait Move: is used to track how many times consecutively a robot does not move.
        # A robot does not move if it yields priority to another robot (currently, this can be adjusted in the future)
        # or when it is in standby mode.
        self.count_wait_to_move = 0

    # Id ---------------------------------------------------------------------------------------------------------------
    def getId(self):
        return self.id

    # Player Id --------------------------------------------------------------------------------------------------------
    def getPlayerId(self):
        return self.player_id

    # Position ---------------------------------------------------------------------------------------------------------
    def getPosition(self):
        return self.position

    def setPosition(self, position):
        self.position = position

    # Path -------------------------------------------------------------------------------------------------------------
    def getPath(self):
        return self.path

    def setPath(self, path):
        self.path = path

    def addFirstPath(self, path):
        """
            Add the variable 'path' at the beginning of the path list.
            Typically, this represents the current position of the robot, which is needed for priority part.
        """
        self.path.insert(0, path)

    def resetPath(self):
        self.path = []

    # Move -------------------------------------------------------------------------------------------------------------
    def getMove(self):
        return self.move

    def setMove(self, move):
        self.move = move

    def resetMove(self):
        self.move = []

    # Common Cells -----------------------------------------------------------------------------------------------------
    def getCommonCells(self):
        return self.common_cells

    def addCommonCells(self, common_cell):
        self.common_cells.append(common_cell)

    def resetCommonCells(self):
        self.common_cells = []

    # Block Robots -----------------------------------------------------------------------------------------------------
    def getBlockRobots(self):
        return self.block_robots

    def addBlockRobot(self, robot):
        self.block_robots.append(robot)

    def resetBlockRobots(self):
        self.block_robots = []

    # Priority ---------------------------------------------------------------------------------------------------------
    def initPriority(self, numberOfRobot):
        """ Initialize the priority list by adding 'True' Boolean values for the number of robots present in the game """
        for _ in range(numberOfRobot):
            self.priority.append(True)

    def getPriority(self):
        return self.priority

    def setPriority(self, index_robot, priority):
        self.priority[index_robot] = priority

    def resetPriority(self):
        size_priority = len(self.priority)
        self.priority = []
        self.initPriority(size_priority)

    def getSumPriority(self):
        sum_priority = 0
        for priority in self.priority:
            if priority:
                sum_priority += 1
        return sum_priority


    # Count Wait to move -----------------------------------------------------------------------------------------------
    def getCountWaitToMove(self):
        return self.count_wait_to_move

    def increaseCountWaitToMove(self):
        self.count_wait_to_move += 1

    def resetCountWaitToMove(self):
        self.count_wait_to_move = 0

    # Mission to execute -----------------------------------------------------------------------------------------------
    def getMissionToExecute(self):
        return self.mission_to_execute

    def addMissionToExecute(self):
        """
            Adding a mission to execute.
            When a mission is assign to the robot, it implies that the selected mission and the associated score are reset (empty).
        """
        self.mission_to_execute = self.selected_mission[0]
        self.selected_mission = []
        self.score_selected_mission = float('-inf')
        self.has_execute_mission = True

    def removeMissionToExecute(self):
        self.mission_to_execute = None
        self.has_execute_mission = False

    # Has mission to execute -------------------------------------------------------------------------------------------
    def has_mission_to_execute(self):
        return self.has_execute_mission

    # Has Mission ------------------------------------------------------------------------------------------------------
    def getHasMission(self):
        return self.has_mission

    def setHasMission(self, value):
        self.has_mission = value

    # Selected Mission -------------------------------------------------------------------------------------------------
    def getSelectedMission(self):
        return self.selected_mission

    def removeSelectedMission(self, mission):
        self.selected_mission.remove(mission)

    # Score Selected Mission
    def getScoreSelectedMission(self):
        return self.score_selected_mission

    def setScoreSelectedMission(self, score):
        self.score_selected_mission = score

    # Action -----------------------------------------------------------------------------------------------------------
    def getNeighbours(self, game):
        """ Retrieve list of neighbours, neighbours team bots and opponent bots"""
        # Get neighbours depending on the current game state and the position of the robot
        neighbours = game.getModel().map().neighbours(self.position)
        robots = game.getAllRobots()
        neighbours_team_robot = []
        neighbours_opponent_robot = []
        # For each robot and each neighboring cell, retrieve the robots that occupy the same cells.
        # Depending on the Id of the robot in the loop (whether it matches the current robot's Id or not),
        # the robot can be added to a different list (team or opponent).
        for cell in neighbours:
            if cell != self.getPosition():
                for robot_player in robots:
                    if robot_player.getId() != self.getId():
                        if robot_player.getPosition() == cell:
                            if robot_player.getPlayerId() == self.getPlayerId():
                                neighbours_team_robot.append(robot_player)
                            else:
                                neighbours_opponent_robot.append(robot_player)

        return neighbours, neighbours_team_robot, neighbours_opponent_robot

    def getAvailableNeighbourPositions(self, neighbours, neighbours_team_robot, neighbours_opponent_robot):
        """ Retrieve available neighbour positions by removing self position, team's robot and opponent's robot """
        # If there are no team robots in the neighboring cells but there are opponent robots,
        # no movement is given (move ID [0] indicates no position are available, so stay in standby).
        if len(neighbours_team_robot) == 0 and len(neighbours_opponent_robot) > 0:
            return [0]
        else:
            # Remove the position of the robot
            if self.position in neighbours:
                neighbours.remove(self.position)
            available_cells = neighbours
            # Remove positions that are already occupied by a team's robot (to avoid collisions).
            for robot in neighbours_team_robot:
                robot_position = robot.getPosition()
                if robot_position in available_cells:
                    available_cells.remove(robot_position)
            # Remove positions that are already occupied by an opponent's robot (to avoid collisions).
            for robot in neighbours_opponent_robot:
                robot_position = robot.getPosition()
                if robot_position in available_cells:
                    available_cells.remove(robot_position)
            # If the rest of the list is empty (== []), return a list with [0]
            # (indicating no positions are available, so the robot stays in standby)
            if len(available_cells) == 0:
                available_cells = [0]

            return available_cells

