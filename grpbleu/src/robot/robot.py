import random

class Robot:
    def __init__(self, id, player_id, position):
        self.id = id
        self.player_id = player_id
        self.position = position

        self.mission_to_execute = None # Mission choisie (Team-1)
        self.selected_mission = []     # Mission que le robot souhaite executer (pas encore choisie)
        self.score_selected_mission = float('-inf') # Score associé à la mission selectionnée
        self.has_execute_mission = False   # Booléen permettant de savoir si la mission est entre d'etre executée ou choisie

        self.path = []
        self.move = []
        self.priority = []
        self.block_robots = []
        self.common_cells = []
        self.count_wait_to_move = 0
        self.has_mission = False # Paramètre plus global pour s'assurer que Robot a une mission choisi ou une mission à exécuter
        self.last_action = None

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
        self.mission_to_execute = self.selected_mission[0]
        self.selected_mission = []
        self.score_selected_mission = float('-inf')
        self.has_execute_mission = True

    def removeMissionToExecute(self):
        self.mission_to_execute = None
        self.has_execute_mission = False

    # Has mission to execute
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

    # ------------------------------------------------------------------------------------------------------------------
    # Action
    # Pas encore utilisé
    def getLastAction(self):
        return self.last_action

    def setLastAction(self, action):
        self.last_action = action

    def getNeighbours(self, game):
        neighbours = game.getModel().map().neighbours(self.position)
        robots = game.getAllRobots()
        neighbours_team_robot = []
        neighbours_opponent_robot = []
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

    def getAvailablePosition(self, neighbours, neighbours_team_robot, neighbours_opponent_robot):
        # On en bouge pas pour les robots du joueur adverse
        if len(neighbours_team_robot) == 0 and len(neighbours_opponent_robot) > 0:
            return [0]
        else:
            if self.position in neighbours:
                neighbours.remove(self.position)
            available_cells = neighbours
            for robot in neighbours_team_robot:
                robot_position = robot.getPosition()
                if robot_position in available_cells:
                    available_cells.remove(robot_position)

            for robot in neighbours_opponent_robot:
                robot_position = robot.getPosition()
                if robot_position in available_cells:
                    available_cells.remove(robot_position)

            if len(available_cells) == 0:
                available_cells = [0]


            return available_cells

