from ..robot import Robot

class Player:
    def __init__(self, id):
        self.id = id
        self.robots = []
        self.numberOfRobots = 0

    def initRobots(self, model):
        self.numberOfRobots = model.numberOfMobiles(iPlayer=self.id)
        for i in range(self.numberOfRobots):
            robot_position = model.mobilePosition(self.id, i+1)
            robot = Robot(i+1, self.id, robot_position)
            self.robots.append(robot)

    def getId(self):
        return self.id

    def getRobots(self):
        return self.robots

    def updateRobotsPosition(self, model):
        for robot in self.robots:
            robot.setPosition(model.mobilePosition(self.id, robot.getId()))

    def countRobotWithoutMissions(self):
        count = 0
        for robot in self.robots:
            if len(robot.getSelectedMission()) == 0 and robot.getMissionToExecute() is None:
                count += 1
        return count

    # Not used for now
    def countRobotWithoutSelectedMissions(self):
        count = 0
        for robot in self.robots:
            if len(robot.getSelectedMission()) == 0:
                count += 1

        return count

    def resetRobotPriority(self, robots):
        for robot in robots:
            robot.resetPriority()
            robot.resetBlockRobots()
            robot.resetCommonCells()

    def updateMissionBoolean(self):
        for robot in self.robots:
            if len(robot.getSelectedMission()) > 0 or robot.getMissionToExecute() is not None:
                robot.setHasMission(True)
            else:
                robot.setHasMission(False)

    def updateRobotBlockerAndPriorities(self, distances, robots):
        self.resetRobotPriority(robots)
        self.updateBlockRobotsAndCommonCells(robots)
        self.updatePriorities(distances, robots)

    def updateBlockRobotsAndCommonCells(self, robots):
        for i in range(len(robots)):
            for j in range(len(robots)):
                if i != j:
                    robot_a = robots[i]
                    robot_b = robots[j]
                    path_a = robot_a.getPath()
                    path_b = robot_b.getPath()
                    common_cell = set(path_a) & set(path_b)
                    if common_cell:
                        robot_a.addBlockRobot(robot_b)
                        robot_a.addCommonCells(common_cell)

    def updatePriorities(self, distances, robots):
        for robot in robots:
            block_robots = robot.getBlockRobots()
            if len(block_robots) > 0:
                common_cells = robot.getCommonCells()
                for common_cells_robot, block_robot in zip(common_cells, block_robots):
                    for common_cell in common_cells_robot:
                        distance_robot = distances[robot.getPosition()][common_cell]
                        distance_block_robot = distances[block_robot.getPosition()][common_cell]
                        if distance_robot > distance_block_robot:
                            robot.setPriority(block_robot.getId() - 1, False)














