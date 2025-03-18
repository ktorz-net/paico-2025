from ..robot import Robot

class Player:
    def __init__(self, id):
        self.id = id
        self.robots = []
        self.numberOfRobots = 0

    def initRobots(self, model):
        """ Init robots depending on the configuration of the game"""
        self.numberOfRobots = model.numberOfMobiles(iPlayer=self.id)
        for i in range(self.numberOfRobots):
            robot_position = model.mobilePosition(self.id, i+1)
            robot = Robot(i+1, self.id, robot_position)
            self.robots.append(robot)

    # Id ---------------------------------------------------------------------------------------------------------------
    def getId(self):
        return self.id

    # Robots ----------------------------------------------------------------------------------------------------------
    def getRobots(self):
        return self.robots

    # Robot Positions
    def updateRobotsPosition(self, model):
        """ Update the position of each robot for the player based on the current state of the game """
        for robot in self.robots:
            robot.setPosition(model.mobilePosition(self.id, robot.getId()))

    # Robot Missions Count
    def countRobotWithoutMissions(self):
        """ Count the robots that don't have any mission to execute or have not selected a mission """
        count = 0
        for robot in self.robots:
            if len(robot.getSelectedMission()) == 0 and robot.getMissionToExecute() is None:
                count += 1
        return count

    # Robot Mission Boolean
    def updateMissionBoolean(self):
        """
            Update the mission status of each robot based on whether they have a mission to execute,
            selected mission(s), or no mission(s) (stand by)
        """
        for robot in self.robots:
            if len(robot.getSelectedMission()) > 0 or robot.getMissionToExecute() is not None:
                robot.setHasMission(True)
            else:
                robot.setHasMission(False)














