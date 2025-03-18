from .mission import Mission
from ..utils.evaluation import Evaluation
from ..utils.display import Display

class MissionManager:
    def __init__(self):
        # All Missions (list of mission.Mission): Represents all the missions without any owner
        self.all_missions = []
        # Selected Missions (list of mission.Mission): Represents all missions that the robot wants to execute.
        # This list provides an idea of where the robots intend to go.
        self.selected_missions = []
        # Available Missions(list of mission.Mission): Represents the free missions without an owner and not selected
        # by any robot
        self.available_missions = []

    def initMissions(self, model):
        """ Initialize all list missions based on the current game state """
        self.all_missions = self.updateAllMissions(model)
        self.selected_missions = self.updateSelectedMissions([])
        self.available_missions = self.updateAvailableMissions(model, [])

    def updateMissions(self, model, robots):
        """ Update all list missions based on the current game state and robots """
        self.all_missions = self.updateAllMissions(model)
        self.selected_missions = self.updateSelectedMissions(robots)
        self.available_missions = self.updateAvailableMissions(model, robots)

    # All Missions -----------------------------------------------------------------------------------------------------
    def getAllMissions(self):
        return self.all_missions

    def updateAllMissions(self, model):
        """ Update all missions based on the current game state """
        result_missions = []
        missions = model.missions()
        for i in range(len(missions)):
            mission = Mission(i+1, missions[i])
            if mission.getOwner() == 0:
                result_missions.append(mission)

        return result_missions

    # Selected Missions ------------------------------------------------------------------------------------------------
    def getSelectedMissions(self):
        return self.selected_missions

    def removeSelectedMission(self, mission):
        self.selected_missions.remove(mission)

    def updateSelectedMissions(self, robots):
        """ Update selected missions based on the robots """
        selected_missions = []
        for robot in robots:
            for mission in robot.getSelectedMission():
                    selected_mission_ids = [mission.getId() for mission in selected_missions]
                    if mission.getId() not in selected_mission_ids:
                        selected_missions.append(mission)
        return selected_missions

    # Available Missions -----------------------------------------------------------------------------------------------
    def getAvailableMissions(self):
        return self.available_missions

    def removeAvailableMission(self, mission):
        self.available_missions.remove(mission)

    def updateAvailableMissions(self, model, robots):
        """
            Update available missions based on the current game state and robots
            Loop 1. Get free mission from model and remove the selected missions from robots (only ids here)
            Loop 2. Find mission in all_missions list depending on their ids and add it to available_mission
        """
        available_missions = []
        free_mission = model.freeMissions()
        for robot in robots:
            selected_missions_robot = robot.getSelectedMission()
            for mission in selected_missions_robot:
                if mission.getId() in free_mission:
                    free_mission.remove(mission.getId())

        for index_mission in free_mission:
            for mission in self.all_missions:
                if mission.getId() == index_mission:
                    available_missions.append(mission)
                    break
        return available_missions

    # Assignator Missions ----------------------------------------------------------------------------------------------
    def assignMissions(self, game, player_id):
        """
        Assign Missions aims to add selected mission(s) for each robot.
        The goal is to determine, at the start of each game state, which robot, without mission to execute,
        is closest to each mission to adjust their trajectories accordingly.

        Multi-robot mode: Assign at least one unique mission to each robot
        to prevent multiple robots from heading to the same location.

        Multi-player mode: This method helps identify the best missions for each robot while considering opponents.
        It adapts the trajectory based on the map and the opponent's robots.
        Robots will avoid missions that give an advantage to the opponent.
        """
        # Get all robots that do not have a mission to execute
        robots = []
        for player in game.getPlayers():
            for robot in player.getRobots():
                if not robot.has_mission_to_execute():
                    robots.append(robot)

        # Update missions to get the latest version and retrieve all missions without owner
        self.updateMissions(game.getModel(), robots)
        free_missions = self.all_missions

        # While at least one robot does not have a selected mission or mission(s) is still available, continue
        while game.getPlayer(player_id).countRobotWithoutMissions() > 0 and len(free_missions) > 0:
            # 1. Evaluate missions and assign missions with the best possible score
            for robot in robots:
                best_score, best_mission = Evaluation.evaluate_all_missions(game, free_missions, robot)
                robot.selected_mission = best_mission
                robot.score_selected_mission = best_score

            # Update the missions after assigning them to the robots
            self.updateMissions(game.getModel(), robots)

            # 2. Process missions so that each robot has a unique mission
            for mission in self.selected_missions:
                best_score = float('-inf')
                best_robot_index = -1
                # Robots with fewer missions need to have priority in order to avoid the case
                # where robots with several missions have the same best mission as a robot
                # with only one selected mission.
                sorted_robots = self.sortRobots(robots)
                for i, robot in enumerate(sorted_robots):
                    if len(robot.getSelectedMission()) > 0:
                        for selected_mission in robot.getSelectedMission():
                            if mission == selected_mission:
                                # If the robot's score is better than the best one,
                                # the current robot becomes the best robot for this mission.
                                # If a robot performs better than the previous one,
                                # the previous robot loses the mission.
                                if robot.getScoreSelectedMission() > best_score:
                                    best_score = robot.getScoreSelectedMission()
                                    if best_robot_index != -1:
                                        sorted_robots[best_robot_index].removeSelectedMission(mission)
                                        sorted_robots[best_robot_index].setScoreSelectedMission(float('-inf'))
                                    best_robot_index = i
                                # Otherwise, the robot with the mission that has the worst score loses its mission.
                                else:
                                    sorted_robots[i].removeSelectedMission(mission)

                # 3. Remove the selected mission and the robot that has been assigned it
                # This step ensures that the next evaluation does not repeatedly select the same missions for robots
                # that have already been assigned the best missions, preventing the same missions from being assigned again.
                selected_mission_ids = []
                for robot in robots:
                    for mission_robot in robot.getSelectedMission():
                        if mission_robot.getId() not in selected_mission_ids:
                            selected_mission_ids.append(mission_robot.getId())
                if mission.getId() in selected_mission_ids and mission in free_missions:
                    free_missions.remove(mission)
                    result_robots = robots
                    for robot in robots:
                        for mission_robot in robot.getSelectedMission():
                            if mission_robot.getId() == mission.getId():
                                result_robots.remove(robot)

                    robots = result_robots

        # Debug: print all robot with selected and execute mission
        # Display.displayRobotMissions(game.getAllRobots())

        game.getPlayer(player_id).updateMissionBoolean()


    def sortRobots(self, robots):
        """ Sort robots based on the number of selected missions they have """
        def countSelectedMissions(robot):
            return len(robot.getSelectedMission())

        return sorted(robots, key=countSelectedMissions)

