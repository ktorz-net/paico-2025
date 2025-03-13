from ..utils.evaluation import Evaluation
from .mission import Mission

from ..utils.display import Display

class MissionManager:
    def __init__(self):
        self.all_missions = []
        self.selected_missions = []
        self.available_missions = []
        # L'objectif de ce tableau sera, une fois qu'il n'y aura plus de free mission, de chercher la mission des
        # des adversaires et de se placer sur la case final
        self.oponent_missions = []

    def initMissions(self, model):
        self.all_missions = self.updateAllMissions(model)
        self.selected_missions = self.updateSelectedMissions([])
        self.available_missions = self.updateAvailableMissions(model, [])

    def getAllMissions(self):
        return self.all_missions

    def getSelectedMissions(self):
        return self.selected_missions

    def getAvailableMissions(self):
        return self.available_missions

    def updateMissions(self, model, robots):
        self.all_missions = self.updateAllMissions(model)
        self.selected_missions = self.updateSelectedMissions(robots)
        self.available_missions = self.updateAvailableMissions(model, robots)

    def updateAllMissions(self, model):
        result_missions = []
        missions = model.missions()
        for i in range(len(missions)):
            mission = Mission(i+1, missions[i])
            if mission.getOwner() == 0:
                result_missions.append(mission)

        return result_missions

    def updateSelectedMissions(self, robots):
        selected_missions = []
        for robot in robots:
            for mission in robot.getSelectedMission():
                    selected_mission_ids = [mission.getId() for mission in selected_missions]
                    if mission.getId() not in selected_mission_ids:
                        selected_missions.append(mission)
        return selected_missions

    def updateAvailableMissions(self, model, robots):
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

    def removeSelectedMission(self, mission):
        self.selected_missions.remove(mission)

    def removeAvailableMission(self, mission):
        self.available_missions.remove(mission)

    def assignMissions(self, game, player_id):
        # Before assigning a mission, need to run "updateAllMissions(model)"
        robots = []
        for player in game.getPlayers():
            for robot in player.getRobots():
                if not robot.has_mission_to_execute():
                    robots.append(robot)

        self.updateMissions(game.getModel(), robots)
        free_missions = self.all_missions

        while game.getPlayer(player_id).countRobotWithoutMissions() > 0 and len(free_missions) > 0:
            # 1. Associer des missions avec le meilleur score possible
            for robot in robots:
                best_score, best_mission = Evaluation.evaluate_all_missions(game, free_missions, robot)
                robot.selected_mission = best_mission
                robot.score_selected_mission = best_score

            self.updateMissions(game.getModel(), robots)
            # 2. Traiter les missions de chaque robot pour que chaque robot ait une mission différente
            for mission in self.selected_missions:
                # TODO; ca devrait etre - infini ici
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
                                if robot.getScoreSelectedMission() > best_score:
                                    best_score = robot.getScoreSelectedMission()
                                    if best_robot_index != -1:
                                        sorted_robots[best_robot_index].removeSelectedMission(mission)
                                        sorted_robots[best_robot_index].setScoreSelectedMission(float('-inf'))
                                    best_robot_index = i
                                else:
                                    sorted_robots[i].removeSelectedMission(mission)

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

        # Display.displayRobotMissions(game.getAllRobots())

    # TODO:  Déplacer cette méthode quelque part
    def sortRobots(self, robots):
        def countSelectedMissions(robot):
            return len(robot.getSelectedMission())

        return sorted(robots, key=countSelectedMissions)

