from hacka.games.py421 import GameSolo


class Evaluation:
    @classmethod
    def evaluate_all_missions(cls, game, missions, robot):

        """Evaluate and select best mission"""
        best_mission = []
        scores = []

        # Évalue chaque mission disponible pour trouver la meilleure
        for mission in missions:
            scores.append(Evaluation.evaluate_mission(game, mission, robot))

        best_score = max(scores)
        indexes_best_score = [i for i, score in enumerate(scores) if score == best_score]
        for i in indexes_best_score:
            best_mission.append(missions[i])

        return best_score, best_mission

    @classmethod
    def evaluate_mission(cls, game, mission, robot):
        """Évalue l'intérêt d'une mission avec gestion optionnelle du VIP"""

        start_dist = game.getDistance(robot.getPosition(), mission.getStart())
        completion_dist = game.getDistance(mission.getStart(), mission.getFinal())
        total_dist = start_dist + completion_dist
        score = float(1000 - total_dist)

        safety_coefficient, defeated_vip_coefficient = cls.evaluate_coefficient_vip(game, mission, robot, start_dist)
        score *= safety_coefficient
        score *= defeated_vip_coefficient
        coefficient_opponent = cls.evaluate_oponent_coefficient(game, robot)
        score *= coefficient_opponent
        coefficient_team = cls.evaluate_team_coefficient(game, robot)
        score *= coefficient_team
        return score

    @classmethod
    def evaluate_coefficient_vip(cls, game, mission, robot, start_dist):
        safety_coefficient = 1.0
        defeated_vip_coefficient = 1.0

        vip_pos = game.getVIP().getVipPosition()
        # Mode avec VIP
        if vip_pos is not None:
            try:
                _, _, safety_coefficient = game.getVIP().avoid_vip(game, robot.getPosition(), mission.start)
                # Bonus si on peut battre le VIP
                vip_dist_start = game.getDistance(vip_pos, mission.start)
                if vip_dist_start > start_dist:
                    defeated_vip_coefficient = 1.2
            except IndexError:
                print("Erreur lors du calcul des distances avec le VIP")

            return safety_coefficient, defeated_vip_coefficient
        # Mode sans VIP: bonus pour missions proches
        else:
            if start_dist < 5:
                safety_coefficient = 1.5
            return safety_coefficient, 1.0

    @classmethod
    def evaluate_oponent_coefficient(cls, game, robot):
        oponent_coefficient = 1.0
        if len(game.getPlayers()) > 1:
            oponent_id = robot.getPlayerId() % 2 + 1
            for oponent_robot in game.getPlayer(oponent_id).getRobots():
                distance = game.getDistance(robot.getPosition(), oponent_robot.getPosition())
                if distance == 1:  # Menace proche
                    oponent_coefficient -= 0.4
                elif distance == 2:  # Menace modérée
                    oponent_coefficient -= 0.2

        return oponent_coefficient

    @classmethod
    def evaluate_team_coefficient(cls, game, robot):
        team_coefficient = 1.0
        for team_robot in game.getPlayer(robot.getPlayerId()).getRobots():
            if team_robot.getId() != robot.getId():
                distance = game.getDistance(robot.getPosition(), team_robot.getPosition())
                if distance == 1:  # Menace proche
                    team_coefficient -= 0.4
                elif distance == 2:  # Menace modérée
                    team_coefficient -= 0.2

        return team_coefficient


    @classmethod
    def reevaluate_mission(cls, game, robot_index, robot_pos, current_mission):
        """Réévalue la mission actuelle et décide de la changer si nécessaire"""
        available_missions = game.getModel().freeMissions()
        available_missions.append(current_mission)
        best_mission = Evaluation.evaluate_all_missions(available_missions, robot_pos, game.getVIP().get_vip_position())

        if best_mission != current_mission:
            return f"mission {robot_index} {best_mission}"
        return None

    @classmethod
    def evaluate_path_safety(cls, game, path):
        """Évalue la sécurité d'un chemin"""
        if not path or not isinstance(path, list):
            return 0

        # Init VIP Position
        if game.getVIP().getVipExistence():
            vip_pos = game.getVIP().getVipPosition()
        else:
            vip_pos = None
        # Init Current Robot Position
        current_robot_pos = path[0]
        # Get Other Robots (Team or Opponent)
        other_robots = []
        for player in game.getPlayers():
            for robot in player.getRobots():
                if robot.getPosition != current_robot_pos:
                    other_robots.append(robot)

        safety_score = 1.0
        map_size = game.getModel().map().size()

        # Pénalité pour proximité avec autres robots
        for pos in path:
            if not isinstance(pos, int) or pos >= map_size:
                continue

            for robot in other_robots:
                if robot.getPosition() is None or robot.getPosition() >= map_size:
                    continue
                try:
                    dist = game.getDistance(pos, robot.getPosition())
                    if dist < 2:
                        safety_score *= 0.5
                    elif dist < 4:
                        safety_score *= 0.8
                except IndexError:
                    continue

        # Pénalité pour proximité avec VIP
        if vip_pos is not None and vip_pos < map_size:
            for pos in path:
                if not isinstance(pos, int) or pos >= map_size:
                    continue

                if pos == vip_pos:
                    safety_score *= 0.1
                elif pos in game.getModel().map().neighbours(vip_pos):
                    safety_score *= 0.5

        return safety_score