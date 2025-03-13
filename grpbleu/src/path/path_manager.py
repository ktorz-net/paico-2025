class PathManager:
    @classmethod
    def assignPaths(cls, game, robots):
        for robot in robots:
            robot.resetPath()
            robot.resetMove()
            PathManager.assignPathAndMoves(game, robot)

    @classmethod
    def assignPathAndMoves(cls, game, robot):
        # Robot se dirige vers la fin de la mission
        if robot.has_mission_to_execute():
            cls.decidePathAndMoves(game, robot, robot.getMissionToExecute().getFinal())
        else:
            # Robot se dirige vers le début de la mission
            if len(robot.getSelectedMission()) > 0:
                cls.decidePathAndMoves(game, robot, robot.getSelectedMission()[0].getStart())
            else:
                # Robot n'a pas de missions
                robot.setHasMission(False)

        robot.addFirstPath(robot.getPosition())

    @classmethod
    def decidePathAndMoves(cls, game, robot,  final_position):
        multi_path = [game.path(robot.getPosition(), final_position)]
        # TODO : Probème récursivité
        # multi_path = game.multi_path(robot.getPosition(), final_position)
        # Retourne le nombre d'obstacles (robot et vip distincts)
        count_obstacle = []
        for move, path in multi_path:
            count_robot, count_vip = PathManager.countObstacleOnPath(game, path, robot)
            count_obstacle.append((count_robot, count_vip))

        # Récupère le chemin et les mouvements avec le moins d'obstacles possibles
        min_obstacle_move = None
        min_obstacle_path = None
        min_obstacle_count = float('inf')
        for (count_robot, count_vip), (move, path) in zip(count_obstacle, multi_path):
            if count_vip == 1:
                count_vip += 10  # On veut éviter au maximum de croiser un vip donc s'il est sur notre chemin, c'est mauvais
            total_obstacles = count_robot + count_vip

            if total_obstacles < min_obstacle_count:
                min_obstacle_count = total_obstacles
                min_obstacle_path = path
                min_obstacle_move = move

        # Associe le meilleur chemin et mouvements au robot
        robot.setPath(min_obstacle_path)
        robot.setMove(min_obstacle_move)

    # TODO: Not used
    @classmethod
    def hasObstacleOnPath(cls, game, path, robot):
        robots = []
        for player in game.getPlayers():
            for robot_player in player.getRobots():
                if robot_player.getId() != robot.getId():
                    robots.append(robot_player)

        for tile in path:
            for robot_player in robots:
                if tile == robot_player.getPosition():
                    return True
                if game.getVIP().getVipExistence():
                    if tile == game.getVIP().getPosition():
                        return True

        return False

    @classmethod
    def countObstacleOnPath(cls, game, path, robot):
        robots = []
        for player in game.getPlayers():
            for robot_player in player.getRobots():
                if robot_player.getId() != robot.getId():
                    robots.append(robot_player)

        count_robot = 0
        count_vip = 0
        for tile in path:
            for robot_player in robots:
                if tile == robot_player.getPosition():
                    count_robot += 1
                if game.getVIP().getVipExistence():
                    if tile == game.getVIP().getVipPosition():
                        count_vip += 1

        return count_robot, count_vip