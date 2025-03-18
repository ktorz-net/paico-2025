class PathManager:
    @classmethod
    def assignPaths(cls, game, robots):
        """ Assign paths and moves for all given robots """
        # For each robot, resets their path and move,
        # then calculate and assign the best path and move depending on the current state of the game.
        for robot in robots:
            robot.resetPath()
            robot.resetMove()
            PathManager.assignPathAndMoves(game, robot)

    @classmethod
    def assignPathAndMoves(cls, game, robot):
        """ Assigns a path and moves to a robot based on its mission status """
        # 1. Robot go to the final cell of the mission
        if robot.has_mission_to_execute():
            cls.decidePathAndMoves(game, robot, robot.getMissionToExecute().getFinal())
        else:
            # 2. Robot go to the start cell of the mission
            # For this case, take always the first selected mission of the robot
            if len(robot.getSelectedMission()) > 0:
                cls.decidePathAndMoves(game, robot, robot.getSelectedMission()[0].getStart())
            else:
                # 3. Robot don't have mission
                robot.setHasMission(False)

        # The path of each robot needs to include its current position.
        # Why: This is necessary to handle cases where a robot might be on the path of another robot
        # and doesn't have any path left. This can occur either when a robot is waiting for a mission to be selected or validated,
        # or when it is in standby, meaning it has no assigned mission.
        robot.addFirstPath(robot.getPosition())

    @classmethod
    def decidePathAndMoves(cls, game, robot,  final_position):
        """
            Determines the best path and move for the robot to reach the final position.
            final_position: position to reach. Can be start or final cell of the mission
        """
        # Get multiple possible paths to reach the final position.
        multi_path = [game.path(robot.getPosition(), final_position)]
        # Retrieve and store the count of any obstacles (Robots and VIP) for each generated path
        count_obstacle = []
        for move, path in multi_path:
            count_robot, count_vip = PathManager.countObstacleOnPath(game, path, robot)
            count_obstacle.append((count_robot, count_vip))

        # Retrieve the path with the fewest obstacles
        min_obstacle_move = None
        min_obstacle_path = None
        min_obstacle_count = float('inf')
        for (count_robot, count_vip), (move, path) in zip(count_obstacle, multi_path):
            # If the path contains the vip, apply a penalty more important than a robot
            # Penalty collision with a vip is more important than a robot
            if count_vip == 1:
                count_vip += 10
            total_obstacles = count_robot + count_vip

            if total_obstacles < min_obstacle_count:
                min_obstacle_count = total_obstacles
                min_obstacle_path = path
                min_obstacle_move = move

        # Assign to robot the best path and moves
        robot.setPath(min_obstacle_path)
        robot.setMove(min_obstacle_move)

    @classmethod
    def countObstacleOnPath(cls, game, path, robot):
        """ Counts the number of obstacles (robots and VIPs) along the given path """
        # Get All robots without current robot
        robots = []
        for player in game.getPlayers():
            for robot_player in player.getRobots():
                if robot_player.getId() != robot.getId():
                    robots.append(robot_player)

        # For each tile in the given path, check if a robot or VIP is present on that tile
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