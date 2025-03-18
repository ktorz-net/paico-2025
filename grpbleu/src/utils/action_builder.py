import random
from .evaluation import Evaluation

class ActionBuilder:
    @classmethod
    def action_decide(cls, game, robot):
        """ Retrieve an action for the robot """
        # 1. Robot choose the mission
        if len(robot.getSelectedMission()) > 0 and robot.getSelectedMission()[0].getStart() == robot.getPosition():
            action = f"mission {robot.getId()} {robot.getSelectedMission()[0].getId()}"
            robot.addMissionToExecute()
            return action

        # 2. Robot validate the mission
        if robot.getMissionToExecute() is not None and robot.getMissionToExecute().getFinal() == robot.getPosition():
            action = f"mission {robot.getId()} {robot.getMissionToExecute().getId()}"
            robot.removeMissionToExecute()
            return action

        # 3.Robot go to Start or Final cell of the mission
        return cls.decide_move_action(game, robot)

    @classmethod
    def decide_move_action(cls, game, robot):
        """ Depending on the robot's next move and its safety, retrieve the appropriate action/movement to perform """
        # If the robot doesn't have any move to make (meaning it doesn't have a mission),
        # it will perform a random action in the neighboring cells to keep moving continuously.
        if len(robot.getMove()) == 0 or robot.getMove()[0] == 0:
            neighbours, neighbours_team_robot, neighbours_opponent_robot = robot.getNeighbours(game)
            available_cells = robot.getAvailableNeighbourPositions(neighbours, neighbours_team_robot, neighbours_opponent_robot)
            return cls.getRandomAction(game, available_cells, robot)
        # Evaluate the safety of the next movement.
        safety = Evaluation.evaluate_path_safety(game, robot.getPath())
        # If the next move is safe, take it
        if safety > 0.01:
            return f"move {robot.getId()} {robot.getMove()[0]}"
        # Else, stand by
        # TODO: Area for improvement: Use multi-path finding here to always perform an action (movement).
        else:
            return f"move {robot.getId()} 0"

    @classmethod
    def getRandomAction(cls, game, available_cells, robot):
        """ From the available cell, retrieve a random cell to perform a new action/movement """
        action = f"move {robot.getId()} "
        if len(available_cells) > 1:
            cell = random.choice(available_cells)
        else:
            cell = available_cells[0]

        # Get the direction to move from the robot's current position to a new random cell.
        direction = ActionBuilder.getDirection(game, robot, cell)
        action += str(direction)
        return action

    @classmethod
    def getDirection(cls, game, robot, cell):
        """ Retrieve the direction from the robot's current position (start cell) to the target cell (final cell) """
        directions = [3, 6, 9, 12]
        for direction in directions:
            target_tile = game.getModel().map().clockposition(robot.getPosition(), direction)
            if target_tile == cell:
                return direction
        # If no target tile corresponds to the cell, return 0, indicating no movement to perform.
        return 0

    @classmethod
    def format_actions(cls, actions_str):
        """ Format the actions to follow the order: Mission ... Move ... """
        actions = actions_str.split()
        missions = []
        moves = []
        i = 0
        while i < len(actions):
            if actions[i] == "mission":
                missions.append(f"{actions[i + 1]} {actions[i + 2]}")
                i += 3
            elif actions[i] == "move":
                moves.append(f"{actions[i + 1]} {actions[i + 2]}")
                i += 3
            else:
                i += 1

        result = ""
        if missions:
            result += "mission " + " ".join(missions)
        if moves:
            if result:
                result += " "
            result += "move " + " ".join(moves)
        return result

    @classmethod
    def checkResultAction(cls, game, actions):
        """ Check the actions to be performed and generate new ones for actions that involve collisions and penalties """
        # actions -> mission 2 1 move 1 3 move 3 0
        # actions_split -> [mission, 2, 1, move, 1, 3, move, 3, 0]
        actions_split = actions.split()
        # action_list -> [mission 2 1, move 1 3, move 3 0]
        action_list = [" ".join(actions_split[i:i + 3]) for i in range(0, len(actions_split), 3)]
        robots = game.getAllRobots()
        # From the 'move' action, retrieve the target tile for each robot
        target_tile_robots = cls.predictNextTiles(game, robots, action_list)
        # From the target tiles, identify tiles that are the same
        same_tiles = cls.getSameTiles(target_tile_robots)
        # From the same_tiles table, identify robots that will collide and incur penalties
        collision_robots = cls.getCollisionRobots(game, robots, action_list, same_tiles)
        # Generate new actions to avoid penalties caused by collisions
        new_actions = cls.getNewActions(collision_robots)
        # Create final actions that will be performed after resolving collisions and penalties
        final_actions = cls.getFinalActions(new_actions, action_list)
        return " ".join(final_actions)

    @classmethod
    def predictNextTiles(cls, game, robots, actions):
        """ With the actions and the robots (positions), predict the next cell of the robot"""
        target_tile_robots = [0 for _ in range(len(robots))]
        for i, robot in enumerate(robots):
            for action in actions:
                action_split = action.split(" ")
                if action_split[0] == "move":
                    if robot.getId() == int(action_split[1]):
                        target_tile = game.getModel().map().clockposition(robot.getPosition(), int(action_split[2]))
                        target_tile_robots[i] = target_tile

        return target_tile_robots

    @classmethod
    def getSameTiles(cls, tiles):
        """ From a list of tiles, retrieve only the tiles that appear more than once """
        same_tiles = []
        tiles_set = set(tiles)
        if 0 in tiles_set:
            tiles.remove(0)
        for tile_set in tiles_set:
            count = 0
            for tile in tiles:
                if tile == tile_set:
                    count += 1
                if count > 1:
                    same_tiles.append(tile)
        return same_tiles

    @classmethod
    def getCollisionRobots(cls, game, robots, actions, tiles):
        """ Retrieve the robots that have the same next target tile """
        collision_robots = []
        if tiles:
            for action in actions:
                action_split = action.split(" ")
                if action_split[0] == "move":
                    move_direction = int(action_split[2])
                    for robot in robots:
                        target_tile = game.getModel().map().clockposition(robot.getPosition(), move_direction)
                        if target_tile in tiles and robot not in collision_robots:
                            collision_robots.append(robot)

        return collision_robots

    @classmethod
    def sortRobotsByPriorities(cls, robots):
        """ Sort the robots based on their number of priorities over other robots """
        def countPriority(robot):
            # Count the number of True values (since True = 1 and False = 0 in Python)
            return sum(robot.getPriority())
        # Sort in descending order (more True values = higher priority)
        return sorted(robots, key=countPriority, reverse=True)

    @classmethod
    def getNewActions(cls, robots):
        """ Retrieve new actions for given robots """
        new_actions = []
        if len(robots) > 0:
            sorted_robot = cls.sortRobotsByPriorities(robots)
            for i, robot in enumerate(sorted_robot):
                if i != 0:
                    # TODO: Area of improvement: Instead of having robots wait for the highest-priority one to pass,
                    #  idea: they could move in the opposite direction to avoid potential blockages.
                    new_actions.append(f"move {robot.getId()} 0")
        return new_actions

    @classmethod
    def getFinalActions(cls, new_actions, actions):
        """
            Build the new final actions by replacing actions from the original list with actions from the new list

            actions = [move 1 3, mission 2 4, move 3 6]
            new_actions = [move 1, 0, move 3 6]
            -> final_actions = [move 1,0, move 3 6, mission 2 4]
        """
        if len(new_actions) > 0:
            final_actions = list(new_actions)
            final_actions_id = {action.split(" ")[1] for action in new_actions}
            for action in actions:
                action_id = action.split(" ")[1]
                if action_id not in final_actions_id:
                    final_actions.append(action)
                    final_actions_id.add(action_id)
            return final_actions
        return actions
