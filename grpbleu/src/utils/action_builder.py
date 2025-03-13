import random
from .evaluation import Evaluation

class ActionBuilder:
    @classmethod
    def action_decide(cls, game, robot):
        # 1. Choisir la mission
        if len(robot.getSelectedMission()) > 0 and robot.getSelectedMission()[0].getStart() == robot.getPosition():
            action = f"mission {robot.getId()} {robot.getSelectedMission()[0].getId()}"
            robot.addMissionToExecute()
            return action

        # 2. Valider la mission
        if robot.getMissionToExecute() is not None and robot.getMissionToExecute().getFinal() == robot.getPosition():
            action = f"mission {robot.getId()} {robot.getMissionToExecute().getId()}"
            robot.removeMissionToExecute()
            return action

        # 3. Robot se dirige vers le point final ou le point de la mission
        return cls.decide_move_action(game, robot)

    @classmethod
    def decide_move_action(cls, game, robot):
        # Le robot n'a pas de mouvements à faire
        if len(robot.getMove()) == 0 or robot.getMove()[0] == 0:
            neighbours, neighbours_team_robot, neighbours_opponent_robot = robot.getNeighbours(game)
            available_cells = robot.getAvailablePosition(neighbours, neighbours_team_robot, neighbours_opponent_robot)
            return cls.getRandomAction(game, available_cells, robot)

        safety = Evaluation.evaluate_path_safety(game, robot.getPath())
        if safety > 0.01:
            # self.logger.info(f"Chemin choisi avec sécurité {safety:.2f}")
            return f"move {robot.getId()} {robot.getMove()[0]}"
        else:
            # TODO: Utiliser un chemin jugé sur -> Multi Path Finding
            return f"move {robot.getId()} 0"

    @classmethod
    def getRandomAction(cls, game, available_cells, robot):
        action = f"move {robot.getId()} "
        if len(available_cells) > 1:
            cell = random.choice(available_cells)
        else:
            cell = available_cells[0]

        direction = ActionBuilder.getDirection(game, robot, cell)
        action += str(direction)
        return action

    @classmethod
    def getDirection(cls, game, robot, cell):
        directions = [3, 6, 9, 12]
        for direction in directions:
            target_tile = game.getModel().map().clockposition(robot.getPosition(), direction)
            if target_tile == cell:
                return direction

        return 0

    @classmethod
    def format_actions(cls, actions_str):
        """Formate les actions pour respecter l'ordre: Mission ... Move ...

        Args:
            actions_str (str): Chaîne contenant toutes les actions non formatées

        Returns:
            str: Actions formatées dans l'ordre correct
        """
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
        actions_split = actions.split()
        action_list = [" ".join(actions_split[i:i + 3]) for i in range(0, len(actions_split), 3)]
        robots = game.getAllRobots()
        target_tile_robots = cls.getNewTiles(game, robots, action_list)
        same_tiles = cls.getSameTiles(target_tile_robots)
        collision_robots = cls.getCollisionRobots(game, robots, action_list, same_tiles)
        new_actions = cls.getNewActions(collision_robots)
        final_actions = cls.getFinalActions(new_actions, action_list)

        return " ".join(final_actions)

    @classmethod
    def getNewTiles(cls, game, robots, actions):
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
        def countPriority(robot):
            return sum(robot.getPriority())  # Compte le nombre de True (car True = 1 et False = 0 en Python)

        return sorted(robots, key=countPriority, reverse=True)  # Tri décroissant (plus de True = plus prioritaire)

    @classmethod
    def getNewActions(cls, robots):
        new_actions = []
        if len(robots) > 0:
            sorted_robot = cls.sortRobotsByPriorities(robots)

            for i, robot in enumerate(sorted_robot):
                if i != 0:
                    new_actions.append(f"move {robot.getId()} 0")

        return new_actions

    @classmethod
    def getFinalActions(cls, new_actions, actions):
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
