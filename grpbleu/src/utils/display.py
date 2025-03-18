class Display:
    @classmethod
    def displayDistanceMatrix(cls, game):
        """ Display distance matrix in readable format """
        print("\nDistance Matrix (showing non-zero distances):")
        map_size = game.getModel().map().size()

        print("   ", end="")
        for j in range(map_size):
            if any(game.getDistance(i, j) != 0 for i in range(map_size)):
                print(f"{j:3}", end=" ")
        print()

        for i in range(map_size):
            if any(game.getDistance(i, j) != 0 for j in range(map_size)):
                print(f"{i:2} ", end="")
                for j in range(map_size):
                    if any(game.getDistance(k, j) != 0 for k in range(map_size)):
                        print(f"{game.getDistance(i, j):3}", end=" ")
                print()

    @classmethod
    def displayRobotMissions(cls, robots):
        """ Display selected and execute missions of robots in readable format """
        message = ""
        for robot in robots:
            message_selected_missions = ""
            for mission in robot.getSelectedMission():
                message_selected_missions += f"{mission.getId()} "
            message_missions_to_execute = ""
            if robot.getMissionToExecute() is not None:
                message_missions_to_execute += f"{robot.getMissionToExecute().getId()} "
            message_robot = f"Player {robot.getPlayerId()} Robot {robot.getId()} Mission Selected [{message_selected_missions}] Mission in progress [{message_missions_to_execute}]"
            message += message_robot + "\n"

        print(message)