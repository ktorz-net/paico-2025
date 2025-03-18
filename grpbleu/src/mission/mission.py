class Mission:
    def __init__(self, number_mission, mission):
        # Id: Number of the mission
        self.id = number_mission
        # Start: First cell of the mission
        # Final: Last cell of the mission
        # Reward: Price given after finishing the mission
        # Owner: User assigned to the mission
        self.start, self.final, self.reward, self.owner = mission.list()

    # Id ---------------------------------------------------------------------------------------------------------------
    def getId(self):
        return self.id

    # Start ------------------------------------------------------------------------------------------------------------
    def getStart(self):
        return self.start

    # Final ------------------------------------------------------------------------------------------------------------
    def getFinal(self):
        return self.final

    # Reward -----------------------------------------------------------------------------------------------------------
    def getReward(self):
        return self.reward

    # Bot --------------------------------------------------------------------------------------------------------------
    def getOwner(self):
        return self.owner
