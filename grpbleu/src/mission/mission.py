class Mission:
    def __init__(self, number_mission, mission):
        self.id = number_mission
        self.start, self.final, self.reward, self.owner = mission.list()

    def getId(self):
        return self.id

    def getStart(self):
        return self.start

    def getFinal(self):
        return self.final

    def getReward(self):
        return self.reward

    def getOwner(self):
        return self.owner
