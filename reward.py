import numpy as np

class Rewarder():
    def __init__(self, gamemode):
        self.gamemode = gamemode

    def rewardCardPlayed(self, agent, action, status, cardPlayed, mem):
        if status == -1:
            # out of index played
            agent.update(action, -100)
        elif status == 0:
            # wrong card played, lost a bomb tick
            agent.update(action, -1)
        elif status == 1:
            # correct card played
            # print("CORRECT CARD", cardPlayed, "PLAYED OFF MEM", mem)
            # TODO CALC CARD PLAYED REWARD
            if mem != 'xx':
                # played off of knowledge
                agent.update(action, 1)
            else:
                # played randomly
                agent.update(action, -100)

    def rewardDiscard(self, agent, action, agentActionCompleted, dmem, dcard, clocksGained):
        if not agentActionCompleted:
            # out of index discarded
            agent.update(action, -100)
        elif not clocksGained:
            # discarded with max clocks
            pass
        elif clocksGained:
            # TODO CALCULATE DISCARDED CARD REWARD
            if dmem != 'xx':
                # discarded off of knowledge
                agent.update(action, 0.05)
            else:
                # discarded randomly
                agent.update(action, -0.5)

    def rewardInfo(self, piagent, tpiagent, action, infoValue):
        piagent.update(action, infoValue)
