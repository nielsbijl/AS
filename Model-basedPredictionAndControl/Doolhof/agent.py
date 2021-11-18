from policy import policy
from doolhof import doolhof


class agent:
    def __init__(self, doolhof: doolhof, policy: policy):
        self.doolhof = doolhof
        self.policy = policy

    def valueFunction(self):
        """Een valuefunction dit is een mapping van states naar values.
         Hiervoor kan je dezelfde datastructuur aanhouden als bij de omgeving (e.g. een lijst)."""
        pass

    def choseAction(self, state, policy):
        """Een functie die een actie kiest op basis van een policy en een state"""
        pass

    def valueIteration(self):
        """Een implementatie van value iteration"""
        pass

    def __str__(self):
        return "policy: %s" % self.policy
