from state import state


class policy:
    def __init__(self, name: str):
        self.name = name

    def selectAction(self, valueFunction, state: state) -> int:
        """select_action die op basis van diens value function en een state een actie terug geeft.
         Voor een mvp kan je beginnen met een random policy. """
        pass

    def __str__(self):
        return "name: %s" % self.name
