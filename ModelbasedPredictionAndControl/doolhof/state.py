from dataclasses import dataclass


@dataclass
class state:
    def __init__(self, pos: tuple, reward: int, done: bool, value: float = 0):
        self.position = pos
        self.reward = reward
        self.done = done
        self.value = value

    def __str__(self):
        return "position: %s, reward: %s,  done: %s,  value: %s" % (self.position, self.reward, self.done, self.value)
