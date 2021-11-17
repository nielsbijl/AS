class state:
    def __init__(self, pos: tuple, reward: int, done: bool):
        self.position = pos
        self.reward = reward
        self.done = done

    def __str__(self):
        return "position: %s, reward: %s,  done: %s" % (self.position, self.reward, self.done)
