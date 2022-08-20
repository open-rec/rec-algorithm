from algorithm.structure.json_obj import JsonObj


class ScoreItem(JsonObj):

    def __init__(self, item="", score=0.0):
        self.item = item
        self.score = score
