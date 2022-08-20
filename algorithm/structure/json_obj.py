import json


class JsonObj:

    def __init__(self, item="", score=0.0):
        self._item = item
        self._score = score

    def __str__(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
