"""
Future adapter. Not enabled in v2 UI yet.

The euroleague-api package supports competition_code:
  E = EuroLeague
  U = EuroCup

We keep this module behind the common provider interface so the model layer
does not need to change when EuroLeague/EuroCup are enabled.
"""

class EuroLeagueProvider:
    def __init__(self, competition_code="E"):
        self.competition_code=competition_code

    def status(self):
        return {
            "enabled": False,
            "competition_code": self.competition_code,
            "message": "Adapter scaffold ready; normalization will be enabled in a later release."
        }
