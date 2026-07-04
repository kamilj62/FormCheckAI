from app.plugins.lstm_plugin import LSTMPlugin
from app.plugins.oly_plugin import OLYPlugin
from app.plugins.rule_plugin import RulePlugin


def build_runtime(models, feature_contracts, feature_builders):

    return {
        "lstm": LSTMPlugin(
            models["lstm"],
            models["class_names"],
            feature_contracts
        ),

        "oly": OLYPlugin(
            models["oly"],
            models["oly_labels"],
            feature_contracts,
            feature_builders["oly"]
        ),

        "rules": RulePlugin()
    }