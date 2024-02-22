from src import Ensemble
import joblib

if __name__ == "__main__":
    ensemble = joblib.load('ensemble.pkl')
    print(ensemble.predict_proba(["Gago ka putang ina", 'OIDAjodiajisjdai', "You are a fucking bitch"]))
    # print(Ensemble.HardVotingEnsemble)
    # print(Ensemble.SoftVotingEnsemble)
    # print(Ensemble.StackingEnsemble)

    exit(0)
