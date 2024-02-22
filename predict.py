from src import Ensemble
import joblib

if __name__ == "__main__":
    ensemble = joblib.load('ensemble.pkl')
    print(ensemble.predict(["Gago ka putang ina", 'OIDAjodiajisjdai', "You are a fucking bitch", "NAKO  NAHIYA  YUNG  KAPAL  NG  PERA  NI  BINAY ", "fuck you binay gago ka"]))
    # print(Ensemble.HardVotingEnsemble)
    # print(Ensemble.SoftVotingEnsemble)
    # print(Ensemble.StackingEnsemble)

    exit(0)
