from src import Ensemble
import joblib

if __name__ == "__main__":
    ensemble = joblib.load('ensemble.pkl')
    quotes = [
        "Gago ka putang ina", 
        'OIDAjodiajisjdai', 
        "You are a fucking bitch", 
        "NAKO  NAHIYA  YUNG  KAPAL  NG  PERA  NI  BINAY ", 
        "fuck you binay gago ka",
    ]
    print(ensemble.estimators_)
    for estimator in ensemble.estimators_:
        print(estimator.predict_proba(quotes))
    print(ensemble.predict_proba(quotes))
    # print(ensemble.predict_proba(["Gago ka putang ina", 'OIDAjodiajisjdai', "You are a fucking bitch", "NAKO  NAHIYA  YUNG  KAPAL  NG  PERA  NI  BINAY ", "fuck you binay gago ka"]))

    # print(Ensemble.HardVotingEnsemble)
    # print(Ensemble.SoftVotingEnsemble)
    # print(Ensemble.StackingEnsemble)

    exit(0)
