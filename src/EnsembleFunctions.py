import numpy as np

def concatenate_predict(bayes, lstm, bert, inputs: list):
  bayes_pred = bayes.predict(inputs)
  lstm_pred = lstm.predict(inputs)
  bert_pred = bert.predict(inputs)

  return np.array([
    bayes_pred,
    lstm_pred,
    bert_pred,
  ])

def concatenate_predict_proba(bayes, lstm, bert, inputs: list):
  bayes_pred = bayes.predict_proba(inputs)
  lstm_pred = lstm.predict_proba(inputs)
  bert_pred = bert.predict_proba(inputs)

  return np.array([
    bayes_pred,
    lstm_pred,
    bert_pred,
  ])

def hard_voting(bayes, lstm, bert, inputs: list):
  preds = concatenate_predict(bayes, lstm, bert, inputs)
  results = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(),
    axis=0,
    arr=preds,
  )
  return results

def soft_voting(bayes, lstm, bert, inputs: list):
  preds = concatenate_predict_proba(bayes, lstm, bert, inputs)
  results = np.average(
    preds, 
    axis=0,
  )
  return results
  
def stacking(
    bayes,
    lstm,
    bert,
    logistic_regression,
    inputs: list,
  ):
  preds = concatenate_predict_proba(bayes, lstm, bert, inputs)

  # Get only the predictions for hate speech
  # Since logistic regression is implemented
  # to assume that you give it the data for 1
  preds = preds[:, :, 1:]

  # Transpose so all learners' preds are on the same row
  transposed_preds = preds.T[0]

  return logistic_regression.predict_proba(transposed_preds)