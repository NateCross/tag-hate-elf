# Naive Bayes

## Fit Code

```python
@_fit_context(prefer_skip_nested_validation=True)
def fit(self, X, y, sample_weight=None):
    """Fit Naive Bayes classifier according to X, y.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,)
        Target values.

    sample_weight : array-like of shape (n_samples,), default=None
        Weights applied to individual samples (1. for unweighted).

    Returns
    -------
    self : object
        Returns the instance itself.
    """
    X, y = self._check_X_y(X, y)
    _, n_features = X.shape

    labelbin = LabelBinarizer()
    Y = labelbin.fit_transform(y)
    self.classes_ = labelbin.classes_
    if Y.shape[1] == 1:
        if len(self.classes_) == 2:
            Y = np.concatenate((1 - Y, Y), axis=1)
        else:  # degenerate case: just one class
            Y = np.ones_like(Y)

    # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
    # We convert it to np.float64 to support sample_weight consistently;
    # this means we also don't have to cast X to floating point
    if sample_weight is not None:
        Y = Y.astype(np.float64, copy=False)
        sample_weight = _check_sample_weight(sample_weight, X)
        sample_weight = np.atleast_2d(sample_weight)
        Y *= sample_weight.T

    class_prior = self.class_prior

    # Count raw events from data before updating the class log prior
    # and feature log probas
    n_classes = Y.shape[1]
    self._init_counters(n_classes, n_features)
    self._count(X, Y)
    alpha = self._check_alpha()
    self._update_feature_log_prob(alpha)
    self._update_class_log_prior(class_prior=class_prior)
    return self
```