"""Lazy loading and prediction for the pre-trained sentiment ensemble.

The saved artifact (``ml_models/sentiment_ensemble_model.joblib``) holds a
dictionary with a TF-IDF vectorizer, three classifiers and the class labels.
It is loaded once, cached, and never retrained or rewritten by the app.
"""

import logging
import re
import warnings
from collections import Counter
from pathlib import Path

import joblib
from django.conf import settings

logger = logging.getLogger(__name__)

CLASSIFIERS = ('logistic_regression', 'linear_svm', 'sgd_classifier')
REQUIRED_KEYS = ('tfidf_vectorizer', 'classes') + CLASSIFIERS
VALID_LABELS = ('Negative', 'Neutral', 'Positive')

DEFAULT_MODEL_LOCATION = Path('ml_models') / 'sentiment_ensemble_model.joblib'

_VERSION_PATTERN = re.compile(r'from version (\S+) when using version (\S+)')


class MLServiceError(Exception):
    """Raised when the model cannot be loaded or a prediction cannot be made."""


class _ModelCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.path = None
        self.model = None
        self.error = None
        self.warning = None
        self.attempted = False


_cache = _ModelCache()


def model_path():
    """Configured model path, based on the project root."""
    configured = getattr(settings, 'SENTIMENT_MODEL_PATH', None)
    return Path(configured) if configured else Path(settings.BASE_DIR) / DEFAULT_MODEL_LOCATION


def reset_cache():
    """Forget the loaded model (used by tests and after settings change)."""
    _cache.reset()


def _version_warning(caught):
    for item in caught:
        message = str(item.message)
        if 'InconsistentVersionWarning' not in item.category.__name__:
            continue
        match = _VERSION_PATTERN.search(message)
        if match:
            trained, installed = (group.rstrip('.') for group in match.groups())
            return (
                f'Model was trained with scikit-learn {trained} but '
                f'scikit-learn {installed} is installed. Predictions may be '
                'unreliable; the saved model was not retrained or replaced.'
            )
        return (
            f'Model was saved with a different scikit-learn version ({message}). '
            'The saved model was not retrained or replaced.'
        )
    return None


def _incompatible(path, reason):
    _cache.error = (
        f'Sentiment model at {path} is incompatible: {reason}. '
        'The saved model was not retrained or replaced.'
    )
    raise MLServiceError(_cache.error)


def load_model(force=False):
    """Return the cached model payload, loading it from disk only once."""
    path = model_path()

    if not force and _cache.attempted and _cache.path == path:
        if _cache.model is None:
            raise MLServiceError(_cache.error)
        return _cache.model

    _cache.reset()
    _cache.path = path
    _cache.attempted = True

    if not path.exists():
        _cache.error = (
            f'Sentiment model file not found at {path}. '
            'Check ML_MODELS_DIR/SENTIMENT_MODEL_PATH in settings.'
        )
        raise MLServiceError(_cache.error)
    if not path.is_file():
        _cache.error = f'Sentiment model path {path} is not a file.'
        raise MLServiceError(_cache.error)

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            payload = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not fatal
        _cache.error = (
            f'Sentiment model at {path} could not be loaded '
            f'({exc.__class__.__name__}: {exc}). The saved model was not modified.'
        )
        logger.exception('Failed to load sentiment model from %s', path)
        raise MLServiceError(_cache.error) from exc

    _cache.warning = _version_warning(caught)

    if not isinstance(payload, dict):
        _incompatible(path, 'expected a dictionary of trained estimators')

    missing = [key for key in REQUIRED_KEYS if key not in payload]
    if missing:
        _incompatible(path, f"missing component(s): {', '.join(missing)}")

    classes = [str(label) for label in payload['classes']]
    if sorted(classes) != sorted(VALID_LABELS):
        _incompatible(
            path,
            f"expected classes {list(VALID_LABELS)}, found {classes}",
        )

    if not callable(getattr(payload['tfidf_vectorizer'], 'transform', None)):
        _incompatible(path, "'tfidf_vectorizer' cannot transform text")
    for name in CLASSIFIERS:
        if not callable(getattr(payload[name], 'predict', None)):
            _incompatible(path, f"'{name}' cannot predict")

    _cache.model = payload
    if _cache.warning:
        logger.warning('Sentiment model compatibility: %s', _cache.warning)
    return payload


def get_status():
    """Diagnostics for the UI: path, availability, and any compatibility note."""
    path = model_path()
    try:
        load_model()
    except MLServiceError as exc:
        return {'path': str(path), 'loaded': False, 'error': str(exc), 'warning': None}
    return {'path': str(path), 'loaded': True, 'error': None, 'warning': _cache.warning}


def resolve_vote(votes, classes):
    """Hard voting: most votes wins, ties follow the saved class order."""
    unknown = [label for label in votes if label not in classes]
    if unknown:
        raise MLServiceError(f'Model returned an unexpected class label: {unknown[0]!r}.')

    counts = Counter(votes)
    best_label = None
    best_count = -1
    for label in classes:
        count = counts.get(label, 0)
        if count > best_count:
            best_label, best_count = label, count

    if best_label is None or best_count <= 0:
        raise MLServiceError('Model returned no usable predictions.')
    if best_label not in VALID_LABELS:
        raise MLServiceError(f'Model returned an unexpected class label: {best_label!r}.')
    return best_label


def predict(text):
    """Predict one sentiment label for a single review text."""
    cleaned = (text or '').strip()
    if not cleaned:
        raise MLServiceError('Enter some review text to predict a sentiment.')

    model = load_model()
    try:
        features = model['tfidf_vectorizer'].transform([cleaned])
        votes = [str(model[name].predict(features)[0]) for name in CLASSIFIERS]
    except MLServiceError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not fatal
        logger.exception('Prediction failed')
        raise MLServiceError(
            f'Prediction failed ({exc.__class__.__name__}: {exc}).'
        ) from exc

    return resolve_vote(votes, [str(label) for label in model['classes']])
