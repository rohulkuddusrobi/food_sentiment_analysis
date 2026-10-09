"""Reusable ORM filter helpers shared by the explorer and analysis pages.

All helpers return the queryset unchanged when their filter is empty, so a
page can always be rendered with or without filters using the same code path.
"""

from urllib.parse import urlencode

from django.db.models import Q

from . import analytics
from .analytics import apply_date_filter
from .models import Review

#: Filter value that selects records with a missing/unavailable value.
MISSING_VALUE = 'none'


#: Canonical sentiment label for each lower-case variant (case-tolerant input).
SENTIMENT_LOOKUP = {label.lower(): label for label in analytics.SENTIMENT_LABELS}


def apply_search(queryset, term):
    """Match a keyword against review text and summary (case-insensitive)."""
    term = (term or '').strip()
    if not term:
        return queryset
    return queryset.filter(Q(text__icontains=term) | Q(summary__icontains=term))


def apply_rating_filter(queryset, value):
    """Filter by a single rating (1-5) or by a missing rating."""
    value = (value or '').strip()
    if not value:
        return queryset
    if value == MISSING_VALUE:
        return queryset.filter(Q(rating__isnull=True) | ~Q(rating__gte=1, rating__lte=5))
    if value.isdigit() and 1 <= int(value) <= 5:
        return queryset.filter(rating=int(value))
    return queryset


def apply_sentiment_filter(queryset, value):
    """Filter by a rating-derived sentiment label or by an unavailable one.

    Unknown labels are ignored so a bad value never narrows the queryset.
    """
    value = (value or '').strip()
    if not value:
        return queryset
    if value == MISSING_VALUE:
        return queryset.filter(Q(sentiment__isnull=True) | Q(sentiment=''))
    label = SENTIMENT_LOOKUP.get(value.lower())
    if label is None:
        return queryset
    return queryset.filter(sentiment__iexact=label)


def build_review_queryset(
    base=None,
    q='',
    rating='',
    sentiment='',
    start_date=None,
    end_date=None,
):
    """Apply search, rating, sentiment and date filters to a Review queryset."""
    queryset = base if base is not None else Review.objects.all()
    queryset = apply_date_filter(queryset, start_date, end_date)
    queryset = apply_search(queryset, q)
    queryset = apply_rating_filter(queryset, rating)
    queryset = apply_sentiment_filter(queryset, sentiment)
    return queryset


def querystring_without(params, exclude=()):
    """Rebuild a query string, dropping the named keys (e.g. ``page``)."""
    pairs = []
    for key, values in params.lists():
        if key in exclude:
            continue
        for value in values:
            pairs.append((key, value))
    return urlencode(pairs)
