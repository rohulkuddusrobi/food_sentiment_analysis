"""Memory-conscious CSV export helpers.

Two exports are built here:

* Review rows for the Review Explorer, streamed straight from the ORM.
* A KPI summary for the dashboard, with explicit date filters and the
  denominator used by every percentage.

Only ``Review`` records are ever read; ``PredictionHistory`` (ML predictions)
is deliberately excluded so exported data stays comparable with the analytics.

Every exported cell passes through :func:`sanitize_cell`, which neutralises
spreadsheet formula injection for values beginning with ``=``, ``+``, ``-``,
``@`` (including values preceded by whitespace or control characters).
"""

import csv

from django.conf import settings
from django.utils import timezone

#: Characters that make spreadsheet applications treat a cell as a formula.
FORMULA_TRIGGERS = ('=', '+', '-', '@')

#: Control characters that can also start a formula once pasted into a sheet.
CONTROL_TRIGGERS = ('\t', '\r')

#: Rows fetched per database round-trip while streaming an export.
EXPORT_CHUNK_SIZE = 500

REVIEW_EXPORT_HEADER = (
    'review_text',
    'summary',
    'product_id',
    'rating',
    'sentiment',
    'review_date',
    'imported_at',
)

ANALYTICS_EXPORT_HEADER = (
    'section',
    'metric',
    'value',
    'definition',
)


def sanitize_cell(value):
    """Return a cell value that is safe to open in a spreadsheet.

    Strings that look like a formula (optionally after leading whitespace) are
    prefixed with a single quote so the formula is stored as plain text.
    Numbers and other non-string values are returned unchanged.
    """
    if value is None:
        return ''
    if not isinstance(value, str):
        return value
    if value[:1] in CONTROL_TRIGGERS:
        return "'" + value
    if value.lstrip()[:1] in FORMULA_TRIGGERS:
        return "'" + value
    return value


class _Echo:
    """Minimal file-like object that returns what it was given (streaming)."""

    def write(self, value):
        return value


def iter_csv(header, rows):
    """Yield CSV lines: the sanitised header first, then every sanitised row."""
    writer = csv.writer(_Echo())
    yield writer.writerow([sanitize_cell(cell) for cell in header])
    for row in rows:
        yield writer.writerow([sanitize_cell(cell) for cell in row])


def format_datetime(value):
    """Format an optional datetime as ``YYYY-MM-DD HH:MM:SS`` (local time)."""
    if value is None:
        return ''
    if settings.USE_TZ and timezone.is_aware(value):
        value = timezone.localtime(value)
    return value.strftime('%Y-%m-%d %H:%M:%S')


def export_filename(prefix):
    """Download filename such as ``reviews-20261009-164700.csv``."""
    stamp = timezone.localtime().strftime('%Y%m%d-%H%M%S')
    return f'{prefix}-{stamp}.csv'


def review_rows(queryset):
    """Yield one export row per review, streaming the queryset in chunks."""
    fields = (
        'text',
        'summary',
        'rating',
        'product_id',
        'review_date',
        'sentiment',
        'created_at',
    )
    for review in queryset.only(*fields).iterator(chunk_size=EXPORT_CHUNK_SIZE):
        yield [
            review.text,
            review.summary or '',
            review.product_id or '',
            review.rating if review.rating is not None else '',
            review.sentiment or '',
            format_datetime(review.review_date),
            format_datetime(review.created_at),
        ]


def _percent(value):
    return '' if value is None else f'{value:.1f}'


def _average(value):
    return '' if value is None else f'{value:.2f}'


def build_analytics_rows(
    kpis,
    start_date=None,
    end_date=None,
    scope_total=0,
    database_total=0,
):
    """Build the analytics summary rows (header is added by :func:`iter_csv`).

    ``kpis`` comes from :func:`dashboard.analytics.compute_kpis` and must be
    calculated on the same date-filtered queryset the dashboard shows.
    """
    generated = format_datetime(timezone.now())
    date_active = 'yes' if (start_date or end_date) else 'no'
    sentiment_base = kpis['sentiment_base']

    return [
        ['scope', 'Generated (UTC)', generated, 'Export timestamp'],
        [
            'scope',
            'Start date',
            start_date.isoformat() if start_date else '',
            'Inclusive lower bound on review date; empty means no lower bound',
        ],
        [
            'scope',
            'End date',
            end_date.isoformat() if end_date else '',
            'Inclusive upper bound on review date; empty means no upper bound',
        ],
        ['scope', 'Date filter active', date_active, 'Whether a date filter narrowed the dashboard'],
        ['scope', 'Reviews in scope', scope_total, 'Imported reviews after the date filter'],
        ['scope', 'Reviews in database', database_total, 'All imported reviews, unfiltered'],
        ['kpi', 'Total reviews', kpis['total'], 'Count of reviews in scope'],
        [
            'kpi',
            'Positive reviews',
            kpis['positive'],
            'Rating-derived sentiment Positive (rating 4 or 5)',
        ],
        [
            'kpi',
            'Neutral reviews',
            kpis['neutral'],
            'Rating-derived sentiment Neutral (rating 3)',
        ],
        [
            'kpi',
            'Negative reviews',
            kpis['negative'],
            'Rating-derived sentiment Negative (rating 1 or 2)',
        ],
        [
            'kpi',
            'Reviews with rating-derived sentiment',
            sentiment_base,
            'Denominator for the sentiment share percentages (valid ratings 1-5)',
        ],
        [
            'kpi',
            'Positive share (%)',
            _percent(kpis['positive_pct']),
            'Positive reviews / reviews with rating-derived sentiment x 100',
        ],
        [
            'kpi',
            'Neutral share (%)',
            _percent(kpis['neutral_pct']),
            'Neutral reviews / reviews with rating-derived sentiment x 100',
        ],
        [
            'kpi',
            'Negative share (%)',
            _percent(kpis['negative_pct']),
            'Negative reviews / reviews with rating-derived sentiment x 100',
        ],
        [
            'kpi',
            'Reviews with a valid rating',
            kpis['rated'],
            'Reviews whose rating is between 1 and 5',
        ],
        [
            'kpi',
            'Missing rating reviews',
            kpis['unrated'],
            'Reviews without a valid 1-5 rating',
        ],
        [
            'kpi',
            'Missing rating share (%)',
            _percent(kpis['unrated_pct']),
            'Missing rating reviews / total reviews in scope x 100',
        ],
        [
            'kpi',
            'Average rating',
            _average(kpis['avg_rating']),
            'Mean of the valid ratings (1-5); empty when no valid rating exists',
        ],
        [
            'note',
            'Export scope',
            '',
            'KPI summary only: no chart series, no ML predictions '
            '(PredictionHistory is excluded)',
        ],
    ]
