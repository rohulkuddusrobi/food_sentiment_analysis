"""Database-driven analytics for the Business Intelligence dashboard.

Every KPI, chart dataset and insight shown on the home page is computed here
from ``Review`` records (rating-derived sentiment only).

``PredictionHistory`` records are intentionally never read here: ML predictions
are a separate workflow and must not be mixed with the imported dataset
statistics.
"""

from datetime import datetime, time, timedelta, timezone as dt_timezone

from django.conf import settings
from django.db.models import Avg, Count, Q
from django.db.models.functions import TruncMonth
from django.utils import timezone

from .models import Sentiment

POSITIVE = Sentiment.POSITIVE.value
NEUTRAL = Sentiment.NEUTRAL.value
NEGATIVE = Sentiment.NEGATIVE.value

SENTIMENT_LABELS = (POSITIVE, NEUTRAL, NEGATIVE)
SENTIMENT_COLORS = {POSITIVE: '#198754', NEUTRAL: '#6c757d', NEGATIVE: '#dc3545'}
RATING_VALUES = (1, 2, 3, 4, 5)
RATING_COLORS = ('#dc3545', '#fd7e14', '#ffc107', '#0d6efd', '#198754')

PREVIEW_LIMIT = 8
MAX_MONTHS = 24


def percentage(part, base):
    """Return a 0-100 percentage rounded to one decimal, or ``None``."""
    if not base:
        return None
    return round(100.0 * part / base, 1)


def format_percent(value):
    return f'{value:.1f}%' if value is not None else '\u2014'


def _day_start(date_value):
    moment = datetime.combine(date_value, time.min)
    if settings.USE_TZ:
        return timezone.make_aware(moment, dt_timezone.utc)
    return moment


def date_range_bounds(start_date, end_date):
    """Return ``(gte, lt)`` aware datetimes for an inclusive date range."""
    gte = _day_start(start_date) if start_date else None
    lt = _day_start(end_date + timedelta(days=1)) if end_date else None
    return gte, lt


def apply_date_filter(queryset, start_date, end_date):
    """Restrict reviews to the given inclusive date range.

    When a range is active, reviews without a review date are excluded because
    they cannot be placed inside the range.
    """
    if not start_date and not end_date:
        return queryset
    filtered = queryset.filter(review_date__isnull=False)
    gte, lt = date_range_bounds(start_date, end_date)
    if gte is not None:
        filtered = filtered.filter(review_date__gte=gte)
    if lt is not None:
        filtered = filtered.filter(review_date__lt=lt)
    return filtered


def compute_kpis(queryset):
    """Calculate the dashboard KPI values from a (already filtered) queryset."""
    total = queryset.count()

    sentiment_counts = {
        label: queryset.filter(sentiment=label).count() for label in SENTIMENT_LABELS
    }
    sentiment_base = sum(sentiment_counts.values())

    valid_ratings = queryset.filter(rating__gte=1, rating__lte=5)
    rated = valid_ratings.count()
    unrated = total - rated

    average = valid_ratings.aggregate(value=Avg('rating'))['value']
    average = round(average, 2) if average is not None else None

    return {
        'total': total,
        'positive': sentiment_counts[POSITIVE],
        'negative': sentiment_counts[NEGATIVE],
        'neutral': sentiment_counts[NEUTRAL],
        'sentiment_counts': sentiment_counts,
        'sentiment_base': sentiment_base,
        'positive_pct': percentage(sentiment_counts[POSITIVE], sentiment_base),
        'negative_pct': percentage(sentiment_counts[NEGATIVE], sentiment_base),
        'neutral_pct': percentage(sentiment_counts[NEUTRAL], sentiment_base),
        'positive_pct_display': format_percent(
            percentage(sentiment_counts[POSITIVE], sentiment_base)
        ),
        'negative_pct_display': format_percent(
            percentage(sentiment_counts[NEGATIVE], sentiment_base)
        ),
        'neutral_pct_display': format_percent(
            percentage(sentiment_counts[NEUTRAL], sentiment_base)
        ),
        'unrated': unrated,
        'unrated_pct': percentage(unrated, total),
        'unrated_pct_display': format_percent(percentage(unrated, total)),
        'rated': rated,
        'avg_rating': average,
        'avg_rating_display': f'{average:.2f}' if average is not None else '\u2014',
    }


def sentiment_chart(kpis):
    """Doughnut dataset for rating-derived sentiment."""
    data = [kpis['sentiment_counts'][label] for label in SENTIMENT_LABELS]
    return {
        'labels': list(SENTIMENT_LABELS),
        'data': data,
        'colors': [SENTIMENT_COLORS[label] for label in SENTIMENT_LABELS],
        'has_data': sum(data) > 0,
    }


def rating_chart(queryset):
    """Bar dataset for the 1-5 rating distribution."""
    rows = (
        queryset.filter(rating__in=RATING_VALUES)
        .values('rating')
        .annotate(count=Count('id'))
    )
    counts = {row['rating']: row['count'] for row in rows}
    data = [counts.get(rating, 0) for rating in RATING_VALUES]
    return {
        'labels': [str(rating) for rating in RATING_VALUES],
        'data': data,
        'colors': list(RATING_COLORS),
        'has_data': sum(data) > 0,
    }


def monthly_volume_chart(queryset):
    """Line dataset of review volume per month, using valid review dates."""
    rows = (
        queryset.filter(review_date__isnull=False)
        .annotate(month=TruncMonth('review_date'))
        .values('month')
        .annotate(count=Count('id'))
        .order_by('month')
    )
    labels = []
    data = []
    for row in rows:
        labels.append(row['month'].strftime('%Y-%m'))
        data.append(row['count'])
    if len(labels) > MAX_MONTHS:
        labels = labels[-MAX_MONTHS:]
        data = data[-MAX_MONTHS:]
    return {
        'labels': labels,
        'data': data,
        'has_data': bool(data),
    }


def sentiment_by_rating_chart(queryset):
    """Grouped bar dataset: sentiment counts inside each rating 1-5."""
    rows = (
        queryset.filter(rating__in=RATING_VALUES, sentiment__in=SENTIMENT_LABELS)
        .values('rating', 'sentiment')
        .annotate(count=Count('id'))
    )
    matrix = {label: [0] * len(RATING_VALUES) for label in SENTIMENT_LABELS}
    for row in rows:
        index = RATING_VALUES.index(row['rating'])
        matrix[row['sentiment']][index] = row['count']

    total = sum(sum(values) for values in matrix.values())
    return {
        'labels': [str(rating) for rating in RATING_VALUES],
        'datasets': [
            {
                'label': label,
                'data': matrix[label],
                'color': SENTIMENT_COLORS[label],
            }
            for label in SENTIMENT_LABELS
        ],
        'has_data': total > 0,
    }


def build_insights(kpis, monthly, is_filtered=False):
    """Plain-language statements derived only from the calculated statistics."""
    if kpis['total'] == 0:
        return [
            'No reviews are available for the selected filters. '
            'Upload a CSV file to populate the dashboard.'
        ]

    scope = ' in the selected date range' if is_filtered else ''
    insights = []
    base = kpis['sentiment_base']

    if base:
        counts = kpis['sentiment_counts']
        top_label, top_count = max(counts.items(), key=lambda item: item[1])
        if top_count:
            insights.append(
                f'{top_label} is the largest rating-derived sentiment group: '
                f'{top_count} of {base} rated reviews '
                f'({format_percent(percentage(top_count, base))}){scope}.'
            )
        negative = counts[NEGATIVE]
        if negative:
            insights.append(
                f'{format_percent(percentage(negative, base))} of rating-derived '
                f'feedback is Negative ({negative} of {base} reviews with a valid '
                f'rating){scope}.'
            )
        else:
            insights.append(
                f'No Negative rating-derived feedback among the {base} reviews '
                f'with a valid rating{scope}.'
            )
    else:
        insights.append(
            'No reviews with rating-derived sentiment are available, so '
            'sentiment insights cannot be calculated.'
        )

    if kpis['avg_rating'] is not None:
        insights.append(
            f"Average rating is {kpis['avg_rating_display']} out of 5 based on "
            f"{kpis['rated']} valid ratings{scope}."
        )
    else:
        insights.append(
            'No valid ratings (1 to 5) are available, so the average rating '
            'cannot be calculated.'
        )

    labels = monthly['labels']
    values = monthly['data']
    if len(labels) >= 2:
        prev_label, prev_value = labels[-2], values[-2]
        cur_label, cur_value = labels[-1], values[-1]
        delta = cur_value - prev_value
        if delta > 0:
            trend = (
                f'rose from {prev_value} in {prev_label} to {cur_value} in '
                f'{cur_label} (+{delta})'
            )
            if prev_value:
                trend += f' (+{percentage(delta, prev_value):.1f}%)'
        elif delta < 0:
            trend = (
                f'fell from {prev_value} in {prev_label} to {cur_value} in '
                f'{cur_label} ({delta})'
            )
        else:
            trend = (
                f'was unchanged at {cur_value} between {prev_label} and '
                f'{cur_label}'
            )
        insights.append(f'Monthly review volume {trend}{scope}.')
    elif len(labels) == 1:
        insights.append(
            'Only one month of dated reviews is available, so a monthly review '
            'volume change cannot be calculated.'
        )
    else:
        insights.append(
            'No reviews with a valid review date are available for monthly '
            'volume analysis.'
        )

    return insights


def latest_reviews(queryset, limit=PREVIEW_LIMIT):
    """Most recently imported reviews inside the current filter."""
    return list(queryset.order_by('-created_at', '-id')[:limit])


def build_dashboard(queryset, start_date=None, end_date=None, preview_limit=PREVIEW_LIMIT):
    """Assemble every dashboard payload from a single Review queryset."""
    filtered = apply_date_filter(queryset, start_date, end_date)
    is_filtered = bool(start_date or end_date)

    kpis = compute_kpis(filtered)
    charts = {
        'sentiment': sentiment_chart(kpis),
        'rating': rating_chart(filtered),
        'monthly': monthly_volume_chart(filtered),
        'sentiment_by_rating': sentiment_by_rating_chart(filtered),
    }

    return {
        'kpis': kpis,
        'charts': charts,
        'insights': build_insights(kpis, charts['monthly'], is_filtered=is_filtered),
        'latest_reviews': latest_reviews(filtered, limit=preview_limit),
        'is_filtered': is_filtered,
        'total_all': queryset.count(),
        'filtered_count': filtered.count(),
        'start_date': start_date,
        'end_date': end_date,
    }
