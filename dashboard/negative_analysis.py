"""Negative feedback analysis built from rating-derived sentiment.

Everything here is computed from imported ``Review`` records whose rating-derived
sentiment is Negative. ``PredictionHistory`` records are never used.

The term-frequency step is deliberately simple and transparent: text is
lower-cased, split on non-letters, short words and a fixed English stop-word
list are dropped, and the remaining words are counted over a bounded sample of
reviews. Frequency describes wording only - it is never presented as an
explanation of why customers were dissatisfied.
"""

import re
from collections import Counter

from django.db.models import Count
from django.db.models.functions import TruncMonth

from . import analytics, queries
from .models import Sentiment

NEGATIVE = Sentiment.NEGATIVE.value

MAX_TERMS = 10
TERM_SAMPLE_LIMIT = 500
REPRESENTATIVE_LIMIT = 12

WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")

TERMS_METHOD_NOTE = (
    'Text is lower-cased, split on non-letters, and filtered for words '
    'shorter than 3 characters plus a fixed English stop-word list.'
)
TERMS_CAVEAT = (
    'Term frequency describes wording only; it does not show what '
    'caused the dissatisfaction.'
)

STOP_WORDS = frozenset({
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'any', 'can',
    'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him',
    'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who',
    'did', 'its', 'let', 'she', 'too', 'use', 'that', 'with', 'have',
    'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been',
    'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just',
    'like', 'long', 'make', 'many', 'more', 'only', 'over', 'such',
    'take', 'than', 'them', 'well', 'were', 'what', 'about', 'after',
    'again', 'also', 'back', 'because', 'before', 'being', 'between',
    'both', 'does', 'during', 'each', 'even', 'first', 'going', 'good',
    'got', 'into', 'its', 'last', 'made', 'may', 'most', 'other', 'own',
    'people', 'said', 'same', 'should', 'some', 'still', 'such', 'take',
    'than', 'then', 'there', 'these', 'they', 'think', 'those', 'through',
    'too', 'under', 'until', 'use', 'very', 'want', 'well', 'went', 'were',
    'what', 'when', 'where', 'which', 'while', 'will', 'with', 'work',
    'would', 'year', 'your', 'really', 'quite', 'just', 'get', 'got',
    'thing', 'things', 'much', 'ever', 'never', 'always', 'only',
})


def frequent_terms(queryset, limit=MAX_TERMS, sample_limit=TERM_SAMPLE_LIMIT):
    """Count frequent words in review text over a bounded sample.

    Returns a dict with the top terms, the sampled document count and whether
    the sample had to be truncated because the queryset was larger.
    """
    total = queryset.count()
    sampled_texts = (
        queryset.order_by('-created_at', '-id')
        .values_list('text', flat=True)[:sample_limit]
    )

    term_counts = Counter()
    document_counts = Counter()
    sampled = 0
    token_total = 0

    for text in sampled_texts:
        sampled += 1
        words = []
        for raw in WORD_RE.findall(text or ''):
            word = raw.lower()
            if len(word) >= 3 and word not in STOP_WORDS:
                words.append(word)
        if not words:
            continue
        token_total += len(words)
        term_counts.update(words)
        document_counts.update(set(words))

    terms = [
        {
            'term': term,
            'count': count,
            'documents': document_counts[term],
            'documents_pct': analytics.percentage(document_counts[term], sampled),
        }
        for term, count in term_counts.most_common(limit)
    ]

    return {
        'terms': terms,
        'sampled': sampled,
        'total': total,
        'truncated': sampled < total,
        'tokens': token_total,
        'method_note': TERMS_METHOD_NOTE,
        'caveat': TERMS_CAVEAT,
    }


def _monthly_negative_volume(negative_queryset):
    rows = (
        negative_queryset.filter(review_date__isnull=False)
        .annotate(month=TruncMonth('review_date'))
        .values('month')
        .annotate(count=Count('id'))
        .order_by('month')
    )
    labels = [row['month'].strftime('%Y-%m') for row in rows]
    data = [row['count'] for row in rows]
    return {'labels': labels, 'data': data}


def _rating_breakdown(negative_queryset, negative_count):
    rows = (
        negative_queryset.filter(rating__in=analytics.RATING_VALUES)
        .values('rating')
        .annotate(count=Count('id'))
    )
    counts = {row['rating']: row['count'] for row in rows}
    breakdown = []
    for rating in analytics.RATING_VALUES:
        count = counts.get(rating, 0)
        breakdown.append({
            'rating': rating,
            'count': count,
            'share': analytics.percentage(count, negative_count),
            'share_display': analytics.format_percent(
                analytics.percentage(count, negative_count)
            ),
        })
    return breakdown


def build_negative_insights(stats):
    """Plain-language observations, each carrying its sample size/denominator."""
    if stats['scope_total'] == 0:
        return [
            'No reviews match the selected filters, so negative feedback '
            'analysis cannot be calculated.'
        ]
    if stats['base'] == 0:
        return [
            'No reviews with rating-derived sentiment match the selected '
            'filters, so the negative feedback share cannot be calculated.'
        ]

    insights = [
        f"{stats['negative_count']} of {stats['base']} reviews with "
        f"rating-derived sentiment are Negative "
        f"({stats['share_display']})."
    ]

    if not stats['negative_count']:
        insights.append('No rating-derived negative reviews matched the filters.')
        return insights

    present = [row for row in stats['rating_breakdown'] if row['count']]
    if present:
        top = max(present, key=lambda row: (row['count'], -row['rating']))
        insights.append(
            'Rating breakdown of the '
            f"{stats['negative_count']} negative reviews: "
            + ', '.join(f"{row['rating']}\u2605 {row['count']}" for row in present)
            + f" (n = {stats['negative_count']}); most common is "
            f"{top['rating']}\u2605 with {top['count']} "
            f"({top['share_display']})."
        )

    if stats['avg_rating'] is not None:
        insights.append(
            f"Average rating among negative reviews is "
            f"{stats['avg_rating_display']} out of 5 "
            f"(n = {stats['negative_count']})."
        )

    labels = stats['monthly']['labels']
    values = stats['monthly']['data']
    if len(labels) >= 2:
        prev_label, prev_value = labels[-2], values[-2]
        cur_label, cur_value = labels[-1], values[-1]
        delta = cur_value - prev_value
        if delta > 0:
            trend = (
                f'rose from {prev_value} in {prev_label} to {cur_value} in '
                f'{cur_label} (+{delta})'
            )
        elif delta < 0:
            trend = (
                f'fell from {prev_value} in {prev_label} to {cur_value} in '
                f'{cur_label} ({delta})'
            )
        else:
            trend = f'was unchanged at {cur_value} between {prev_label} and {cur_label}'
        insights.append(f'Monthly negative review volume {trend}.')
    elif len(labels) == 1:
        insights.append(
            'Only one month of dated negative reviews is available, so a '
            'monthly volume change cannot be calculated.'
        )
    else:
        insights.append(
            'No negative reviews with a valid review date are available for '
            'monthly volume analysis.'
        )

    terms = stats['terms']
    if terms['terms']:
        listed = ', '.join(
            f"{item['term']} ({item['documents']})"
            for item in terms['terms'][:5]
        )
        sample_note = (
            f" Sample: {terms['sampled']} of {terms['total']} negative reviews."
            if terms['truncated']
            else f" Sample: all {terms['sampled']} negative reviews."
        )
        insights.append(
            f'Most frequent terms in negative review text: {listed}.{sample_note}'
        )
        insights.append(terms['caveat'])
    else:
        insights.append(
            'Not enough negative review text is available to extract frequent terms.'
        )

    return insights


def build_negative_analysis(base_queryset=None, q='', start_date=None, end_date=None):
    """Assemble every value used by the Negative Feedback page."""
    scope_queryset = queries.build_review_queryset(
        base=base_queryset,
        q=q,
        start_date=start_date,
        end_date=end_date,
    )
    negative_queryset = scope_queryset.filter(sentiment=NEGATIVE)

    scope_total = scope_queryset.count()
    negative_count = negative_queryset.count()
    base = scope_queryset.filter(sentiment__in=analytics.SENTIMENT_LABELS).count()

    average = (
        negative_queryset.filter(rating__gte=1, rating__lte=5)
        .aggregate(value=analytics.Avg('rating'))['value']
    )
    average = round(average, 2) if average is not None else None

    rating_breakdown = _rating_breakdown(negative_queryset, negative_count)
    present_ratings = [row for row in rating_breakdown if row['count']]
    top_rating = (
        max(present_ratings, key=lambda row: (row['count'], -row['rating']))
        if present_ratings
        else None
    )

    terms = (
        frequent_terms(negative_queryset)
        if negative_count
        else {
            'terms': [],
            'sampled': 0,
            'total': 0,
            'truncated': False,
            'tokens': 0,
            'method_note': TERMS_METHOD_NOTE,
            'caveat': TERMS_CAVEAT,
        }
    )

    representative = list(
        negative_queryset.order_by('-created_at', '-id')[:REPRESENTATIVE_LIMIT]
    )

    stats = {
        'scope_total': scope_total,
        'base': base,
        'negative_count': negative_count,
        'share': analytics.percentage(negative_count, base),
        'share_display': analytics.format_percent(
            analytics.percentage(negative_count, base)
        ),
        'rating_breakdown': rating_breakdown,
        'top_rating': top_rating,
        'avg_rating': average,
        'avg_rating_display': f'{average:.2f}' if average is not None else '\u2014',
        'monthly': _monthly_negative_volume(negative_queryset),
        'terms': terms,
        'is_filtered': bool(q or start_date or end_date),
    }
    stats['insights'] = build_negative_insights(stats)

    return {
        **stats,
        'representative': representative,
        'representative_limit': REPRESENTATIVE_LIMIT,
        'negative_share_of_scope': analytics.percentage(negative_count, scope_total),
        'negative_share_of_scope_display': analytics.format_percent(
            analytics.percentage(negative_count, scope_total)
        ),
    }
