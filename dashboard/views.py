from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import StreamingHttpResponse
from django.shortcuts import render

from . import analytics, csv_export, ml_service, negative_analysis, queries
from .csv_import import CSVImportError, import_reviews
from .forms import (
    DashboardFilterForm,
    NegativeFilterForm,
    PredictionForm,
    ReviewCSVUploadForm,
    ReviewFilterForm,
)
from .models import PredictionHistory, Review

EXPLORER_FILTER_KEYS = ('q', 'rating', 'sentiment', 'start_date', 'end_date')
NEGATIVE_FILTER_KEYS = ('q', 'start_date', 'end_date')


def _active_filters(params, keys, valid):
    """True when the user submitted at least one usable filter value."""
    if not valid:
        return False
    return any((params.get(key) or '').strip() for key in keys)


def _public_model_status():
    """Model diagnostics with the server path reduced to its file name."""
    status = dict(ml_service.get_status())
    path = status.get('path') or ''
    name = Path(path).name
    status['path'] = name or path
    for key in ('error', 'warning'):
        if status.get(key) and path:
            status[key] = status[key].replace(path, status['path'])
    return status


def _redact_model_path(message):
    """Reduce the configured model path to its file name in user-facing text."""
    path = str(ml_service.model_path() or '')
    if not path:
        return str(message)
    return str(message).replace(path, Path(path).name)


def home(request):
    """Data-driven BI dashboard built from imported Review records."""
    filter_form = DashboardFilterForm(request.GET)
    start_date = end_date = None
    if filter_form.is_valid():
        start_date = filter_form.cleaned_data.get('start_date')
        end_date = filter_form.cleaned_data.get('end_date')

    dashboard = analytics.build_dashboard(
        Review.objects.all(),
        start_date=start_date,
        end_date=end_date,
        preview_limit=analytics.PREVIEW_LIMIT,
    )

    active_filter_params = {}
    if start_date:
        active_filter_params['start_date'] = start_date.isoformat()
    if end_date:
        active_filter_params['end_date'] = end_date.isoformat()

    context = {
        **dashboard,
        'filter_form': filter_form,
        'filter_active': dashboard['is_filtered'],
        'preview_limit': analytics.PREVIEW_LIMIT,
        'filter_querystring': urlencode(active_filter_params),
    }
    return render(request, 'dashboard/home.html', context)


def upload_reviews(request):
    summary = None
    if request.method == 'POST':
        form = ReviewCSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                summary = import_reviews(form.cleaned_data['file'])
            except CSVImportError as exc:
                summary = None
                messages.error(request, str(exc))
            else:
                if summary['imported']:
                    messages.success(
                        request,
                        f"Imported {summary['imported']} review(s) from the file.",
                    )
                if summary['duplicates']:
                    messages.warning(
                        request,
                        f"Skipped {summary['duplicates']} duplicate review(s).",
                    )
                if summary['invalid']:
                    messages.warning(
                        request,
                        f"Skipped {summary['invalid']} invalid row(s).",
                    )
                if summary['processed'] and not summary['imported'] and not summary['duplicates'] and not summary['invalid']:
                    messages.info(request, 'No rows were imported from this file.')
                if summary.get('notice'):
                    messages.info(request, summary['notice'])
    else:
        form = ReviewCSVUploadForm()

    return render(
        request,
        'dashboard/upload.html',
        {'form': form, 'summary': summary},
    )


def predict_review(request):
    form = PredictionForm(request.POST or None)
    result = None

    if request.method == 'POST' and form.is_valid():
        text = form.cleaned_data['text']
        try:
            label = ml_service.predict(text)
        except ml_service.MLServiceError as exc:
            messages.error(request, _redact_model_path(str(exc)))
        else:
            PredictionHistory.objects.create(text=text, predicted_sentiment=label)
            result = {'text': text, 'sentiment': label}

    limit = getattr(settings, 'PREDICTION_HISTORY_LIMIT', 10)
    history = list(PredictionHistory.objects.all()[:limit])
    model_status = _public_model_status()

    return render(
        request,
        'dashboard/predict.html',
        {
            'form': form,
            'result': result,
            'history': history,
            'history_limit': limit,
            'model_status': model_status,
        },
    )


def _filtered_reviews(params):
    """Build the explorer queryset for the given filter parameters.

    Returns ``(queryset, filter_form, filters_valid)`` so the page and the CSV
    export always apply exactly the same filter rules.
    """
    filter_form = ReviewFilterForm(params)
    filters_valid = filter_form.is_valid()
    if filters_valid:
        data = filter_form.cleaned_data
        queryset = queries.build_review_queryset(
            q=data.get('q', ''),
            rating=data.get('rating', ''),
            sentiment=data.get('sentiment', ''),
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
        )
    else:
        queryset = Review.objects.all()
    return queryset.order_by('-created_at', '-id'), filter_form, filters_valid


def review_explorer(request):
    """Paginated, searchable list of imported reviews."""
    queryset, filter_form, filters_valid = _filtered_reviews(request.GET)

    page_size = getattr(settings, 'REVIEW_PAGE_SIZE', 20)
    paginator = Paginator(queryset, page_size)
    page_obj = paginator.get_page(request.GET.get('page'))

    return render(
        request,
        'dashboard/reviews.html',
        {
            'filter_form': filter_form,
            'filters_valid': filters_valid,
            'filters_active': _active_filters(
                request.GET, EXPLORER_FILTER_KEYS, filters_valid
            ),
            'page_obj': page_obj,
            'reviews': page_obj.object_list,
            'page_range': paginator.get_elided_page_range(page_obj.number),
            'match_count': paginator.count,
            'page_size': page_size,
            'page_querystring': queries.querystring_without(
                request.GET, exclude=('page',)
            ),
            'start_index': page_obj.start_index(),
            'end_index': page_obj.end_index(),
        },
    )


def negative_feedback(request):
    """Rating-derived negative review analysis with search and date filters."""
    filter_form = NegativeFilterForm(request.GET)
    filters_valid = filter_form.is_valid()

    if filters_valid:
        data = filter_form.cleaned_data
        analysis = negative_analysis.build_negative_analysis(
            q=data.get('q', ''),
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
        )
    else:
        analysis = negative_analysis.build_negative_analysis()

    analysis.update(
        {
            'filter_form': filter_form,
            'filters_valid': filters_valid,
            'filters_active': _active_filters(
                request.GET, NEGATIVE_FILTER_KEYS, filters_valid
            ),
        }
    )
    return render(request, 'dashboard/negative.html', analysis)


def export_reviews(request):
    """Stream the reviews matching the current explorer filters as CSV."""
    queryset, _filter_form, _valid = _filtered_reviews(request.GET)

    response = StreamingHttpResponse(
        csv_export.iter_csv(csv_export.REVIEW_EXPORT_HEADER, csv_export.review_rows(queryset)),
        content_type='text/csv; charset=utf-8',
    )
    response['Content-Disposition'] = (
        f'attachment; filename="{csv_export.export_filename("reviews")}"'
    )
    response['X-Content-Type-Options'] = 'nosniff'
    return response


def export_analytics(request):
    """Stream the dashboard KPI summary for the current date filter as CSV."""
    filter_form = DashboardFilterForm(request.GET)
    start_date = end_date = None
    if filter_form.is_valid():
        start_date = filter_form.cleaned_data.get('start_date')
        end_date = filter_form.cleaned_data.get('end_date')

    filtered = analytics.apply_date_filter(Review.objects.all(), start_date, end_date)
    kpis = analytics.compute_kpis(filtered)

    rows = csv_export.build_analytics_rows(
        kpis,
        start_date=start_date,
        end_date=end_date,
        scope_total=filtered.count(),
        database_total=Review.objects.count(),
    )

    response = StreamingHttpResponse(
        csv_export.iter_csv(csv_export.ANALYTICS_EXPORT_HEADER, iter(rows)),
        content_type='text/csv; charset=utf-8',
    )
    response['Content-Disposition'] = (
        f'attachment; filename="{csv_export.export_filename("analytics-summary")}"'
    )
    response['X-Content-Type-Options'] = 'nosniff'
    return response
