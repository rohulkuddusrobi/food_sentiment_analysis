"""Streaming, batched import of review CSV files.

The upload is never written to disk and never fully read into memory: rows are
pulled one at a time from a ``csv.reader`` and written to the database in
fixed-size batches.
"""

import codecs
import csv
import re
from datetime import datetime, timezone as dt_timezone

from django.conf import settings
from django.db import IntegrityError, transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from .models import Review, make_content_hash, sentiment_from_rating

BATCH_SIZE = 500
MAX_ERROR_DETAILS = 25
READ_CHUNK = 64 * 1024
REQUIRED_COLUMN = 'Text'

TEXT_HEADER = 'text'
SCORE_HEADER = 'score'
PRODUCT_HEADER = 'productid'
TIME_HEADER = 'time'
SUMMARY_HEADER = 'summary'

#: Accepted header names per logical column, in order of preference. The first
#: header present in the file wins, so ``rating_number`` beats a text-style
#: ``Rating`` column and the readable ``Review Text`` beats a pre-cleaned one.
COLUMN_ALIASES = {
    TEXT_HEADER: ('text', 'review text', 'cleaned review'),
    SCORE_HEADER: ('score', 'rating_number', 'rating'),
    PRODUCT_HEADER: ('productid', 'product id', 'product_id'),
    TIME_HEADER: ('time', 'review date', 'review_date'),
    SUMMARY_HEADER: ('summary', 'review title', 'title'),
}

csv.field_size_limit(10 * 1024 * 1024)

#: "Rated 4 out of 5 stars" / "4 of 5 stars" style ratings used by review sites.
RATING_TEXT_PATTERN = re.compile(r'rated\s+([1-5])\s+(?:out\s+)?of\s+5\s+stars?', re.IGNORECASE)


class CSVImportError(Exception):
    """Raised when the whole file cannot be imported at all."""


def review_content_hash(text, product_id):
    return make_content_hash(text, product_id)


def build_header_map(fieldnames):
    """Map lower-cased header names to the column index they refer to."""
    mapping = {}
    for index, name in enumerate(fieldnames or []):
        if name is None:
            continue
        key = str(name).strip().lower()
        if key:
            mapping.setdefault(key, index)
    return mapping


def resolve_header_index(columns, header_key):
    """Pick the first file column that satisfies a logical header name."""
    for alias in COLUMN_ALIASES[header_key]:
        index = columns.get(alias)
        if index is not None:
            return index
    return None


def detect_encoding(file_obj):
    """Peek at the first bytes of the upload and pick a text encoding."""
    try:
        file_obj.seek(0)
    except (AttributeError, ValueError, OSError):
        pass
    try:
        sample = file_obj.read(8192)
    except (AttributeError, OSError):
        sample = b''
    if not sample:
        raise CSVImportError('The uploaded file is empty.')

    if isinstance(sample, str):
        text_sample = sample
        encoding = 'utf-8-sig'
    else:
        encoding = 'utf-8-sig'
        try:
            sample.decode('utf-8')
        except UnicodeDecodeError:
            encoding = 'latin-1'
        text_sample = sample.decode(encoding, 'replace')

    if not text_sample.strip().lstrip('\ufeff'):
        raise CSVImportError('The uploaded file is empty.')

    try:
        file_obj.seek(0)
    except (AttributeError, ValueError, OSError):
        pass
    return encoding


def iter_lines(file_obj, encoding):
    """Yield decoded lines from a binary (or text) stream chunk by chunk."""
    decoder = codecs.getincrementaldecoder(encoding)(errors='replace')
    pending = ''
    while True:
        chunk = file_obj.read(READ_CHUNK)
        if not chunk:
            break
        pending += chunk if isinstance(chunk, str) else decoder.decode(chunk)
        lines = pending.splitlines(keepends=True)
        last = lines[-1] if lines else ''
        if last.endswith(('\n', '\r')):
            pending = ''
        else:
            pending = lines.pop() if lines else ''
        for line in lines:
            yield line
    pending += decoder.decode(b'', final=True)
    if pending:
        yield pending


def parse_rating(raw):
    """Return an int rating in 1-5, or ``None`` when missing/invalid."""
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        match = RATING_TEXT_PATTERN.fullmatch(value)
        return int(match.group(1)) if match else None
    if number != number or number in (float('inf'), float('-inf')):
        return None
    if number != int(number):
        return None
    number = int(number)
    if 1 <= number <= 5:
        return number
    return None


def parse_review_date(raw):
    """Parse common CSV date formats into an aware datetime, or ``None``."""
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None

    parsed = None
    if value.isdigit() and len(value) in (10, 13):
        seconds = int(value) / (1000 if len(value) == 13 else 1)
        try:
            parsed = datetime.fromtimestamp(seconds, tz=dt_timezone.utc)
        except (OverflowError, OSError, ValueError):
            parsed = None
    if parsed is None:
        parsed = parse_datetime(value)
    if parsed is None:
        for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y-%m-%d %H:%M:%S'):
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    if settings.USE_TZ and timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed, dt_timezone.utc)
    return parsed


def import_reviews(file_obj):
    """Import reviews from an uploaded CSV file.

    Returns a summary dict with the number of rows processed, imported,
    duplicated, invalid, plus any row level errors. Raises
    :class:`CSVImportError` when the file itself is unusable.
    """
    summary = {
        'processed': 0,
        'imported': 0,
        'duplicates': 0,
        'invalid': 0,
        'unrated': 0,
        'errors': [],
        'columns': [],
        'batch_size': BATCH_SIZE,
    }

    encoding = detect_encoding(file_obj)
    reader = csv.reader(iter_lines(file_obj, encoding))
    try:
        header = next(reader)
    except StopIteration:
        raise CSVImportError('The uploaded file is empty.')
    except csv.Error as exc:
        raise CSVImportError(f'The file could not be parsed as CSV: {exc}.')

    header = [cell for cell in header] if header else []
    summary['columns'] = [str(cell).strip() for cell in header if cell and str(cell).strip()]

    if not summary['columns']:
        raise CSVImportError('The uploaded file is empty.')

    columns = build_header_map(header)
    text_index = resolve_header_index(columns, TEXT_HEADER)
    if text_index is None:
        available = ', '.join(summary['columns'])
        accepted = ', '.join(name.title() for name in COLUMN_ALIASES[TEXT_HEADER])
        raise CSVImportError(
            f"Missing required column '{REQUIRED_COLUMN}' "
            f'(accepted headers: {accepted}). Found columns: {available}.'
        )

    score_index = resolve_header_index(columns, SCORE_HEADER)
    product_index = resolve_header_index(columns, PRODUCT_HEADER)
    time_index = resolve_header_index(columns, TIME_HEADER)
    summary_index = resolve_header_index(columns, SUMMARY_HEADER)
    expected_width = len(header)

    batch = []

    def add_error(line_number, reason):
        summary['invalid'] += 1
        if len(summary['errors']) < MAX_ERROR_DETAILS:
            summary['errors'].append(f'Row {line_number}: {reason}')

    def flush():
        if batch:
            _insert_batch(batch, summary)
            del batch[:]

    try:
        for line_number, raw_row in enumerate(reader, start=2):
            if not raw_row or not any(str(cell or '').strip() for cell in raw_row):
                continue
            summary['processed'] += 1

            if len(raw_row) != expected_width:
                add_error(
                    line_number,
                    f'malformed row (expected {expected_width} columns, found {len(raw_row)})',
                )
                continue

            review_text = str(raw_row[text_index] or '')
            if not review_text.strip():
                add_error(line_number, 'missing review text')
                continue

            rating = parse_rating(raw_row[score_index]) if score_index is not None else None
            product_id = (
                str(raw_row[product_index] or '').strip()
                if product_index is not None
                else ''
            )
            batch.append(
                Review(
                    text=review_text,
                    summary=str(raw_row[summary_index] or '').strip()
                    if summary_index is not None
                    else '',
                    rating=rating,
                    product_id=product_id,
                    review_date=parse_review_date(raw_row[time_index])
                    if time_index is not None
                    else None,
                    sentiment=sentiment_from_rating(rating),
                    content_hash=review_content_hash(review_text, product_id),
                )
            )
            if len(batch) >= BATCH_SIZE:
                flush()
    except csv.Error as exc:
        raise CSVImportError(f'The file could not be parsed as CSV: {exc}.')
    finally:
        try:
            file_obj.seek(0)
        except (AttributeError, ValueError, OSError):
            pass

    flush()

    if summary['processed'] == 0:
        summary['notice'] = 'The file has a header but no data rows.'
    return summary


def _insert_batch(batch, summary):
    """Write one batch, skipping duplicates inside the batch and in the DB."""
    local_hashes = set()
    candidates = []
    for review in batch:
        if review.content_hash in local_hashes:
            summary['duplicates'] += 1
            continue
        local_hashes.add(review.content_hash)
        candidates.append(review)

    if not candidates:
        return

    existing = set(
        Review.objects.filter(
            content_hash__in=[review.content_hash for review in candidates]
        ).values_list('content_hash', flat=True)
    )

    to_create = []
    for review in candidates:
        if review.content_hash in existing:
            summary['duplicates'] += 1
            continue
        to_create.append(review)

    if not to_create:
        return

    saved = []
    try:
        with transaction.atomic():
            Review.objects.bulk_create(to_create, batch_size=BATCH_SIZE)
        saved = to_create
    except IntegrityError:
        for review in to_create:
            try:
                with transaction.atomic():
                    review.save(force_insert=True)
                saved.append(review)
            except IntegrityError:
                summary['duplicates'] += 1

    summary['imported'] += len(saved)
    summary['unrated'] += sum(1 for review in saved if review.sentiment in (None, ''))
