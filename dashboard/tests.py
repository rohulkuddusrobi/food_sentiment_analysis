import csv
import io
import os
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import mock

import joblib
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse

from . import analytics, csv_export, csv_import, ml_service, negative_analysis, queries
from .csv_import import BATCH_SIZE, import_reviews, parse_rating, parse_review_date
from .forms import DashboardFilterForm, NegativeFilterForm, PredictionForm, ReviewFilterForm
from .models import PredictionHistory, Review, make_content_hash, sentiment_from_rating


def csv_upload(content, name='reviews.csv'):
    if isinstance(content, str):
        content = content.encode('utf-8')
    return SimpleUploadedFile(name, content, content_type='text/csv')


def post_upload(client, content, name='reviews.csv'):
    return client.post(reverse('upload_reviews'), {'file': csv_upload(content, name)})


def message_texts(response):
    return [str(message) for message in response.context['messages']]


def export_rows(response):
    """Decode a streaming CSV response into a list of parsed rows."""
    content = b''.join(response.streaming_content).decode('utf-8')
    return list(csv.reader(io.StringIO(content)))


class SentimentMappingTests(TestCase):
    def test_ratings_map_to_sentiment(self):
        self.assertEqual(sentiment_from_rating(1), 'Negative')
        self.assertEqual(sentiment_from_rating(2), 'Negative')
        self.assertEqual(sentiment_from_rating(3), 'Neutral')
        self.assertEqual(sentiment_from_rating(4), 'Positive')
        self.assertEqual(sentiment_from_rating(5), 'Positive')
        self.assertEqual(sentiment_from_rating('4'), 'Positive')

    def test_invalid_ratings_have_no_sentiment(self):
        for value in (None, '', '   ', 'abc', 0, 6, -1, 4.5, 'N/A'):
            self.assertIsNone(sentiment_from_rating(value), value)

    def test_parse_rating(self):
        self.assertEqual(parse_rating(' 5 '), 5)
        self.assertEqual(parse_rating('4.0'), 4)
        self.assertIsNone(parse_rating('4.5'))
        self.assertIsNone(parse_rating('score'))
        self.assertIsNone(parse_rating(None))

    def test_parse_review_date(self):
        parsed = parse_review_date('2024-05-01 12:30:00')
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.year, 2024)
        self.assertIsNotNone(parse_review_date('2024-05-01T12:30:00Z'))
        self.assertIsNotNone(parse_review_date('1398960000'))
        self.assertIsNone(parse_review_date('not-a-date'))
        self.assertIsNone(parse_review_date(''))


class ReviewModelTests(TestCase):
    def test_unrated_review_has_no_sentiment(self):
        review = Review.objects.create(text='no rating here')
        self.assertIsNone(review.sentiment)
        self.assertIsNone(review.rating)
        self.assertEqual(review.content_hash, make_content_hash('no rating here', ''))

    def test_review_str(self):
        review = Review(text='nice', product_id='B001', sentiment='Positive')
        self.assertIn('Positive', str(review))


class UploadPageTests(TestCase):
    def test_home_page_still_works(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dashboard Overview')

    def test_upload_page_renders(self):
        response = self.client.get(reverse('upload_reviews'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="file"')
        self.assertContains(response, 'Upload and import')


class UploadImportTests(TestCase):
    def test_valid_csv_is_imported(self):
        content = (
            'Text,Score,ProductId,Time,Summary\n'
            'Great product,5,B0001,2024-05-01 12:30:00,Loved it\n'
            'It is okay,3,B0002,2024-06-01T00:00:00Z,Mediocre\n'
            'Awful experience,1,B0003,1398960000,Bad\n'
        )
        response = post_upload(self.client, content)
        self.assertEqual(response.status_code, 200)

        summary = response.context['summary']
        self.assertEqual(summary['processed'], 3)
        self.assertEqual(summary['imported'], 3)
        self.assertEqual(summary['duplicates'], 0)
        self.assertEqual(summary['invalid'], 0)

        self.assertEqual(Review.objects.count(), 3)
        positive = Review.objects.get(product_id='B0001')
        self.assertEqual(positive.sentiment, 'Positive')
        self.assertEqual(positive.rating, 5)
        self.assertEqual(positive.summary, 'Loved it')
        self.assertIsNotNone(positive.review_date)

        self.assertEqual(Review.objects.get(product_id='B0002').sentiment, 'Neutral')
        self.assertEqual(Review.objects.get(product_id='B0003').sentiment, 'Negative')

    def test_headers_are_case_insensitive(self):
        response = post_upload(self.client, 'text,score\n"good stuff",4\n')
        summary = response.context['summary']
        self.assertEqual(summary['imported'], 1)
        self.assertEqual(Review.objects.get().sentiment, 'Positive')

    def test_optional_columns_can_be_missing(self):
        response = post_upload(self.client, 'Text\nOnly review text here\n')
        summary = response.context['summary']
        self.assertEqual(summary['imported'], 1)
        review = Review.objects.get()
        self.assertIsNone(review.rating)
        self.assertIsNone(review.sentiment)
        self.assertEqual(review.product_id, '')

    def test_missing_text_column_is_rejected(self):
        response = post_upload(self.client, 'Score,ProductId\n5,B0001\n')
        self.assertIsNone(response.context['summary'])
        self.assertEqual(Review.objects.count(), 0)
        self.assertTrue(
            any('Missing required column' in text for text in message_texts(response))
        )

    def test_empty_file_is_rejected(self):
        response = post_upload(self.client, b'')
        self.assertIsNone(response.context['summary'])
        self.assertEqual(Review.objects.count(), 0)
        errors = [str(error) for error in response.context['form'].errors.get('file', [])]
        self.assertTrue(any('empty' in error.lower() for error in errors), errors)

    def test_whitespace_only_file_is_rejected(self):
        response = post_upload(self.client, b'   \n\n')
        self.assertIsNone(response.context['summary'])
        self.assertTrue(any('empty' in text.lower() for text in message_texts(response)))

    def test_header_only_file_reports_notice(self):
        response = post_upload(self.client, 'Text,Score\n')
        summary = response.context['summary']
        self.assertEqual(summary['processed'], 0)
        self.assertEqual(summary['imported'], 0)
        self.assertIn('notice', summary)

    def test_non_csv_extension_is_rejected(self):
        response = post_upload(self.client, 'Text\nhello\n', name='reviews.txt')
        self.assertFormError(
            response.context['form'],
            'file',
            'Please choose a file with a .csv extension.',
        )
        self.assertEqual(Review.objects.count(), 0)

    def test_row_without_text_is_invalid(self):
        content = 'Text,Score\nGood one,5\n   ,2\nAlso good,1\n'
        response = post_upload(self.client, content)
        summary = response.context['summary']
        self.assertEqual(summary['processed'], 3)
        self.assertEqual(summary['imported'], 2)
        self.assertEqual(summary['invalid'], 1)
        self.assertEqual(summary['duplicates'], 0)
        self.assertTrue(any('missing review text' in e for e in summary['errors']))
        self.assertEqual(Review.objects.count(), 2)

    def test_malformed_row_is_invalid(self):
        content = 'Text,Score\nGood one,5,extra,B0001\nFine,4\n'
        response = post_upload(self.client, content)
        summary = response.context['summary']
        self.assertEqual(summary['invalid'], 1)
        self.assertEqual(summary['imported'], 1)
        self.assertTrue(any('malformed row' in e for e in summary['errors']))

    def test_short_row_is_invalid(self):
        content = 'Text,Score,ProductId\nOnly text\nGood one,5,B0001\n'
        response = post_upload(self.client, content)
        summary = response.context['summary']
        self.assertEqual(summary['invalid'], 1)
        self.assertEqual(summary['imported'], 1)

    def test_duplicate_rows_in_file_are_skipped(self):
        content = (
            'Text,Score,ProductId\n'
            'Same review,5,B0001\n'
            'Same review,5,B0001\n'
            'Other review,1,B0001\n'
        )
        response = post_upload(self.client, content)
        summary = response.context['summary']
        self.assertEqual(summary['processed'], 3)
        self.assertEqual(summary['imported'], 2)
        self.assertEqual(summary['duplicates'], 1)
        self.assertEqual(Review.objects.count(), 2)

    def test_rows_already_in_database_are_skipped(self):
        Review.objects.create(
            text='Already imported',
            product_id='B0007',
            rating=4,
            sentiment='Positive',
            content_hash=make_content_hash('Already imported', 'B0007'),
        )
        content = 'Text,Score,ProductId\nAlready imported,4,B0007\nNew review,2,B0007\n'
        response = post_upload(self.client, content)
        summary = response.context['summary']
        self.assertEqual(summary['imported'], 1)
        self.assertEqual(summary['duplicates'], 1)
        self.assertEqual(Review.objects.count(), 2)

    def test_invalid_rating_leaves_sentiment_unset(self):
        content = 'Text,Score\nGreat,9\nTerrible,abc\nMeh,\n'
        response = post_upload(self.client, content)
        summary = response.context['summary']
        self.assertEqual(summary['imported'], 3)
        self.assertEqual(summary['invalid'], 0)
        self.assertEqual(summary['unrated'], 3)
        self.assertEqual(Review.objects.filter(sentiment__isnull=True).count(), 3)

    def test_large_file_is_imported_in_batches(self):
        total = BATCH_SIZE * 2 + 37
        buffer = io.StringIO()
        buffer.write('Text,Score,ProductId\n')
        for index in range(total):
            buffer.write(f'Review number {index},4,B{index:05d}\n')
        response = post_upload(self.client, buffer.getvalue())
        summary = response.context['summary']
        self.assertEqual(summary['processed'], total)
        self.assertEqual(summary['imported'], total)
        self.assertEqual(summary['invalid'], 0)
        self.assertEqual(summary['duplicates'], 0)
        self.assertEqual(Review.objects.count(), total)

    def test_bom_and_crlf_are_handled(self):
        content = 'Text,Score\r\nNice one,5\r\n'
        response = post_upload(
            self.client, b'\xef\xbb\xbf' + content.encode('utf-8')
        )
        summary = response.context['summary']
        self.assertEqual(summary['imported'], 1)
        self.assertEqual(Review.objects.get().text, 'Nice one')

    def test_get_request_has_no_summary(self):
        response = self.client.get(reverse('upload_reviews'))
        self.assertIsNone(response.context['summary'])


class DirectImportTests(TestCase):
    def test_import_reviews_helper_summary(self):
        upload = csv_upload('Text,Score\nSolid,4\n')
        summary = import_reviews(upload)
        self.assertEqual(summary['columns'], ['Text', 'Score'])
        self.assertEqual(summary['imported'], 1)
        self.assertEqual(Review.objects.count(), 1)

    def test_quoted_multiline_text(self):
        content = 'Text,Score\n"Line one\nLine two",2\n'
        summary = import_reviews(csv_upload(content))
        self.assertEqual(summary['imported'], 1)
        self.assertIn('Line two', Review.objects.get().text)


class DatasetHeaderAliasTests(TestCase):
    """Import behaviour for the real dataset schema (Amazon_Reviews_Cleaned_Phase2)."""

    HEADER = [
        'Reviewer Name', 'Profile Link', 'Country', 'Review Count', 'Review Date',
        'Rating', 'Review Title', 'Review Text', 'Date of Experience',
        'Cleaned Review', 'Rating_Number', 'Sentiment', 'Review_Length',
    ]

    def build_csv(self, rows):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(self.HEADER)
        writer.writerows(rows)
        return buffer.getvalue()

    def dataset_row(self, index, rating, title, text, cleaned=None, date='2024-09-16T13:44:26.000Z'):
        return [
            f'Reviewer {index}', f'/users/{index}', 'US', '1 review', date,
            f'Rated {rating} out of 5 stars', title, text, '2024-09-15',
            cleaned if cleaned is not None else text.lower(), rating,
            {1: 'Negative', 2: 'Negative', 3: 'Neutral', 4: 'Positive', 5: 'Positive'}[rating],
            len(text),
        ]

    def test_dataset_headers_are_imported(self):
        content = self.build_csv([
            self.dataset_row(1, 1, 'Terrible service', 'The parcel never arrived and support ignored me.'),
            self.dataset_row(2, 5, 'Great shop', 'Fast delivery, item exactly as described.'),
            self.dataset_row(3, 3, 'It is fine', 'Average experience, nothing to complain about.'),
            self.dataset_row(4, 1, 'Terrible service', 'The parcel never arrived and support ignored me.'),
        ])
        summary = import_reviews(csv_upload(content, 'dataset.csv'))
        self.assertEqual(summary['processed'], 4)
        self.assertEqual(summary['imported'], 3)
        self.assertEqual(summary['duplicates'], 1)
        self.assertEqual(summary['invalid'], 0)

        negative = Review.objects.get(rating=1)
        self.assertEqual(negative.text, 'The parcel never arrived and support ignored me.')
        self.assertEqual(negative.summary, 'Terrible service')
        self.assertEqual(negative.sentiment, 'Negative')
        self.assertEqual(negative.product_id, '')
        self.assertEqual(negative.review_date.year, 2024)
        self.assertEqual(Review.objects.get(rating=5).sentiment, 'Positive')
        self.assertEqual(Review.objects.get(rating=3).sentiment, 'Neutral')

    def test_review_text_preferred_over_cleaned_review(self):
        content = self.build_csv([
            self.dataset_row(
                1, 5, 'Nice', 'Original Casing Matters', cleaned='nice and lowercased'
            ),
        ])
        import_reviews(csv_upload(content))
        self.assertEqual(Review.objects.get().text, 'Original Casing Matters')

    def test_falls_back_to_cleaned_review_only(self):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(['Cleaned Review', 'Rating_Number'])
        writer.writerow(['works on its own', 4])
        summary = import_reviews(csv_upload(buffer.getvalue()))
        self.assertEqual(summary['imported'], 1)
        review = Review.objects.get()
        self.assertEqual(review.text, 'works on its own')
        self.assertEqual(review.rating, 4)

    def test_score_header_beats_rating_number(self):
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(['Text', 'Score', 'Rating_Number'])
        writer.writerow(['two columns', 2, 5])
        import_reviews(csv_upload(buffer.getvalue()))
        self.assertEqual(Review.objects.get().rating, 2)

    def test_platform_style_rating_text_is_parsed(self):
        self.assertEqual(parse_rating('Rated 4 out of 5 stars'), 4)
        self.assertEqual(parse_rating('rated 1 of 5 stars'), 1)
        self.assertIsNone(parse_rating('Rated 6 out of 5 stars'))
        self.assertIsNone(parse_rating('not a rating'))

    def test_missing_column_message_lists_accepted_headers(self):
        with self.assertRaises(csv_import.CSVImportError) as context:
            import_reviews(csv_upload('Score,Rating_Number\n5,5\n'))
        message = str(context.exception)
        self.assertIn('Missing required column', message)
        self.assertIn('Review Text', message)


class _FakeVectorizer:
    def transform(self, texts):
        return texts


class _FakeClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, features):
        return [self.label]


def fake_model(logistic='Positive', svm='Positive', sgd='Negative', classes=None):
    """Stand-in for the saved ensemble so tests never need the artifact."""
    return {
        'tfidf_vectorizer': _FakeVectorizer(),
        'logistic_regression': _FakeClassifier(logistic),
        'linear_svm': _FakeClassifier(svm),
        'sgd_classifier': _FakeClassifier(sgd),
        'classes': classes
        if classes is not None
        else ['Negative', 'Neutral', 'Positive'],
    }


STATUS_OK = {
    'path': 'ml_models/sentiment_ensemble_model.joblib',
    'loaded': True,
    'error': None,
    'warning': None,
}

STATUS_DOWN = {
    'path': 'ml_models/sentiment_ensemble_model.joblib',
    'loaded': False,
    'error': 'Sentiment model file not found.',
    'warning': None,
}


class MLServiceTests(TestCase):
    def setUp(self):
        ml_service.reset_cache()

    def tearDown(self):
        ml_service.reset_cache()

    def test_model_path_is_based_on_project_root(self):
        path = ml_service.model_path()
        self.assertTrue(path.is_absolute())
        self.assertEqual(path.name, 'sentiment_ensemble_model.joblib')

    def test_empty_text_is_rejected(self):
        for value in ('', '   ', None):
            with self.assertRaises(ml_service.MLServiceError) as context:
                ml_service.predict(value)
            self.assertIn('review text', str(context.exception))

    def test_hard_voting_picks_majority(self):
        model = fake_model(logistic='Positive', svm='Positive', sgd='Negative')
        with mock.patch.object(ml_service, 'load_model', return_value=model):
            self.assertEqual(ml_service.predict('Great product'), 'Positive')

    def test_hard_voting_can_predict_each_class(self):
        for label in ml_service.VALID_LABELS:
            model = fake_model(logistic=label, svm=label, sgd=label)
            with mock.patch.object(ml_service, 'load_model', return_value=model):
                self.assertEqual(ml_service.predict('some text'), label)

    def test_hard_voting_tie_follows_saved_class_order(self):
        model = fake_model(logistic='Positive', svm='Neutral', sgd='Negative')
        with mock.patch.object(ml_service, 'load_model', return_value=model):
            self.assertEqual(ml_service.predict('mixed feelings'), 'Negative')

    def test_hard_voting_ignores_out_of_order_votes(self):
        model = fake_model(logistic='Negative', svm='Negative', sgd='Positive')
        with mock.patch.object(ml_service, 'load_model', return_value=model):
            self.assertEqual(ml_service.predict('bad'), 'Negative')

    def test_unexpected_label_is_reported(self):
        model = fake_model(logistic='Positive', svm='Positive', sgd='Positive')
        model['sgd_classifier'].label = 'Unsure'
        with mock.patch.object(ml_service, 'load_model', return_value=model):
            with self.assertRaises(ml_service.MLServiceError):
                ml_service.predict('anything')

    def test_prediction_error_is_wrapped(self):
        class Exploding:
            def predict(self, features):
                raise RuntimeError('vectorizer exploded')

        model = fake_model()
        model['linear_svm'] = Exploding()
        with mock.patch.object(ml_service, 'load_model', return_value=model):
            with self.assertLogs('dashboard.ml_service', level='ERROR'):
                with self.assertRaises(ml_service.MLServiceError) as context:
                    ml_service.predict('text')
        self.assertIn('Prediction failed', str(context.exception))

    @override_settings(SENTIMENT_MODEL_PATH='/nonexistent/model/does_not_exist.joblib')
    def test_missing_model_file_reports_error(self):
        ml_service.reset_cache()
        with self.assertRaises(ml_service.MLServiceError) as context:
            ml_service.predict('some review')
        self.assertIn('not found', str(context.exception))

        status = ml_service.get_status()
        self.assertFalse(status['loaded'])
        self.assertIn('not found', status['error'])

    def test_corrupt_model_file_reports_error(self):
        with NamedTemporaryFile(suffix='.joblib', delete=False) as handle:
            handle.write(b'this is not a joblib payload')
            path = Path(handle.name)
        try:
            with override_settings(SENTIMENT_MODEL_PATH=path):
                ml_service.reset_cache()
                with self.assertLogs('dashboard.ml_service', level='ERROR'):
                    with self.assertRaises(ml_service.MLServiceError) as context:
                        ml_service.predict('some review')
                self.assertIn('could not be loaded', str(context.exception))
                self.assertFalse(ml_service.get_status()['loaded'])
        finally:
            path.unlink(missing_ok=True)

    def test_incompatible_payload_reports_error(self):
        with NamedTemporaryFile(suffix='.joblib', delete=False) as handle:
            path = Path(handle.name)
        try:
            joblib.dump({'tfidf_vectorizer': _FakeVectorizer()}, path)
            with override_settings(SENTIMENT_MODEL_PATH=path):
                ml_service.reset_cache()
                with self.assertRaises(ml_service.MLServiceError) as context:
                    ml_service.load_model()
                message = str(context.exception)
                self.assertIn('incompatible', message)
                self.assertIn('missing component', message)
                self.assertIn('not retrained', message)
                self.assertFalse(ml_service.get_status()['loaded'])
        finally:
            path.unlink(missing_ok=True)

    def test_incompatible_classes_report_error(self):
        with NamedTemporaryFile(suffix='.joblib', delete=False) as handle:
            path = Path(handle.name)
        try:
            joblib.dump(fake_model(classes=['good', 'bad']), path)
            with override_settings(SENTIMENT_MODEL_PATH=path):
                ml_service.reset_cache()
                with self.assertRaises(ml_service.MLServiceError) as context:
                    ml_service.load_model()
                self.assertIn('incompatible', str(context.exception))
        finally:
            path.unlink(missing_ok=True)

    def test_status_reports_loaded_model(self):
        with mock.patch.object(ml_service, 'load_model', return_value=fake_model()):
            status = ml_service.get_status()
        self.assertTrue(status['loaded'])
        self.assertIsNone(status['error'])

    def test_repeated_loads_reuse_cached_model(self):
        path = Path(ml_service.model_path())
        if not path.exists():
            self.skipTest('trained model artifact is not available')
        first = ml_service.load_model()
        second = ml_service.load_model()
        self.assertIs(first, second)
        label = ml_service.predict('This product is amazing, I love it!')
        self.assertIn(label, ml_service.VALID_LABELS)
        self.assertEqual(sorted(ml_service.VALID_LABELS), ['Negative', 'Neutral', 'Positive'])


class PredictionHistoryTests(TestCase):
    def test_history_is_stored_separately_from_reviews(self):
        review = Review.objects.create(
            text='Imported review',
            product_id='B0001',
            rating=5,
            sentiment=sentiment_from_rating(5),
        )
        PredictionHistory.objects.create(text='Typed review', predicted_sentiment='Negative')

        review.refresh_from_db()
        self.assertEqual(review.sentiment, 'Positive')
        self.assertEqual(review.rating, 5)
        self.assertEqual(PredictionHistory.objects.count(), 1)
        self.assertEqual(Review.objects.count(), 1)

    def test_history_is_ordered_newest_first(self):
        PredictionHistory.objects.create(text='first', predicted_sentiment='Positive')
        PredictionHistory.objects.create(text='second', predicted_sentiment='Negative')
        texts = list(
            PredictionHistory.objects.values_list('text', flat=True)
        )
        self.assertEqual(texts, ['second', 'first'])

    def test_str_and_short_text(self):
        history = PredictionHistory.objects.create(
            text='x' * 200, predicted_sentiment='Neutral'
        )
        self.assertIn('Neutral', str(history))
        self.assertTrue(history.short_text.endswith('...'))
        self.assertEqual(len(history.short_text), 80)


class PredictPageTests(TestCase):
    def get_page(self, status=STATUS_OK):
        with mock.patch.object(ml_service, 'get_status', return_value=status):
            return self.client.get(reverse('predict_review'))

    def post_page(self, text, predict_result='Positive', status=STATUS_OK, side_effect=None):
        with mock.patch.object(ml_service, 'get_status', return_value=status), \
                mock.patch.object(ml_service, 'predict', return_value=predict_result,
                                  side_effect=side_effect) as predict:
            response = self.client.post(reverse('predict_review'), {'text': text})
        return response, predict

    def test_get_renders_form_and_status(self):
        response = self.get_page()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'name="text"')
        self.assertContains(response, 'Predict Sentiment')
        self.assertContains(response, 'Model loaded and ready.')

    def test_nav_links_are_preserved(self):
        response = self.get_page()
        self.assertContains(response, f'href="{reverse("home")}"')
        self.assertContains(response, f'href="{reverse("upload_reviews")}"')
        self.assertContains(response, f'href="{reverse("predict_review")}"')

    def test_model_status_shows_only_the_file_name(self):
        real_path = str(ml_service.model_path())
        status = dict(
            STATUS_OK,
            path=real_path,
            warning=f'Trained elsewhere: {real_path}',
        )
        response = self.get_page(status)
        self.assertContains(response, 'Model file: sentiment_ensemble_model.joblib')
        self.assertNotContains(response, real_path)

    def test_model_error_does_not_leak_the_server_path(self):
        real_path = str(ml_service.model_path())
        status = dict(
            STATUS_DOWN,
            path=real_path,
            error=f'Sentiment model file not found at {real_path}.',
        )
        response = self.get_page(status)
        self.assertContains(response, 'sentiment_ensemble_model.joblib')
        self.assertNotContains(response, real_path)

    def test_missing_text_shows_validation_error(self):
        response, predict = self.post_page('')
        self.assertEqual(response.status_code, 200)
        errors = [str(error) for error in response.context['form'].errors.get('text', [])]
        self.assertTrue(any('review text' in error for error in errors), errors)
        predict.assert_not_called()
        self.assertEqual(PredictionHistory.objects.count(), 0)

    def test_valid_prediction_is_stored_and_shown(self):
        response, predict = self.post_page('Absolutely wonderful', predict_result='Positive')
        predict.assert_called_once_with('Absolutely wonderful')

        result = response.context['result']
        self.assertEqual(result['sentiment'], 'Positive')
        self.assertContains(response, 'Predicted sentiment')

        history = PredictionHistory.objects.get()
        self.assertEqual(history.text, 'Absolutely wonderful')
        self.assertEqual(history.predicted_sentiment, 'Positive')

    def test_prediction_error_is_reported_without_history(self):
        response, predict = self.post_page(
            'some review',
            side_effect=ml_service.MLServiceError('Model file not found.'),
        )
        predict.assert_called_once()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(PredictionHistory.objects.count(), 0)
        self.assertIsNone(response.context['result'])
        self.assertTrue(
            any('not found' in str(m) for m in response.context['messages'])
        )

    def test_prediction_error_shows_only_the_model_file_name(self):
        real_path = str(ml_service.model_path())
        response, predict = self.post_page(
            'some review',
            side_effect=ml_service.MLServiceError(
                f'Sentiment model file not found at {real_path}.'
            ),
        )
        predict.assert_called_once()
        self.assertEqual(response.status_code, 200)
        shown = ' '.join(str(message) for message in response.context['messages'])
        self.assertNotIn(real_path, shown)
        self.assertIn('sentiment_ensemble_model.joblib', shown)

    def test_unavailable_model_is_explained_on_the_page(self):
        response = self.get_page(status=STATUS_DOWN)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Model unavailable')
        self.assertContains(response, 'Sentiment model file not found.')

    def test_recent_history_is_limited(self):
        for index in range(15):
            PredictionHistory.objects.create(
                text=f'review {index}', predicted_sentiment='Neutral'
            )
        response = self.get_page()
        limit = response.context['history_limit']
        self.assertEqual(len(response.context['history']), limit)
        self.assertEqual(len(response.context['history']), 10)
        self.assertEqual(response.context['history'][0].text, 'review 14')

    def test_history_table_is_shown(self):
        PredictionHistory.objects.create(text='stored review', predicted_sentiment='Negative')
        response = self.get_page()
        self.assertContains(response, 'stored review')
        self.assertContains(response, 'Recent predictions')

    def test_empty_history_message(self):
        response = self.get_page()
        self.assertContains(response, 'No predictions yet')

    def test_form_object_is_available(self):
        response = self.get_page()
        self.assertIsInstance(response.context['form'], PredictionForm)


class DashboardDatasetMixin:
    """Six imported reviews covering every rating-derived outcome.

    pos5  rating 5, Positive, dated 2024-05-10
    pos4  rating 4, Positive, dated 2024-05-20
    neu3  rating 3, Neutral,  dated 2024-06-05
    neg1  rating 1, Negative, dated 2024-06-15
    neg2  rating 2, Negative, no review date
    empty no rating, no sentiment, dated 2024-06-20
    """

    def setUp(self):
        utc = dt_timezone.utc
        self.pos5 = Review.objects.create(
            text='Excellent product, five stars',
            rating=5,
            sentiment=sentiment_from_rating(5),
            review_date=datetime(2024, 5, 10, tzinfo=utc),
        )
        self.pos4 = Review.objects.create(
            text='Very good purchase overall',
            rating=4,
            sentiment=sentiment_from_rating(4),
            review_date=datetime(2024, 5, 20, tzinfo=utc),
        )
        self.neu3 = Review.objects.create(
            text='It is okay, nothing special',
            rating=3,
            sentiment=sentiment_from_rating(3),
            review_date=datetime(2024, 6, 5, tzinfo=utc),
        )
        self.neg1 = Review.objects.create(
            text='Broke immediately, very bad',
            rating=1,
            sentiment=sentiment_from_rating(1),
            review_date=datetime(2024, 6, 15, tzinfo=utc),
        )
        self.neg2 = Review.objects.create(
            text='Disappointed with the quality',
            rating=2,
            sentiment=sentiment_from_rating(2),
            review_date=None,
        )
        self.empty = Review.objects.create(
            text='No rating given for this one',
            rating=None,
            sentiment=None,
            review_date=datetime(2024, 6, 20, tzinfo=utc),
        )


class DashboardKPITests(DashboardDatasetMixin, TestCase):
    def get_dashboard(self, params=None):
        return self.client.get(reverse('home'), params or {})

    def test_kpi_counts(self):
        response = self.get_dashboard()
        kpis = response.context['kpis']
        self.assertEqual(kpis['total'], 6)
        self.assertEqual(kpis['positive'], 2)
        self.assertEqual(kpis['negative'], 2)
        self.assertEqual(kpis['neutral'], 1)
        self.assertEqual(kpis['unrated'], 1)
        self.assertEqual(kpis['sentiment_base'], 5)
        self.assertEqual(kpis['rated'], 5)

    def test_average_rating_uses_only_valid_ratings(self):
        # (5 + 4 + 3 + 1 + 2) / 5 = 3.0 -- the unrated review is excluded.
        kpis = self.get_dashboard().context['kpis']
        self.assertEqual(kpis['avg_rating'], 3.0)
        self.assertEqual(kpis['avg_rating_display'], '3.00')
        self.assertEqual(kpis['rated'], 5)

    def test_percentages_use_rated_base(self):
        kpis = self.get_dashboard().context['kpis']
        self.assertEqual(kpis['positive_pct'], 40.0)
        self.assertEqual(kpis['negative_pct'], 40.0)
        self.assertEqual(kpis['neutral_pct'], 20.0)
        self.assertEqual(
            round(kpis['positive_pct'] + kpis['negative_pct'] + kpis['neutral_pct'], 1),
            100.0,
        )
        self.assertEqual(kpis['positive_pct_display'], '40.0%')

    def test_missing_rating_percentage_uses_all_reviews(self):
        kpis = self.get_dashboard().context['kpis']
        self.assertEqual(kpis['unrated_pct'], 16.7)
        self.assertEqual(kpis['unrated_pct_display'], '16.7%')

    def test_kpi_cards_are_rendered(self):
        response = self.get_dashboard()
        self.assertContains(response, 'Total reviews')
        self.assertContains(response, 'Average rating')
        self.assertContains(response, 'Missing rating')
        self.assertContains(response, 'of rated reviews')

    def test_dashboard_renders_with_status_200(self):
        self.assertEqual(self.get_dashboard().status_code, 200)


class DashboardEmptyStateTests(TestCase):
    def test_empty_database_renders_zero_state(self):
        response = self.client.get(reverse('home'))
        self.assertEqual(response.status_code, 200)

        kpis = response.context['kpis']
        self.assertEqual(kpis['total'], 0)
        self.assertEqual(kpis['positive'], 0)
        self.assertEqual(kpis['negative'], 0)
        self.assertEqual(kpis['neutral'], 0)
        self.assertEqual(kpis['unrated'], 0)
        self.assertEqual(kpis['rated'], 0)
        self.assertIsNone(kpis['avg_rating'])
        self.assertEqual(kpis['avg_rating_display'], '—')
        self.assertIsNone(kpis['positive_pct'])
        self.assertEqual(kpis['positive_pct_display'], '—')

    def test_empty_database_shows_helpful_message(self):
        response = self.client.get(reverse('home'))
        self.assertContains(response, 'No reviews to analyse')
        self.assertContains(response, 'Upload a CSV file')
        self.assertContains(response, 'No reviews to preview yet')

    def test_empty_database_charts_report_no_data(self):
        charts = self.client.get(reverse('home')).context['charts']
        self.assertFalse(charts['sentiment']['has_data'])
        self.assertFalse(charts['rating']['has_data'])
        self.assertFalse(charts['monthly']['has_data'])
        self.assertFalse(charts['sentiment_by_rating']['has_data'])
        self.assertEqual(charts['sentiment']['data'], [0, 0, 0])
        self.assertEqual(charts['rating']['data'], [0, 0, 0, 0, 0])

    def test_empty_database_has_insufficient_data_insight(self):
        insights = self.client.get(reverse('home')).context['insights']
        self.assertEqual(len(insights), 1)
        self.assertIn('No reviews are available', insights[0])


class DashboardChartTests(DashboardDatasetMixin, TestCase):
    def get_charts(self, params=None):
        return self.client.get(reverse('home'), params or {}).context['charts']

    def test_sentiment_doughnut_data(self):
        chart = self.get_charts()['sentiment']
        self.assertEqual(chart['labels'], ['Positive', 'Neutral', 'Negative'])
        self.assertEqual(chart['data'], [2, 1, 2])
        self.assertTrue(chart['has_data'])

    def test_rating_distribution_data(self):
        chart = self.get_charts()['rating']
        self.assertEqual(chart['labels'], ['1', '2', '3', '4', '5'])
        self.assertEqual(chart['data'], [1, 1, 1, 1, 1])
        self.assertTrue(chart['has_data'])

    def test_monthly_volume_data(self):
        chart = self.get_charts()['monthly']
        # Review without a date (neg2) is excluded from the monthly series.
        self.assertEqual(chart['labels'], ['2024-05', '2024-06'])
        self.assertEqual(chart['data'], [2, 3])
        self.assertTrue(chart['has_data'])

    def test_sentiment_by_rating_data(self):
        chart = self.get_charts()['sentiment_by_rating']
        self.assertEqual(chart['labels'], ['1', '2', '3', '4', '5'])
        datasets = {item['label']: item['data'] for item in chart['datasets']}
        self.assertEqual(datasets['Positive'], [0, 0, 0, 1, 1])
        self.assertEqual(datasets['Neutral'], [0, 0, 1, 0, 0])
        self.assertEqual(datasets['Negative'], [1, 1, 0, 0, 0])
        self.assertTrue(chart['has_data'])

    def test_chart_data_is_embedded_as_json(self):
        response = self.client.get(reverse('home'))
        self.assertContains(response, 'id="dashboard-chart-data"')
        self.assertContains(response, '"has_data": true')
        self.assertContains(response, '2024-06')

    def test_charts_do_not_use_prediction_history(self):
        PredictionHistory.objects.create(
            text='Prediction only record', predicted_sentiment='Negative'
        )
        charts = self.get_charts()
        self.assertEqual(charts['sentiment']['data'], [2, 1, 2])
        self.assertEqual(sum(charts['rating']['data']), 5)


class DashboardFilterTests(DashboardDatasetMixin, TestCase):
    def get_page(self, params):
        return self.client.get(reverse('home'), params)

    def test_date_range_filters_kpis(self):
        response = self.get_page({'start_date': '2024-06-01', 'end_date': '2024-06-30'})
        kpis = response.context['kpis']
        # June: neu3, neg1, empty (neg2 has no date, pos5/pos4 are May)
        self.assertEqual(kpis['total'], 3)
        self.assertEqual(kpis['positive'], 0)
        self.assertEqual(kpis['negative'], 1)
        self.assertEqual(kpis['neutral'], 1)
        self.assertEqual(kpis['unrated'], 1)
        self.assertEqual(kpis['rated'], 2)
        self.assertEqual(kpis['avg_rating'], 2.0)

    def test_date_range_filters_charts_consistently(self):
        response = self.get_page({'start_date': '2024-06-01', 'end_date': '2024-06-30'})
        kpis = response.context['kpis']
        charts = response.context['charts']
        self.assertEqual(kpis['total'], response.context['filtered_count'])
        self.assertEqual(
            kpis['total'],
            sum(charts['rating']['data']) + kpis['unrated'],
        )
        self.assertEqual(charts['monthly']['labels'], ['2024-06'])
        self.assertEqual(charts['monthly']['data'], [3])
        self.assertEqual(charts['sentiment']['data'], [0, 1, 1])

    def test_start_date_only(self):
        kpis = self.get_page({'start_date': '2024-06-01'}).context['kpis']
        self.assertEqual(kpis['total'], 3)  # June reviews, undated neg2 excluded

    def test_end_date_only(self):
        kpis = self.get_page({'end_date': '2024-05-31'}).context['kpis']
        self.assertEqual(kpis['total'], 2)
        self.assertEqual(kpis['positive'], 2)

    def test_filtered_values_are_marked_active(self):
        response = self.get_page({'start_date': '2024-06-01'})
        self.assertTrue(response.context['filter_active'])
        self.assertContains(response, 'Date filter active')
        self.assertContains(response, '3 of 6 reviews in range')

    def test_selected_dates_are_preserved_in_the_form(self):
        response = self.get_page({'start_date': '2024-06-01', 'end_date': '2024-06-30'})
        self.assertContains(response, 'value="2024-06-01"')
        self.assertContains(response, 'value="2024-06-30"')

    def test_reset_option_is_offered(self):
        response = self.get_page({'start_date': '2024-06-01'})
        self.assertContains(response, 'Reset')
        self.assertContains(response, f'href="{reverse("home")}"')

    def test_invalid_start_date_shows_error_and_keeps_all_data(self):
        response = self.get_page({'start_date': 'not-a-date'})
        self.assertEqual(response.status_code, 200)
        errors = response.context['filter_form'].errors.get('start_date', [])
        self.assertTrue(any('valid start date' in str(error) for error in errors), errors)
        self.assertContains(response, 'valid start date')
        self.assertEqual(response.context['kpis']['total'], 6)

    def test_invalid_end_date_shows_error(self):
        response = self.get_page({'end_date': '2024-13-45'})
        self.assertContains(response, 'valid end date')
        self.assertEqual(response.context['kpis']['total'], 6)

    def test_start_after_end_shows_error(self):
        response = self.get_page({'start_date': '2024-06-30', 'end_date': '2024-06-01'})
        self.assertContains(response, 'start date must be on or before')
        self.assertFalse(response.context['filter_active'])
        self.assertEqual(response.context['kpis']['total'], 6)

    def test_empty_filter_parameters_are_valid(self):
        response = self.get_page({})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.context['filter_form'].errors)
        self.assertFalse(response.context['filter_active'])
        self.assertEqual(response.context['kpis']['total'], 6)


class DashboardInsightTests(DashboardDatasetMixin, TestCase):
    def get_insights(self, params=None):
        return self.client.get(reverse('home'), params or {}).context['insights']

    def test_insights_describe_calculated_statistics(self):
        insights = self.get_insights()
        joined = ' '.join(insights)
        self.assertIn(
            'Positive is the largest rating-derived sentiment group: 2 of 5 rated reviews (40.0%)',
            joined,
        )
        self.assertIn(
            '40.0% of rating-derived feedback is Negative (2 of 5 reviews with a valid rating)',
            joined,
        )
        self.assertIn('Average rating is 3.00 out of 5 based on 5 valid ratings', joined)
        self.assertIn(
            'Monthly review volume rose from 2 in 2024-05 to 3 in 2024-06 (+1) (+50.0%)',
            joined,
        )
        self.assertEqual(len(insights), 4)

    def test_insights_avoid_causal_claims(self):
        joined = ' '.join(self.get_insights()).lower()
        for forbidden in ('because', 'caused by', 'customers prefer', 'trend shows'):
            self.assertNotIn(forbidden, joined)

    def test_undated_reviews_do_not_break_monthly_insight(self):
        Review.objects.filter(review_date__isnull=False).delete()
        insights = self.get_insights()
        self.assertTrue(
            any('No reviews with a valid review date' in item for item in insights)
        )

    def test_single_month_has_no_volume_change(self):
        Review.objects.all().delete()
        Review.objects.create(
            text='Only one month of data',
            rating=5,
            sentiment=sentiment_from_rating(5),
            review_date=datetime(2024, 6, 1, tzinfo=dt_timezone.utc),
        )
        insights = self.get_insights()
        self.assertTrue(
            any('Only one month of dated reviews' in item for item in insights)
        )

    def test_no_valid_ratings_insight(self):
        Review.objects.all().delete()
        Review.objects.create(text='Unrated only', rating=None, sentiment=None)
        insights = ' '.join(self.get_insights())
        self.assertIn('No reviews with rating-derived sentiment', insights)
        self.assertIn('No valid ratings (1 to 5)', insights)

    def test_negative_share_insight_when_zero(self):
        Review.objects.all().delete()
        Review.objects.create(
            text='All positive', rating=5, sentiment=sentiment_from_rating(5)
        )
        insights = ' '.join(self.get_insights())
        self.assertIn('No Negative rating-derived feedback', insights)

    def test_filtered_insights_mention_the_range(self):
        insights = self.get_insights(
            {'start_date': '2024-06-01', 'end_date': '2024-06-30'}
        )
        with_scope = [item for item in insights if 'selected date range' in item]
        self.assertTrue(with_scope)
        self.assertTrue(
            any('Average rating is 2.00 out of 5' in item for item in with_scope)
        )


class DashboardPreviewTests(DashboardDatasetMixin, TestCase):
    def test_latest_reviews_are_shown(self):
        response = self.client.get(reverse('home'))
        self.assertContains(response, 'Latest imported reviews')
        self.assertContains(response, 'Excellent product, five stars')
        self.assertContains(response, 'No rating given for this one')

    def test_preview_shows_rating_sentiment_and_date(self):
        response = self.client.get(reverse('home'))
        self.assertContains(response, 'Not rated')
        self.assertContains(response, '2024-06-20')
        self.assertContains(response, 'badge text-bg-success')
        self.assertContains(response, 'badge text-bg-danger')

    def test_preview_is_limited(self):
        for index in range(20):
            Review.objects.create(
                text=f'Extra review number {index}',
                rating=3,
                sentiment=sentiment_from_rating(3),
            )
        response = self.client.get(reverse('home'))
        reviews = response.context['latest_reviews']
        self.assertEqual(len(reviews), analytics.PREVIEW_LIMIT)
        self.assertEqual(len(reviews), 8)
        self.assertEqual(reviews[0].text, 'Extra review number 19')

    def test_links_to_upload_and_prediction_pages(self):
        response = self.client.get(reverse('home'))
        self.assertContains(response, f'href="{reverse("upload_reviews")}"')
        self.assertContains(response, f'href="{reverse("predict_review")}"')
        self.assertContains(response, 'Upload CSV')
        self.assertContains(response, 'Predict Sentiment')

    def test_prediction_history_is_not_mixed_into_analytics(self):
        PredictionHistory.objects.create(
            text='ML ONLY SECRET RECORD', predicted_sentiment='Negative'
        )
        response = self.client.get(reverse('home'))
        self.assertEqual(response.context['kpis']['total'], 6)
        self.assertEqual(response.context['kpis']['negative'], 2)
        self.assertNotContains(response, 'ML ONLY SECRET RECORD')

    def test_upload_and_prediction_pages_still_work_with_filter_params(self):
        upload = self.client.get(reverse('upload_reviews'), {'start_date': '2024-06-01'})
        predict = self.client.get(reverse('predict_review'), {'start_date': '2024-06-01'})
        self.assertEqual(upload.status_code, 200)
        self.assertEqual(predict.status_code, 200)
        self.assertContains(upload, 'Upload and import')
        self.assertContains(predict, 'name="text"')


class AnalyticsHelperTests(TestCase):
    def test_percentage(self):
        self.assertEqual(analytics.percentage(1, 4), 25.0)
        self.assertEqual(analytics.percentage(0, 4), 0.0)
        self.assertIsNone(analytics.percentage(1, 0))

    def test_format_percent(self):
        self.assertEqual(analytics.format_percent(12.5), '12.5%')
        self.assertEqual(analytics.format_percent(None), '—')

    def test_date_range_bounds_are_half_open(self):
        from datetime import date

        gte, lt = analytics.date_range_bounds(date(2024, 6, 1), date(2024, 6, 30))
        self.assertEqual(gte.hour, 0)
        self.assertEqual((lt - gte).days, 30)

    def test_apply_date_filter_without_dates_keeps_queryset(self):
        queryset = Review.objects.all()
        self.assertEqual(analytics.apply_date_filter(queryset, None, None), queryset)


class ReviewExplorerTests(DashboardDatasetMixin, TestCase):
    def get_page(self, params=None):
        return self.client.get(reverse('reviews'), params or {})

    def seed_searchable_reviews(self, count=6):
        for index in range(count):
            Review.objects.create(
                text=f'review body number {index}',
                rating=3,
                sentiment=sentiment_from_rating(3),
            )


    def test_page_renders_with_all_reviews(self):
        response = self.get_page()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Review Explorer')
        self.assertContains(response, 'Keyword')
        self.assertEqual(len(response.context['reviews']), 6)
        self.assertEqual(response.context['match_count'], 6)
        self.assertIsInstance(response.context['filter_form'], ReviewFilterForm)

    def test_newest_reviews_are_listed_first(self):
        reviews = list(self.get_page().context['reviews'])
        self.assertEqual(reviews[0], self.empty)
        self.assertEqual(reviews[-1], self.pos5)

    def test_search_matches_review_text(self):
        response = self.get_page({'q': 'quality'})
        self.assertEqual(
            [review.pk for review in response.context['reviews']],
            [self.neg2.pk],
        )
        self.assertEqual(response.context['match_count'], 1)

    def test_search_matches_summary(self):
        extra = Review.objects.create(
            text='Body of the review mentions nothing special',
            summary='Summary contains the secretword',
            rating=4,
            sentiment=sentiment_from_rating(4),
        )
        response = self.get_page({'q': 'secretword'})
        self.assertEqual([review.pk for review in response.context['reviews']], [extra.pk])

    def test_search_input_is_escaped(self):
        response = self.get_page({'q': '<script>alert(9)</script>'})
        content = response.content.decode('utf-8')
        self.assertIn('&lt;script&gt;alert(9)&lt;/script&gt;', content)
        self.assertNotIn('<script>alert(9)</script>', content)

    def test_overlong_search_is_rejected_without_applying_filters(self):
        response = self.get_page({'q': 'a' * 101})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Search terms cannot exceed 100 characters.')
        self.assertEqual(len(response.context['reviews']), 6)
        self.assertFalse(response.context['filters_active'])

    def test_rating_filter(self):
        response = self.get_page({'rating': '1'})
        self.assertEqual(
            [review.pk for review in response.context['reviews']], [self.neg1.pk]
        )

    def test_missing_rating_filter(self):
        response = self.get_page({'rating': 'none'})
        self.assertEqual(
            [review.pk for review in response.context['reviews']], [self.empty.pk]
        )

    def test_sentiment_filter(self):
        response = self.get_page({'sentiment': 'negative'})
        self.assertEqual(
            {review.pk for review in response.context['reviews']},
            {self.neg1.pk, self.neg2.pk},
        )

    def test_missing_sentiment_filter(self):
        response = self.get_page({'sentiment': 'none'})
        self.assertEqual(
            [review.pk for review in response.context['reviews']], [self.empty.pk]
        )

    def test_invalid_rating_shows_error_and_keeps_all_rows(self):
        response = self.get_page({'rating': '9'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Select a valid rating.')
        self.assertEqual(len(response.context['reviews']), 6)

    def test_invalid_sentiment_shows_error(self):
        response = self.get_page({'sentiment': 'Angry'})
        self.assertContains(response, 'Select a valid sentiment.')
        self.assertEqual(len(response.context['reviews']), 6)

    def test_date_range_filter(self):
        response = self.get_page({'start_date': '2024-06-01', 'end_date': '2024-06-30'})
        pks = {review.pk for review in response.context['reviews']}
        self.assertEqual(pks, {self.neu3.pk, self.neg1.pk, self.empty.pk})

    def test_search_and_sentiment_filters_combine(self):
        response = self.get_page({'q': 'bad', 'sentiment': 'negative'})
        self.assertEqual(
            [review.pk for review in response.context['reviews']], [self.neg1.pk]
        )

    def test_active_filter_is_flagged(self):
        response = self.get_page({'q': 'quality'})
        self.assertTrue(response.context['filters_active'])
        self.assertContains(response, 'Filters active')

    @override_settings(REVIEW_PAGE_SIZE=5)
    def test_results_are_paginated(self):
        first = self.get_page()
        self.assertEqual(len(first.context['reviews']), 5)
        self.assertEqual(first.context['match_count'], 6)
        self.assertTrue(first.context['page_obj'].has_next())
        self.assertFalse(first.context['page_obj'].has_previous())

        second = self.get_page({'page': 2})
        self.assertEqual(len(second.context['reviews']), 1)
        self.assertTrue(second.context['page_obj'].has_previous())
        self.assertEqual(second.context['page_obj'].number, 2)

    @override_settings(REVIEW_PAGE_SIZE=5)
    def test_pagination_links_preserve_filters(self):
        self.seed_searchable_reviews()
        response = self.get_page({'q': 'review body', 'page': 1})
        self.assertEqual(response.context['match_count'], 6)
        self.assertEqual(len(response.context['reviews']), 5)
        self.assertContains(response, '?q=review+body&amp;page=2')

    @override_settings(REVIEW_PAGE_SIZE=5)
    def test_first_page_has_no_previous_link(self):
        self.seed_searchable_reviews()
        response = self.get_page({'q': 'review body'})
        self.assertContains(response, 'page=2')
        self.assertNotContains(response, '>Previous<')
        self.assertNotContains(response, 'page=0')

    def test_invalid_page_parameter_falls_back_to_first_page(self):
        response = self.get_page({'page': 'not-a-number'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['page_obj'].number, 1)
        self.assertEqual(len(response.context['reviews']), 6)

    def test_empty_result_message(self):
        response = self.get_page({'q': 'no-such-phrase-zzz'})
        self.assertContains(response, 'No reviews match the current search and filters.')
        self.assertEqual(len(response.context['reviews']), 0)
        self.assertEqual(response.context['match_count'], 0)

    def test_long_review_text_is_collapsed(self):
        long_text = 'Long review text ' * 40
        self.empty.text = long_text
        self.empty.save()
        response = self.get_page()
        self.assertContains(response, 'Read full review')
        self.assertContains(response, long_text)

    def test_missing_rating_and_date_are_shown(self):
        response = self.get_page()
        self.assertContains(response, 'Not rated')
        self.assertContains(response, '&mdash;')


class NegativeFeedbackTests(DashboardDatasetMixin, TestCase):
    def get_page(self, params=None):
        return self.client.get(reverse('negative_feedback'), params or {})

    def test_page_renders_with_negative_counts(self):
        response = self.get_page()
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Negative Feedback Analysis')
        self.assertEqual(response.context['negative_count'], 2)
        self.assertEqual(response.context['base'], 5)
        self.assertEqual(response.context['scope_total'], 6)
        self.assertEqual(response.context['share'], 40.0)
        self.assertEqual(response.context['share_display'], '40.0%')
        self.assertIsInstance(response.context['filter_form'], NegativeFilterForm)

    def test_average_rating_of_negative_reviews(self):
        context = self.get_page().context
        self.assertEqual(context['avg_rating'], 1.5)
        self.assertEqual(context['avg_rating_display'], '1.50')

    def test_rating_breakdown_counts_and_shares(self):
        rows = self.get_page().context['rating_breakdown']
        self.assertEqual([row['rating'] for row in rows], [1, 2, 3, 4, 5])
        self.assertEqual([row['count'] for row in rows], [1, 1, 0, 0, 0])
        self.assertEqual(rows[0]['share'], 50.0)
        self.assertEqual(rows[0]['share_display'], '50.0%')
        self.assertEqual(rows[2]['share'], 0.0)

    def test_most_common_rating_is_reported(self):
        top = self.get_page().context['top_rating']
        self.assertEqual(top['rating'], 1)
        self.assertEqual(top['count'], 1)

    def test_insights_state_counts_and_denominators(self):
        joined = ' '.join(self.get_page().context['insights'])
        self.assertIn(
            '2 of 5 reviews with rating-derived sentiment are Negative (40.0%)',
            joined,
        )
        self.assertIn('n = 2', joined)
        self.assertIn('1.50 out of 5', joined)

    def test_insights_avoid_causal_claims(self):
        joined = ' '.join(self.get_page().context['insights']).lower()
        for forbidden in ('because', 'caused by', 'customers prefer', 'trend shows'):
            self.assertNotIn(forbidden, joined)

    def test_frequent_terms_use_review_text_without_stop_words(self):
        terms = self.get_page().context['terms']
        names = [item['term'] for item in terms['terms']]
        self.assertIn('quality', names)
        self.assertIn('disappointed', names)
        for stop in ('the', 'with', 'very', 'and'):
            self.assertNotIn(stop, names)

    def test_term_document_frequency_and_sample(self):
        terms = self.get_page().context['terms']
        quality = next(item for item in terms['terms'] if item['term'] == 'quality')
        self.assertEqual(quality['documents'], 1)
        self.assertEqual(quality['documents_pct'], 50.0)
        self.assertEqual(terms['sampled'], 2)
        self.assertEqual(terms['total'], 2)
        self.assertFalse(terms['truncated'])

    def test_term_caveat_is_shown(self):
        response = self.get_page()
        self.assertContains(response, 'does not show what caused')
        self.assertContains(response, 'Frequent terms in negative review text')

    def test_representative_reviews_are_listed(self):
        response = self.get_page()
        self.assertEqual(len(response.context['representative']), 2)
        self.assertContains(response, 'Broke immediately, very bad')
        self.assertContains(response, 'Disappointed with the quality')
        self.assertContains(response, 'Representative negative reviews')

    def test_monthly_volume_only_counts_dated_negatives(self):
        monthly = self.get_page().context['monthly']
        self.assertEqual(monthly['labels'], ['2024-06'])
        self.assertEqual(monthly['data'], [1])

    def test_prediction_history_is_never_used(self):
        PredictionHistory.objects.create(
            text='ML ONLY SECRET RECORD', predicted_sentiment='Negative'
        )
        response = self.get_page()
        self.assertEqual(response.context['negative_count'], 2)
        self.assertNotContains(response, 'ML ONLY SECRET RECORD')

    def test_keyword_filter_narrows_the_analysis(self):
        response = self.get_page({'q': 'quality'})
        self.assertEqual(response.context['negative_count'], 1)
        self.assertEqual(response.context['base'], 1)
        self.assertEqual(response.context['share'], 100.0)
        self.assertContains(response, 'Filters active')

    def test_date_range_filter_narrows_the_analysis(self):
        response = self.get_page(
            {'start_date': '2024-06-01', 'end_date': '2024-06-30'}
        )
        context = response.context
        self.assertEqual(context['scope_total'], 3)
        self.assertEqual(context['negative_count'], 1)
        self.assertEqual(context['base'], 2)

    def test_invalid_date_shows_error_and_unfiltered_analysis(self):
        response = self.get_page({'start_date': 'not-a-date'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Enter a valid start date as YYYY-MM-DD.')
        self.assertEqual(response.context['negative_count'], 2)
        self.assertFalse(response.context['filters_active'])

    def test_search_and_date_order_are_validated(self):
        response = self.get_page({'start_date': '2024-07-01', 'end_date': '2024-06-01'})
        self.assertContains(response, 'The start date must be on or before the end date.')
        self.assertEqual(response.context['negative_count'], 2)

    def test_analysis_links_to_other_pages(self):
        response = self.get_page()
        self.assertContains(response, f'href="{reverse("home")}"')
        self.assertContains(response, f'href="{reverse("reviews")}"')


class NegativeFeedbackEmptyStateTests(TestCase):
    def test_empty_database_renders_zero_state(self):
        response = self.client.get(reverse('negative_feedback'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['negative_count'], 0)
        self.assertEqual(response.context['scope_total'], 0)
        self.assertEqual(response.context['avg_rating'], None)
        self.assertEqual(response.context['avg_rating_display'], '\u2014')
        self.assertTrue(
            any('cannot be calculated' in item for item in response.context['insights'])
        )
        self.assertContains(response, 'No reviews match the selected filters')

    def test_no_negative_reviews_message(self):
        Review.objects.create(text='All fine', rating=5, sentiment=sentiment_from_rating(5))
        response = self.client.get(reverse('negative_feedback'))
        self.assertEqual(response.context['negative_count'], 0)
        self.assertContains(response, 'No rating-derived Negative reviews match')
        self.assertContains(response, 'Not enough negative review text')


class NavigationTests(DashboardDatasetMixin, TestCase):
    def test_sidebar_links_to_both_new_pages(self):
        response = self.client.get(reverse('home'))
        self.assertContains(response, f'href="{reverse("reviews")}"')
        self.assertContains(response, f'href="{reverse("negative_feedback")}"')
        self.assertContains(response, 'Review Explorer')
        self.assertContains(response, 'Negative Feedback')

    def test_new_pages_link_back_to_existing_pages(self):
        for name in ('reviews', 'negative_feedback'):
            response = self.client.get(reverse(name))
            self.assertContains(response, f'href="{reverse("home")}"')
            self.assertContains(response, f'href="{reverse("upload_reviews")}"')
            self.assertContains(response, f'href="{reverse("predict_review")}"')

    def test_all_pages_render_with_status_200(self):
        for name in ('home', 'upload_reviews', 'predict_review', 'reviews', 'negative_feedback'):
            response = self.client.get(reverse(name))
            self.assertEqual(response.status_code, 200, name)


class QueryHelperTests(TestCase):
    def test_apply_search_without_term_returns_queryset(self):
        queryset = Review.objects.all()
        self.assertIs(queries.apply_search(queryset, ''), queryset)
        self.assertIs(queries.apply_search(queryset, '   '), queryset)

    def test_apply_filters_without_values_return_queryset(self):
        queryset = Review.objects.all()
        self.assertIs(queries.apply_rating_filter(queryset, ''), queryset)
        self.assertIs(queries.apply_sentiment_filter(queryset, ''), queryset)

    def test_invalid_filter_values_are_ignored(self):
        queryset = Review.objects.all()
        self.assertIs(queries.apply_rating_filter(queryset, '9'), queryset)
        self.assertIs(queries.apply_sentiment_filter(queryset, 'Angry'), queryset)

    def test_build_review_queryset_combines_filters(self):
        Review.objects.create(
            text='Broken item', rating=1, sentiment=sentiment_from_rating(1)
        )
        Review.objects.create(
            text='Great item', rating=5, sentiment=sentiment_from_rating(5)
        )
        queryset = queries.build_review_queryset(
            q='item', rating='1', sentiment='negative'
        )
        self.assertEqual(queryset.count(), 1)
        self.assertEqual(queryset.first().text, 'Broken item')

    def test_querystring_without_drops_only_page(self):
        from django.http import QueryDict

        params = QueryDict('q=foo&rating=1&page=3')
        self.assertEqual(
            queries.querystring_without(params, exclude=('page',)), 'q=foo&rating=1'
        )
        self.assertEqual(
            queries.querystring_without(params, exclude=()), 'q=foo&rating=1&page=3'
        )


class FrequentTermsTests(TestCase):
    def make_negatives(self, count, text):
        for index in range(count):
            Review.objects.create(
                text=f'{text} variant {index}',
                rating=1,
                sentiment=sentiment_from_rating(1),
            )

    def test_sample_limit_marks_truncated_result(self):
        self.make_negatives(3, 'alpha bravo')
        result = negative_analysis.frequent_terms(
            Review.objects.filter(sentiment='Negative'), sample_limit=2
        )
        self.assertEqual(result['sampled'], 2)
        self.assertEqual(result['total'], 3)
        self.assertTrue(result['truncated'])

    def test_full_sample_is_not_marked_truncated(self):
        self.make_negatives(2, 'alpha bravo')
        result = negative_analysis.frequent_terms(
            Review.objects.filter(sentiment='Negative'), sample_limit=10
        )
        self.assertFalse(result['truncated'])
        self.assertEqual(result['sampled'], 2)

    def test_term_limit_is_respected(self):
        for text in (
            'alpha alpha alpha bravo',
            'bravo charlie',
            'delta echo',
            'foxtrot golf',
            'hotel india',
        ):
            Review.objects.create(
                text=text, rating=1, sentiment=sentiment_from_rating(1)
            )
        result = negative_analysis.frequent_terms(
            Review.objects.filter(sentiment='Negative'), limit=3
        )
        self.assertEqual(len(result['terms']), 3)
        self.assertEqual(result['terms'][0]['term'], 'alpha')
        self.assertEqual(result['terms'][0]['count'], 3)
        self.assertEqual(result['terms'][1]['term'], 'bravo')
        self.assertEqual(result['terms'][1]['count'], 2)

    def test_empty_queryset_returns_no_terms(self):
        result = negative_analysis.frequent_terms(Review.objects.none())
        self.assertEqual(result['terms'], [])
        self.assertEqual(result['sampled'], 0)
        self.assertFalse(result['truncated'])


class CsvSanitizeTests(TestCase):
    def test_formula_triggers_are_prefixed(self):
        for value in ('=1+1', '+SUM(A1)', '-danger', '@cmd', '  =indented', '\ttabbed'):
            self.assertEqual(csv_export.sanitize_cell(value), "'" + value, value)

    def test_plain_text_and_numbers_are_unchanged(self):
        self.assertEqual(csv_export.sanitize_cell('Great product'), 'Great product')
        self.assertEqual(csv_export.sanitize_cell('2024-06-15'), '2024-06-15')
        self.assertEqual(csv_export.sanitize_cell('2024-06-15 10:00:00'), '2024-06-15 10:00:00')
        self.assertEqual(csv_export.sanitize_cell(5), 5)
        self.assertEqual(csv_export.sanitize_cell(40.0), 40.0)
        self.assertEqual(csv_export.sanitize_cell(None), '')

    def test_iter_csv_emits_sanitised_header_and_rows(self):
        lines = list(csv_export.iter_csv(['a', 'b'], [['=x', 'ok']]))
        self.assertEqual(lines[0].strip(), 'a,b')
        self.assertEqual(lines[1].strip(), "'=x,ok")

    def test_format_datetime(self):
        naive = datetime(2024, 6, 15, 8, 30, 0)
        self.assertEqual(csv_export.format_datetime(naive), '2024-06-15 08:30:00')
        self.assertEqual(csv_export.format_datetime(None), '')

    def test_export_filename_format(self):
        self.assertRegex(
            csv_export.export_filename('reviews'), r'^reviews-\d{8}-\d{6}\.csv$'
        )
        self.assertRegex(
            csv_export.export_filename('analytics-summary'),
            r'^analytics-summary-\d{8}-\d{6}\.csv$',
        )


class ReviewExportTests(DashboardDatasetMixin, TestCase):
    def export(self, params=None):
        return self.client.get(reverse('export_reviews'), params or {})

    def test_streams_csv_with_content_type_and_filename(self):
        response = self.export()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.streaming)
        self.assertTrue(response['Content-Type'].startswith('text/csv'))
        self.assertRegex(
            response['Content-Disposition'],
            r'^attachment; filename="reviews-\d{8}-\d{6}\.csv"$',
        )

    def test_header_row_lists_exported_fields(self):
        rows = export_rows(self.export())
        self.assertEqual(
            rows[0],
            [
                'review_text',
                'summary',
                'product_id',
                'rating',
                'sentiment',
                'review_date',
                'imported_at',
            ],
        )
        self.assertEqual(len(rows), 7)

    def test_rows_carry_review_fields(self):
        rows = export_rows(self.export())
        row = next(r for r in rows[1:] if r[0] == self.neg1.text)
        self.assertEqual(row[1], '')
        self.assertEqual(row[2], '')
        self.assertEqual(row[3], '1')
        self.assertEqual(row[4], 'Negative')
        self.assertTrue(row[5].startswith('2024-06-15'))
        self.assertRegex(row[6], r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$')

    def test_unrated_review_exports_empty_rating_and_sentiment(self):
        rows = export_rows(self.export())
        row = next(r for r in rows[1:] if r[0] == self.empty.text)
        self.assertEqual(row[3], '')
        self.assertEqual(row[4], '')
        self.assertEqual(row[2], '')

    def test_search_and_rating_filters_are_applied(self):
        rows = export_rows(self.export({'q': 'bad', 'rating': '1'}))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][0], self.neg1.text)

    def test_date_range_filters_are_applied(self):
        rows = export_rows(
            self.export({'start_date': '2024-06-01', 'end_date': '2024-06-30'})
        )
        self.assertEqual(len(rows), 4)
        exported = {row[0] for row in rows[1:]}
        self.assertEqual(
            exported, {self.neu3.text, self.neg1.text, self.empty.text}
        )

    def test_sentiment_filter_is_applied(self):
        rows = export_rows(self.export({'sentiment': 'Negative'}))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row[4] for row in rows[1:]}, {'Negative'})

    def test_empty_result_returns_header_only(self):
        response = self.export({'q': 'no-such-phrase-zzz'})
        rows = export_rows(response)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 'review_text')

    def test_invalid_filter_falls_back_to_every_review(self):
        rows = export_rows(self.export({'rating': '9'}))
        self.assertEqual(len(rows), 7)

    def test_formula_injection_is_neutralised(self):
        Review.objects.all().delete()
        payloads = ['=SUM(A1:A9)', '+1+1', '-danger', '@cmd', '  =indented', '\ttabbed']
        for payload in payloads:
            Review.objects.create(
                text=payload,
                summary=payload,
                product_id='p1',
                rating=5,
                sentiment=sentiment_from_rating(5),
            )

        rows = export_rows(self.export())
        self.assertEqual(len(rows), len(payloads) + 1)
        self.assertEqual(
            [row[0] for row in rows[1:]], ["'" + p for p in reversed(payloads)]
        )
        for row in rows[1:]:
            self.assertEqual(row[1], "'" + row[0][1:])

    def test_prediction_history_is_never_exported(self):
        PredictionHistory.objects.create(
            text='ML ONLY SECRET RECORD', predicted_sentiment='Negative'
        )
        response = self.export()
        self.assertNotContains(response, 'ML ONLY SECRET RECORD')
        self.assertEqual(len(export_rows(response)), 7)

    def test_explorer_offers_export_with_active_filters(self):
        response = self.client.get(reverse('reviews'), {'q': 'bad', 'rating': '1'})
        self.assertContains(
            response,
            f'href="{reverse("export_reviews")}?q=bad&amp;rating=1"',
        )
        self.assertContains(response, 'Export CSV')
        self.assertEqual(len(export_rows(self.export({'q': 'bad', 'rating': '1'}))), 2)

    def test_invalid_filter_page_offers_unfiltered_export(self):
        response = self.client.get(reverse('reviews'), {'rating': '9'})
        self.assertContains(
            response, f'href="{reverse("export_reviews")}?rating=9"'
        )


class AnalyticsExportTests(DashboardDatasetMixin, TestCase):
    def export(self, params=None):
        return self.client.get(reverse('export_analytics'), params or {})

    def metrics(self, params=None):
        rows = export_rows(self.export(params))
        self.assertEqual(rows[0], ['section', 'metric', 'value', 'definition'])
        return rows, {row[1]: row for row in rows[1:]}

    def test_content_type_and_filename(self):
        response = self.export()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.streaming)
        self.assertTrue(response['Content-Type'].startswith('text/csv'))
        self.assertRegex(
            response['Content-Disposition'],
            r'^attachment; filename="analytics-summary-\d{8}-\d{6}\.csv"$',
        )

    def test_sections_cover_scope_kpis_and_notes(self):
        rows, table = self.metrics()
        sections = {row[0] for row in rows[1:]}
        self.assertEqual(sections, {'scope', 'kpi', 'note'})
        self.assertIn('Total reviews', table)
        self.assertIn('Export scope', table)

    def test_kpi_values_and_denominators(self):
        _rows, table = self.metrics()
        self.assertEqual(table['Total reviews'][2], '6')
        self.assertEqual(table['Positive reviews'][2], '2')
        self.assertEqual(table['Neutral reviews'][2], '1')
        self.assertEqual(table['Negative reviews'][2], '2')
        self.assertEqual(table['Reviews with a valid rating'][2], '5')
        self.assertEqual(table['Missing rating reviews'][2], '1')
        self.assertEqual(table['Average rating'][2], '3.00')

        self.assertEqual(
            table['Reviews with rating-derived sentiment'][2], '5'
        )
        self.assertIn('Denominator', table['Reviews with rating-derived sentiment'][3])
        self.assertEqual(table['Positive share (%)'][2], '40.0')
        self.assertEqual(table['Neutral share (%)'][2], '20.0')
        self.assertEqual(table['Negative share (%)'][2], '40.0')
        self.assertIn('x 100', table['Negative share (%)'][3])
        self.assertIn('/ reviews with rating-derived sentiment', table['Negative share (%)'][3])
        self.assertEqual(table['Missing rating share (%)'][2], '16.7')
        self.assertIn('/ total reviews in scope x 100', table['Missing rating share (%)'][3])

    def test_unfiltered_scope_reports_dates_as_empty(self):
        _rows, table = self.metrics()
        self.assertEqual(table['Start date'][2], '')
        self.assertEqual(table['End date'][2], '')
        self.assertEqual(table['Date filter active'][2], 'no')
        self.assertEqual(table['Reviews in scope'][2], '6')
        self.assertEqual(table['Reviews in database'][2], '6')
        self.assertNotEqual(table['Generated (UTC)'][2], '')

    def test_date_filter_values_and_recalculated_kpis(self):
        params = {'start_date': '2024-06-01', 'end_date': '2024-06-30'}
        _rows, table = self.metrics(params)
        self.assertEqual(table['Start date'][2], '2024-06-01')
        self.assertEqual(table['End date'][2], '2024-06-30')
        self.assertEqual(table['Date filter active'][2], 'yes')
        self.assertEqual(table['Reviews in scope'][2], '3')
        self.assertEqual(table['Reviews in database'][2], '6')
        self.assertEqual(table['Total reviews'][2], '3')
        self.assertEqual(table['Positive reviews'][2], '0')
        self.assertEqual(table['Neutral reviews'][2], '1')
        self.assertEqual(table['Negative reviews'][2], '1')
        self.assertEqual(table['Reviews with rating-derived sentiment'][2], '2')
        self.assertEqual(table['Negative share (%)'][2], '50.0')
        self.assertEqual(table['Neutral share (%)'][2], '50.0')
        self.assertEqual(table['Missing rating reviews'][2], '1')
        self.assertEqual(table['Average rating'][2], '2.00')

    def test_invalid_date_filter_is_ignored_like_the_dashboard(self):
        _rows, table = self.metrics({'start_date': 'not-a-date'})
        self.assertEqual(table['Total reviews'][2], '6')
        self.assertEqual(table['Date filter active'][2], 'no')

    def test_scope_note_excludes_charts_and_ml_predictions(self):
        _rows, table = self.metrics()
        note = table['Export scope']
        self.assertIn('no chart series', note[3])
        self.assertIn('PredictionHistory', note[3])

    def test_prediction_history_does_not_change_kpis(self):
        PredictionHistory.objects.create(
            text='ML ONLY SECRET RECORD', predicted_sentiment='Negative'
        )
        response = self.export()
        self.assertNotContains(response, 'ML ONLY SECRET RECORD')
        rows = export_rows(response)
        table = {row[1]: row for row in rows[1:]}
        self.assertEqual(table['Total reviews'][2], '6')

    def test_home_page_links_to_the_summary_export(self):
        response = self.client.get(reverse('home'))
        self.assertContains(response, f'href="{reverse("export_analytics")}"')
        self.assertContains(response, 'Export summary CSV')

    def test_home_page_export_preserves_the_date_filter(self):
        response = self.client.get(
            reverse('home'), {'start_date': '2024-06-01', 'end_date': '2024-06-30'}
        )
        self.assertContains(
            response,
            f'href="{reverse("export_analytics")}'
            '?start_date=2024-06-01&amp;end_date=2024-06-30"',
        )


class MessageRenderingTests(TestCase):
    def test_success_message_renders_once_with_bootstrap_classes(self):
        response = post_upload(self.client, 'Text\nHello there\n')
        self.assertContains(response, 'alert-success', count=1)
        self.assertContains(response, 'Imported 1 review(s)', count=1)

    def test_error_message_renders_once_as_danger(self):
        response = post_upload(self.client, 'Score,ProductId\n5,B0001\n')
        self.assertContains(response, 'alert-danger', count=1)
        self.assertContains(response, 'Missing required column', count=1)

    def test_messages_area_uses_a_live_region(self):
        response = post_upload(self.client, 'Text\nHello there\n')
        self.assertContains(response, 'aria-live="polite"', count=1)


class UiPolishTests(DashboardDatasetMixin, TestCase):
    def test_navigation_separates_the_four_areas(self):
        home = self.client.get(reverse('home'))
        for caption in ('Analytics', 'Data &amp; ML', 'Review Explorer', 'Negative Feedback'):
            self.assertContains(home, caption, msg_prefix=caption)
        self.assertContains(home, f'href="{reverse("export_analytics")}"')

        reviews = self.client.get(reverse('reviews'))
        self.assertContains(reviews, f'href="{reverse("export_reviews")}"')
        self.assertContains(reviews, 'Export CSV')

    def test_page_headers_are_present(self):
        for name in ('home', 'reviews', 'negative_feedback', 'upload_reviews', 'predict_review'):
            response = self.client.get(reverse(name))
            self.assertContains(response, 'page-header', msg_prefix=name)

    def test_filter_forms_are_labelled_and_searchable(self):
        response = self.client.get(reverse('reviews'))
        self.assertContains(response, 'role="search"')
        self.assertContains(response, 'aria-label="Review filters"')
        self.assertContains(response, 'for="id_q"')
        self.assertContains(response, 'for="id_rating"')

    def test_pagination_is_labelled_for_screen_readers(self):
        for index in range(25):
            Review.objects.create(
                text=f'Bulk review {index}',
                rating=3,
                sentiment=sentiment_from_rating(3),
            )
        response = self.client.get(reverse('reviews'))
        self.assertContains(response, 'aria-label="Review pages"')
        self.assertContains(response, 'aria-label="Primary"')


class SettingsEnvironmentTests(SimpleTestCase):
    def test_environment_overrides_are_read_when_settings_load(self):
        import importlib

        import config.settings as settings_module

        keys = ('DJANGO_DEBUG', 'DJANGO_ALLOWED_HOSTS', 'DJANGO_SECRET_KEY')
        original = {key: os.environ.get(key) for key in keys}
        try:
            os.environ['DJANGO_DEBUG'] = '0'
            os.environ['DJANGO_ALLOWED_HOSTS'] = 'demo.local, test.local'
            os.environ['DJANGO_SECRET_KEY'] = 'env-provided-key'
            importlib.reload(settings_module)
            self.assertFalse(settings_module.DEBUG)
            self.assertEqual(settings_module.ALLOWED_HOSTS, ['demo.local', 'test.local'])
            self.assertEqual(settings_module.SECRET_KEY, 'env-provided-key')

            os.environ['DJANGO_DEBUG'] = 'true'
            os.environ.pop('DJANGO_ALLOWED_HOSTS')
            os.environ.pop('DJANGO_SECRET_KEY')
            importlib.reload(settings_module)
            self.assertTrue(settings_module.DEBUG)
            self.assertEqual(settings_module.ALLOWED_HOSTS, [])
            self.assertTrue(settings_module.SECRET_KEY.startswith('django-insecure-'))
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            importlib.reload(settings_module)

    def test_active_settings_keep_local_demo_defaults(self):
        import config.settings as settings_module

        # `runserver` defaults: debug on, empty host list, insecure dev key.
        # (django.conf.settings.DEBUG is False here because the test runner
        # disables debug for the duration of the suite.)
        self.assertTrue(settings_module.DEBUG)
        self.assertEqual(settings_module.ALLOWED_HOSTS, [])
        self.assertTrue(settings_module.SECRET_KEY.startswith('django-insecure-'))
        self.assertEqual(settings.REVIEW_PAGE_SIZE, 20)
