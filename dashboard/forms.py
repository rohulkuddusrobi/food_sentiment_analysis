from django import forms


class ReviewCSVUploadForm(forms.Form):
    file = forms.FileField(
        label='CSV file',
        help_text='Required column: Text. Optional columns: Score, ProductId, Time, Summary.',
        widget=forms.FileInput(
            attrs={
                'class': 'form-control',
                'accept': '.csv,text/csv,text/plain',
            }
        ),
    )

    def clean_file(self):
        upload = self.cleaned_data['file']
        name = (getattr(upload, 'name', '') or '').lower()
        if not name.endswith('.csv'):
            raise forms.ValidationError('Please choose a file with a .csv extension.')
        if upload.size == 0:
            raise forms.ValidationError('The uploaded file is empty.')
        return upload


class PredictionForm(forms.Form):
    MAX_LENGTH = 10000

    text = forms.CharField(
        label='Review text',
        max_length=MAX_LENGTH,
        strip=True,
        widget=forms.Textarea(
            attrs={
                'class': 'form-control',
                'rows': 5,
                'placeholder': 'Paste a customer review here...',
            }
        ),
        error_messages={
            'required': 'Enter some review text to predict a sentiment.',
            'max_length': f'Review text cannot exceed {MAX_LENGTH} characters.',
        },
    )

    def clean_text(self):
        value = self.cleaned_data['text'].strip()
        if not value:
            raise forms.ValidationError('Enter some review text to predict a sentiment.')
        return value


class DateRangeFilterForm(forms.Form):
    """Shared optional start/end date filter used by every analytics page."""

    DATE_INPUT_FORMATS = ['%Y-%m-%d']

    start_date = forms.DateField(
        required=False,
        label='From',
        input_formats=DATE_INPUT_FORMATS,
        widget=forms.DateInput(
            attrs={
                'type': 'date',
                'class': 'form-control form-control-sm',
                'placeholder': 'YYYY-MM-DD',
            }
        ),
        error_messages={'invalid': 'Enter a valid start date as YYYY-MM-DD.'},
    )
    end_date = forms.DateField(
        required=False,
        label='To',
        input_formats=DATE_INPUT_FORMATS,
        widget=forms.DateInput(
            attrs={
                'type': 'date',
                'class': 'form-control form-control-sm',
                'placeholder': 'YYYY-MM-DD',
            }
        ),
        error_messages={'invalid': 'Enter a valid end date as YYYY-MM-DD.'},
    )

    def clean(self):
        cleaned = super().clean()
        start_date = cleaned.get('start_date')
        end_date = cleaned.get('end_date')
        if start_date and end_date and start_date > end_date:
            raise forms.ValidationError(
                'The start date must be on or before the end date.'
            )
        return cleaned


class DashboardFilterForm(DateRangeFilterForm):
    """Date-only filter for the home dashboard (Phase 4 behaviour)."""


class SearchFilterForm(DateRangeFilterForm):
    """Date range plus a keyword search over review text and summary."""

    q = forms.CharField(
        required=False,
        label='Search',
        max_length=100,
        strip=True,
        widget=forms.TextInput(
            attrs={
                'class': 'form-control form-control-sm',
                'placeholder': 'Search text or summary...',
            }
        ),
        error_messages={'max_length': 'Search terms cannot exceed 100 characters.'},
    )


RATING_FILTER_CHOICES = [
    ('', 'Any rating'),
    ('5', '5 stars'),
    ('4', '4 stars'),
    ('3', '3 stars'),
    ('2', '2 stars'),
    ('1', '1 star'),
    ('none', 'Missing rating'),
]

SENTIMENT_FILTER_CHOICES = [
    ('', 'Any sentiment'),
    ('Positive', 'Positive'),
    ('Neutral', 'Neutral'),
    ('Negative', 'Negative'),
    ('none', 'Unavailable'),
]


class NormalizedChoiceField(forms.ChoiceField):
    """ChoiceField that trims whitespace and matches values case-insensitively.

    Lets a hand-typed query string such as ``?sentiment=negative`` resolve to
    the canonical ``Negative`` choice while still rejecting unknown values.
    """

    def to_python(self, value):
        value = super().to_python(value)
        if value in self.empty_values:
            return ''
        text = str(value).strip()
        for key, _label in self.choices:
            if str(key).lower() == text.lower():
                return key
        return text


class ReviewFilterForm(SearchFilterForm):
    """Keyword, rating, sentiment and date filters for the Review Explorer."""

    rating = NormalizedChoiceField(
        required=False,
        label='Rating',
        choices=RATING_FILTER_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select form-select-sm'}),
        error_messages={'invalid_choice': 'Select a valid rating.'},
    )
    sentiment = NormalizedChoiceField(
        required=False,
        label='Sentiment',
        choices=SENTIMENT_FILTER_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select form-select-sm'}),
        error_messages={'invalid_choice': 'Select a valid sentiment.'},
    )


class NegativeFilterForm(SearchFilterForm):
    """Keyword and date filters for the Negative Feedback analysis page."""
