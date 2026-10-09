import hashlib

from django.db import models


def make_content_hash(text, product_id):
    """Stable hash used to detect duplicate reviews."""
    payload = f'{(product_id or "").strip()}\n{text or ""}'.encode('utf-8', 'replace')
    return hashlib.sha256(payload).hexdigest()


class Sentiment(models.TextChoices):
    NEGATIVE = 'Negative', 'Negative'
    NEUTRAL = 'Neutral', 'Neutral'
    POSITIVE = 'Positive', 'Positive'
    BLANK = '', 'Unrated'


def sentiment_from_rating(rating):
    """Map a 1-5 rating to a rating-derived sentiment.

    Returns ``None`` when the rating is missing, not a number, or outside the
    1-5 range. No sentiment is ever guessed for unrated reviews.
    """
    if rating is None or isinstance(rating, bool):
        return None
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float('inf'), float('-inf')):
        return None
    if value != int(value):
        return None
    number = int(value)
    if number in (1, 2):
        return Sentiment.NEGATIVE
    if number == 3:
        return Sentiment.NEUTRAL
    if number in (4, 5):
        return Sentiment.POSITIVE
    return None


class Review(models.Model):
    text = models.TextField()
    summary = models.TextField(blank=True, default='')
    rating = models.IntegerField(null=True, blank=True)
    product_id = models.CharField(max_length=64, blank=True, default='')
    review_date = models.DateTimeField(null=True, blank=True)
    sentiment = models.CharField(
        max_length=16,
        choices=Sentiment.choices,
        blank=True,
        null=True,
        help_text='Derived from the rating only; empty when unrated.',
    )
    content_hash = models.CharField(
        max_length=64,
        db_index=True,
        editable=False,
        help_text='SHA-256 of product id and review text, used for duplicate detection.',
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            # Measured on the real dataset (20,407 rows): the explorer's
            # ORDER BY created_at DESC went from ~78 ms to ~1 ms and a
            # date-filtered KPI pass from ~168 ms to ~35 ms.
            models.Index(fields=['-created_at'], name='review_created_at_idx'),
            models.Index(fields=['review_date'], name='review_review_date_idx'),
        ]

    def __str__(self):
        label = self.sentiment or 'Unrated'
        return f'{label} review ({self.product_id or "no product"})'

    def save(self, *args, **kwargs):
        if not self.content_hash:
            self.content_hash = make_content_hash(self.text, self.product_id)
        super().save(*args, **kwargs)

    @property
    def rating_display(self):
        return self.rating if self.rating is not None else '—'


class PredictionHistory(models.Model):
    """A single review submitted to the ML model, with its predicted label."""

    text = models.TextField()
    predicted_sentiment = models.CharField(
        max_length=16,
        choices=[choice for choice in Sentiment.choices if choice[0]],
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # ``-id`` keeps the order deterministic when timestamps collide
        # (Windows clock granularity can make timestamps identical).
        ordering = ['-created_at', '-id']

    def __str__(self):
        return f'{self.predicted_sentiment}: {self.text[:40]}'

    @property
    def short_text(self):
        return self.text if len(self.text) <= 80 else f'{self.text[:77]}...'
