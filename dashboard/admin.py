from django.contrib import admin

from .models import PredictionHistory, Review


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('id', 'product_id', 'rating', 'sentiment', 'review_date', 'created_at')
    list_filter = ('sentiment', 'rating')
    search_fields = ('text', 'summary', 'product_id')
    readonly_fields = ('content_hash', 'created_at', 'updated_at')


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'predicted_sentiment', 'short_review', 'created_at')
    list_filter = ('predicted_sentiment',)
    search_fields = ('text',)
    readonly_fields = ('created_at',)

    @admin.display(description='Review')
    def short_review(self, obj):
        return obj.short_text
