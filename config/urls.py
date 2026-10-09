from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from dashboard import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('upload/', views.upload_reviews, name='upload_reviews'),
    path('predict/', views.predict_review, name='predict_review'),
    path('reviews/', views.review_explorer, name='reviews'),
    path('negative/', views.negative_feedback, name='negative_feedback'),
    path('export/reviews.csv', views.export_reviews, name='export_reviews'),
    path('export/analytics.csv', views.export_analytics, name='export_analytics'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
