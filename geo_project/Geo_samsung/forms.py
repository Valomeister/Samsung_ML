from django import forms

class PredictForm(forms.Form):
    image = forms.ImageField(label='Фото')
    real_lat = forms.FloatField(label='Реальная широта', min_value=-90, max_value=90)
    real_lon = forms.FloatField(label='Реальная долгота', min_value=-180, max_value=180)