from django.shortcuts import render

# Create your views here.


from django.shortcuts import render
from .forms import PredictForm
from .model_utils import predict, haversine_distance, build_fig
from PIL import Image

def index(request):
    if request.method == 'POST':
        form = PredictForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            real_lat = form.cleaned_data['real_lat']
            real_lon = form.cleaned_data['real_lon']

            # Открываем изображение напрямую из потока (без сохранения на диск)
            pil_image = Image.open(image_file)

            # Предсказание
            pred_lat, pred_lon = predict(pil_image)

            # Ошибка в км
            error_km = haversine_distance(real_lat, real_lon, pred_lat, pred_lon)

            fig_html = build_fig(pil_image, real_lat, real_lon, pred_lat, pred_lon)

            context = {
                'real_lat': real_lat,
                'real_lon': real_lon,
                'pred_lat': pred_lat,
                'pred_lon': pred_lon,
                'error_km': error_km,
                'plot': fig_html
            }
            return render(request, 'Geo_samsung/result.html', context)
    else:
        form = PredictForm()

    return render(request, 'Geo_samsung/index.html', {'form': form})