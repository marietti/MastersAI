from background_task import background
from .models import Update_Test

@background(schedule=5)
def run_changepoint_5():
    id = Update_Test.objects.first()
    obj = Update_Test.objects.get(id=id.id)
    obj.changepoint = obj.changepoint + 5
    obj.save()

