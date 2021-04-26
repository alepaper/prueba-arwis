from django.shortcuts import render
from django.views.generic.base import TemplateView
from voice_classification import DetectMusicNotes as dmn
from datetime import datetime
import time


def home(request):
    return render(request, "test.html")


def index(request):
    return render(request, "vc_index.html")


class IndexView(TemplateView):
    template_name = "index.html"


async def websocket_view(socket):
    await socket.accept()
    message = ""
    while message != "-1":
        message = await socket.receive()
        detectNote = dmn.DetectNotesFromCQT('', fragment=True, backtrack=False)

        print(datetime.now())
        if message.get('bytes', False):
            t1 = time.time()
            with open('myfile.wav', mode='bw') as f:
                f.write(message.get('bytes', 0))
            t2 = time.time()
            note = detectNote.main_note_in_fragment(pathFile='myfile.wav')
            t3 = time.time()
            print(note)
            await socket.send_text(note['mainNoteLatin'])
            print(f'Se demoró grabando {t2-t1}s y procesando {t3-t2}s')
        else:
            print("No llega audio")
        # await socket.send_text("message")
    await socket.send_text('Cerrando conexión')
    await socket.close()
