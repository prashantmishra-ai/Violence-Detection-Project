from django.http import HttpResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
def index(request):
    return render(request,'index.html')
def developer(request):
    return render(request,'developers.html')
def upload(request):
    if request.method == "POST":
        uploaded_file = request.FILES['files']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name,uploaded_file)
        uploaded_file_path = fs.path(filename)
        context = {'message':'Video Analyzed Successfully', 'results':'Violence Detected'}
        return render(request,"results.html",context)
def sources(request):
    return render(request,'sources.html')
def about(request):
    return render(request,'about.html')