from roboflow import Roboflow
rf = Roboflow(api_key="xrUrjP1rGu65ygnqm6WW")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(4)
dataset = version.download("yolov11")