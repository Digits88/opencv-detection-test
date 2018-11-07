# opencv-detection-test
Repo for testing OpenCV face and eye detection on real-time video stream


### How-To ?

Make sure you have `virtualenv` installed, then run (`$` is for non-privileged mode):
- `$ virtualenv -p python3 venv`
- `$ source ./venv/bin/activate`

This will activate the virtual environment and you'll see a `(venv)` in the shell prompt.

- `(venv) $ pip install -r requirements.txt` to install the dependencies.
- Make sure you have a webcam and run : `(venv) $ python app.py`

You will see the faces in front of your webcam being detected and surrounded by a green rectangle (and by default, along with eyes and smiles)