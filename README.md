# hand-recognition

Final project for CS6670: Computer Vision. A hand recognition tool for self-centered cameras. This tool builds upon Victor Dibia's handtracking tool (https://github.com/victordibia/handtracking), optimizing it for self-oriented camera view. This tool also uses ensembling with Google's MediaPipe Hand Detection. 


Download the video file at https://drive.google.com/file/d/1fKxwCtJ707pXzdSwV_7jfKHnSCflwyJX/view?usp=sharing, then put it in the folder with main.py and run main.py to start labeling.

To do live labeling with camera input run
`python live_labeling.py`
To do live labeling with given video file (or any other video file with a given path) run
`python live_labeling.py --source ./combine_neck_pilot.mp4`


![image](https://user-images.githubusercontent.com/56843532/146818066-bd8205fc-23e1-4d69-8ec1-4e9ca431031b.png)
