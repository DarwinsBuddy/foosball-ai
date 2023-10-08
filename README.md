Outdated
<img src="./diagram-export-10_9_2022%2C%205_07_28%20PM.png" alt="Architecture sketch" width="250"/>


## foosball-ai  
[![codecov](https://codecov.io/gh/DarwinsBuddy/foosball-ai/branch/main/graph/badge.svg?token=ACYNOG1WFW)](https://codecov.io/gh/DarwinsBuddy/foosball-ai)
[![Tests](https://github.com/DarwinsBuddy/foosball-ai/actions/workflows/test.yml/badge.svg)](https://github.com/DarwinsBuddy/foosball-ai/actions/workflows/test.yml)

## Prerequisites  
* python3  
* pip
  
## Install  
  
1. Setup a venv  
```#!/bin/sh  
python3 -m venv ./venv  
```  
2. Activate venv  
```#!/bin/sh  
. ./venv/bin/activate  
```  
3. Install requirements  
```#!/bin/sh  
pip install -r requirements.txt  
```
## Run  
0. (optional) Download video
   1. demo video
      1. orange ball https://drive.google.com/file/d/1tmaV2U_amzUHP3u4lQtom16IFclRu5Fe/view?usp=sharing
      1. yellow ball https://mega.nz/file/UpNSwaBY#7__EPElzGkf6ohM_Oe5kxjJpIV2TUmJ8k63HJV0X4oU
   1. calibration video for camera calibration
      1. orange ball (not needed, since it's without arUco markers)
      2. yellow ball https://mega.nz/file/w98z3ABK#e6rwmejpqAgv3Ipc5CqkkAjdf-M0NEEtcTlGkSc4hUo
1. Activate venv  
> ```#!/bin/sh  
> . ./venv/bin/activate  
> ```
2. Run camera calibration
> ```#!/bin/sh  
> python3 -m foosball -c cam -cv <path-to-calibration-video>.mp4
> ```
3. Run
```#!/bin/sh  
python3 -m foosball -f <path-to-file>.mp4
```
## Misc
### Calibration
#### Color
Color calibration is needed for the ball (or the goals) to be detected.
There are 2 preconfigured ball profiles and 1 goal profile available, accessible
by a cli option. For individual options please refer to the calibration mode where you can select the color range to 
be detected and stored in a color profile accordingly.

> `ball`/`goal` calibration mode
> ```#!/bin/sh  
> python3 -m foosball -f ./demo.mp4 -c [ball|goal]
> ```

#### camera (aruco)
1. Generate Aruco Board
   > ```#!/bin/sh  
   > python3 -m foosball -a
   > ```
2. Print it in DIN A4 format (`aruco.png`)
3. In order to automatically detected the aruco markers and thus
improve performance and accuracy, for each camera an initial calibration has to be done.
Once run this application with calibration mode `cam` and present the printed out version of
the generated aruco board to the camera by
- waving it sturdily in different angles in front of the camera,
- provide the path to a pre-recorded version of it
- or provide a path to a set of pre-recoded images of it

>`cam` calibration mode
> ```#!/bin/sh  
> python3 -m foosball -c cam -cs 50
> ```