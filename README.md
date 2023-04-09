<img src="./diagram-export-10_9_2022%2C%205_07_28%20PM.png" alt="Architecture sketch" width="500"/>


## foosball-ai  
[![codecov](https://codecov.io/gh/DarwinsBuddy/foosball-ai/branch/main/graph/badge.svg?token=ACYNOG1WFW)](https://codecov.io/gh/DarwinsBuddy/foosball-ai)
![Tests](https://github.com/DarwinsBuddy/foosball-ai/actions/workflows/test.yml/badge.svg)

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
0. (optional) Download demo video
https://drive.google.com/file/d/1tmaV2U_amzUHP3u4lQtom16IFclRu5Fe/view?usp=sharing

1. Activate venv  
```#!/bin/sh  
. ./venv/bin/activate  
```  
2. Run
```#!/bin/sh  
python3 -m foosball -f ./demo.mp4
```
## Misc
### Calibration
There is plenty to do for that project, but for now we merely calibrate the color of the ball manually by using
calibration mode like follows
```#!/bin/sh  
python3 -m foosball -f ./demo.mp4 -c
```
