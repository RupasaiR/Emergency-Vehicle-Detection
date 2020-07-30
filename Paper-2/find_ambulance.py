import time
from urllib.request import urlopen
import datetime
import geocoder
import warnings
warnings.filterwarnings("ignore")
import cv2
import os
import boto3
import time
from em_detection import *
import moviepy.editor as mp
url='video1.mp4'

clip = mp.VideoFileClip(url).subclip(0,15)
clip.audio.write_audiofile("theaudio.wav")

rek_client=boto3.client('rekognition',aws_access_key_id='AKIAYZZFFZUHT427AVRZ',aws_secret_access_key="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",region_name='us-west-2')


prev_time = time.time()
delay = 0.1 # in seconds

cam = cv2.VideoCapture('C:\\Users\\rajesh\\Desktop\\Emergency-Vehicle-Detection-master\\Emergency-Vehicle-Detection-master\\Paper-2\\'+url) 
currentframe = 0
ambulance_detected=False
while (True):
    # reading from frame
    ret, frame = cam.read()
    if ret:
        if time.time() - prev_time > delay:
            
            name = './' + str(currentframe) + '.jpg'

            cv2.imwrite(name, frame)

            

            image=name
   
            with open(image,'rb') as img:
              img1=img.read()
            match_response = rek_client.detect_labels(Image={'Bytes': img1})
            y=match_response['Labels']
            p=list(filter(lambda y:y['Name']=='Ambulance',y))
            if len(p)!=0:
                #print(p[0]['Confidence'])
                print('ambulance detected')
                ambulance_detected=True
                break
            else:
                print('no ambulance detected')


            currentframe += 1
            prev_time = time.time()
    else:
        break
siren_detected=False
if(ambulance_detected):
    
    test_file = './theaudio.wav'
    y, sr = librosa.load(test_file, sr=8000)

    
    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)

    siren_detected = predict_probability(y, scaler)
    print('siren detected')


x = datetime.datetime.now()
g=geocoder.ip('me')
x=str(g.latlng[0])+','+str(g.latlng[1])

if(ambulance_detected):
    y='ambulance detected'
else:
    y='ambulance not detected'
if(siren_detected):
    z='siren detected'
else:
    z='siren not detected'
    print(z)

print(x)
data=urlopen("https://api.thingspeak.com/update?api_key=DDVMUO0VW68MEAVG&field1="+str(x)+"&field2="+str(y)+"&field3="+str(z))


