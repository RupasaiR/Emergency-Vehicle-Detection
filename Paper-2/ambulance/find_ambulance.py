import cv2
import os
import boto3
import time


rek_client=boto3.client('rekognition',aws_access_key_id='AKIAYZZFFZUHT427AVRZ',aws_secret_access_key="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",region_name='us-west-2')


prev_time = time.time()
delay = 0.1 # in seconds
# Read the video from specified path
cam = cv2.VideoCapture('C:\\Users\\rajesh\\Desktop\\Emergency-Vehicle-Detection-master\\Emergency-Vehicle-Detection-master\\Paper-2\\ambulance\\video.mp4') 
currentframe = 0
detected=False
while (True):
    # reading from frame
    ret, frame = cam.read()
    if ret:
        if time.time() - prev_time > delay:
            # if video is still left continue creating images
            name = './' + str(currentframe) + '.jpg'
            print('Creating...' + name)
            # writing the extracted images
            cv2.imwrite(name, frame)


            image=name
            print('captured '+image)
            with open(image,'rb') as img:
              img1=img.read()
            match_response = rek_client.detect_labels(Image={'Bytes': img1})
            y=match_response['Labels']
            p=list(filter(lambda y:y['Name']=='Ambulance',y))
            if len(p)!=0:
                print(p[0]['Confidence'])
                print('ambulance detected')
                detected=True
                break
            else:
                print('no ambulance detected')


            currentframe += 1
            prev_time = time.time()
    else:
        break

if(detected):
    # Test whether an Emergency signal is present in an audio sample
    from em_detection import *

    # Read the test file
    test_file = '../police-siren_90bpm_A#_major.wav    '
    y, sr = librosa.load(test_file, sr=8000)

    # Load the scaler obtained from the train data
    scaler_filename = "scaler.save"
    scaler = joblib.load(scaler_filename)

    classes = predict_probability(y, scaler)
    


'''
for i in range(currentframe):
    image=str(i)+'.jpg'
    print('captured '+image)
    with open(image,'rb') as img:
      img1=img.read()
    match_response = rek_client.detect_labels(Image={'Bytes': img1})
    y=match_response['Labels']
    p=list(filter(lambda y:y['Name']=='Ambulance',y))
    if len(p)!=0:
        print(p[0]['Confidence'])
        print('ambulance detected')
    else:
        print('no ambulance detected')
'''
