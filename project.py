# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 08:24:47 2021

@author: WELCOME
"""

# -*- coding: utf-8 -*-
import datetime
import ibm_boto3
from ibm_botocore.client import Config, ClientError
import cv2
import random
import time
from cloudant.client import Cloudant
from cloudant.error import CloudantException 
from cloudant.result import Result
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
face_classifier=cv2.CascadeClassifier(r"C:\Users\WELCOME\Desktop\project\haarcascade_frontalface_default.xml")

COS_ENDPOINT ="https://http://s3.jp-tok.cloud-object-storage.appdomain.cloud"
COS_API_KEY_ID ="Vkb2Pj2rL-CSlbODF0ogd_P7kpKHn1wQgyRPK3S7D9QV"
COS_AUTH_ENDPOINT ="https://iam.cloud.ibm.com/identity/token"
COS_RESOURCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/049467f052384c32bcb417ba31024ad6:309e2811-37bb-4d93-a4ee-18a20c37d331::"
client = Cloudant("apikey-v2-1mr4dt4lpg2fiiieu2no7bc4vvig4915gz2h5qt6z8pu","9e5e5d17b8ef53e0931231193cab42c7",url="https://apikey-v2-1mr4dt4lpg2fiiieu2no7bc4vvig4915gz2h5qt6z8pu:9e5e5d17b8ef53e0931231193cab42c7@8664ae78-5c13-4e5d-abad-b932a101914c-bluemix.cloudantnosqldb.appdomain.cloud")
client.connect()
database_name = "customer_details"
picname=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
picname=picname+".jpg"
pic=datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
model = load_model(r"C:\Users\WELCOME\Desktop\project\humanfaces.h5")
cos = ibm_boto3.resource("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)
def multi_part_upload(bucket_name, item_name, file_path):
    try:
        print("Starting file transfer for {0} to bucket: {1}\n".
              format(item_name, bucket_name))
        # set 5 MB chunks
        part_size = 1024 * 1024 * 5
        # set threadhold to 15 MB
        file_threshold = 1024 * 1024 * 15
        # set the transfer threshold and chunk size
        transfer_config = ibm_boto3.s3.transfer.TransferConfig(
            multipart_threshold=file_threshold,
            multipart_chunksize=part_size)
        
        # the upload_fileobj method will automatically execute a multi-part upload
        # in 5 MB chunks for all files over 15 MB
        with open(file_path, "rb") as file_data:
            cos.Object(bucket_name, item_name).upload_fileobj(
                Fileobj=file_data,
                Config=transfer_config
            )
        print("Transfer for {0} Complete!\n".format(item_name))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to complete multi-part upload: {0}".format(e))
        
video=cv2.VideoCapture(0)
while True:
    #capture the first frame
    print("hie")
    check,frame=video.read()
    print("check",check)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    cv2.imshow('Face detection', frame)
    """"if cv2.waitKey(1) & 0xFF == ord('q'):
        
              break"""
          
    #drawing rectangle boundries for the detected face
    for(x,y,w,h) in faces:
        print("yes its face")
        roi_color = frame[y:y+h, x:x+w]
        cv2.imwrite(picname, roi_color)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (127,0,255), 2)
        cv2.imshow('Face detection', frame)
       
        person=1
        #cloudant db database
        my_database = client.create_database(database_name)       
        #cloud object storage
        multi_part_upload("userimage-120",picname,pic+".jpg")  
        img =image.load_img(picname,grayscale=False,
                     target_size= (64,64))#loading of the image
        x = image.img_to_array(img)
        print(x.shape)
        x = np.expand_dims(x,axis = 0)#changing the shape
        pred = model.predict_classes(x)
        if(pred[0]==0):
            pred = "allu arjun"
        elif(pred[0]==1):
            pred="modi"
        elif(pred[0]==2):
            pred="obama"
        else:
            pred = "selena"
        if my_database.exists():
            print("'{database_name}' successfully created.")
            json_document = {
                    "_id": pic,
                    "link":COS_ENDPOINT+"/userimage-120/"+picname,
                    "prediction":pred
                    }
            new_document = my_database.create_document(json_document)
            if new_document.exists():
                print("Document '{new_document}' successfully created.")
        
        person=0
    #waitKey(1)- for every 1 millisecond new frame will be captured
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
#release the camera
video.release()
    #destroy all windows
cv2.destroyAllWindows()

client.disconnect()




