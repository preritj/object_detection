import requests
import json
import cv2

addr = 'http://localhost:5000'
api_url = addr + '/api'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('/media/storage/AVAretail/MasterCard_shelf/snapshot_2018_07_09_16_21_08.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
while True:
    try:
        # send http request with image and receive response
        response = requests.post(api_url, data=img_encoded.tostring(), headers=headers)
        # decode response
        print(json.loads(response.text))
    except:
        pass
