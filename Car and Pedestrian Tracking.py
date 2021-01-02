import cv2;

# load car video
car_video = cv2.VideoCapture("pedestrians and cars.mp4")

# create car classifier
car_classifier = cv2.CascadeClassifier("cars.xml")
pedestrian_classifier = cv2.CascadeClassifier("pedestrian.xml")

while True:
    (read_successful, frame)  = car_video.read()

    if read_successful:
        #convert image to grayscale
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    #Detect cars and pedestrians
    cars = car_classifier.detectMultiScale(gray_frame)
    pedestrians = pedestrian_classifier.detectMultiScale(gray_frame)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # Show the Video with detected objects
    cv2.imshow("Wohoo!! Cars and Pedestrians Detected",frame)
    key = cv2.waitKey(1)

    #if Q is pressed it will close the video
    if key==81 or key==113:
        break

