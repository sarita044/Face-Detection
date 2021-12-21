import cv2

#load some pre-trained data on face frontals from opencv(haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# to capture video from webcam
webcam = cv2.VideoCapture(0)

#Iterate forever over frames
while True:
    #read the current frame
    successful_frame_read, frame = webcam.read()

    #must convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)
    # Draw rectangle around the faces
    for(x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #Display video with face detection rectangles    
    cv2.imshow('video captured',frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()                      






"""
#choose an image to detect face in
img = cv2.imread('img2.jpg')

#Covert to grayscale
grayscaled_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect faces
#detect objects of different sizes in the input image
#The detected objects are returned as a list of rectangles(coordinates of rectangle.)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around the faces
for(x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


#to show above image
cv2.imshow('Picture of Child',img)
cv2.waitKey()

print("Working Fine : ")
"""
