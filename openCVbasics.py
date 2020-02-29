import cv2
import matplotlib.pyplot as plt
import numpy as np

A = cv2.imread('/home/bb-spr/Downloads/lenna.png',-1)
B = cv2.imread('/home/bb-spr/Downloads/sunset.jpeg',-1)
A = cv2.resize(A,(512,512))
B = cv2.resize(B,(A.shape[0],A.shape[1]))

rgb = cv2.cvtColor(A,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(A,cv2.COLOR_BGR2GRAY)
grayOrig = np.copy(gray)

r,g,b = cv2.split(rgb)    # rgb = cv2.merge((r,g,b))
rgb  = np.dstack((r,g,b))
rgbOrig = np.copy(rgb)
eye= rgbOrig[250:280, 250:280,:]

line = cv2.line(rgb,(0,0),(255,255),(147,96,44),2)  # image, startPoint, endPoint, color, lineThickness
rectangle = cv2.rectangle(rgb,(255,255),(512,512),(0,0,255),-1)
ellipse = cv2.ellipse(rgb, (220,100), (100,50), 45, 0, 360, (0,255,0), 5, cv2.LINE_AA)
text = cv2.putText(rgb,'Text',(200,100), cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2, cv2.LINE_AA)

rgb[250:280, 250:280,:] = eye
sum = np.array(A/2 + B/2, dtype = 'uint8')

gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
kernel = np.ones((5,5),dtype = np.uint8)
gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel) #DILATE, ERODE, OPEN, TOPHAT
# indThres = np.where(gray < 200)
# indThres1 = np.where(gray >= 200)
# gray[indThres] = 0
# gray[indThres1] = 1

currFig0 = plt.figure()
plt.imshow(ellipse)
plt.show()
plt.figure(100)
plt.imshow(gray, cmap = 'gray')
plt.show()
plt.close(currFig0)
plt.close(100)  #plt.close('all')

kernel = np.ones((10,10), np.float32)/100
filt1 = cv2.filter2D(rgbOrig,-1, kernel)
blur = cv2.blur(rgbOrig,(10,10))
gblur = cv2.GaussianBlur(rgbOrig,(9,9),5)
median = cv2.medianBlur(rgbOrig,5)
bilateral  =cv2.bilateralFilter(rgbOrig,9,75,75)
titles  = ['Original', 'Filter', 'Blur','Gblur','median']
filtered = [rgbOrig, filt1,blur, gblur,median]

plt.figure(2)
for i in range(len(titles)):
    plt.subplot(2,3,i+1)
    plt.imshow(filtered[i])
    plt.title(titles[i])
plt.show()

laplacian =cv2.Laplacian(grayOrig,cv2.CV_64F, ksize = 3)
sobelX = cv2.Sobel(grayOrig, cv2.CV_64F, 1, 0, ksize = 3)
sobelY = cv2.Sobel(grayOrig, cv2.CV_64F, 1, 0, ksize = 3)
canny = cv2.Canny(grayOrig, 0,255)

sobelCombined = cv2.bitwise_or(np.uint8(sobelX), np.uint8(sobelY)) #can also use np.bitwise_or

gradients = [laplacian,sobelX, sobelY, sobelCombined,canny]
titles = ['Laplacian', 'SobelX', 'SobelY','sobelCombined','canny']

plt.figure(3)
for i in range(len(titles)):
    plt.subplot(2,3,i+1)
    plt.imshow(gradients[i], cmap = 'gray')
    plt.title(titles[i])
plt.show()

logo = cv2.imread('/home/bb-spr/Downloads/OpenCV_Logo.png', -1)
logo_gray  = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, thres = cv2.threshold(logo_gray,127,255,0)
contours = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)       #cv2.contourArea(contour)
cv2.drawContours(logo,contours[1],-1,(127,127,0),3)
logo = logo[:,:,[2,1,0]]

plt.figure()
plt.imshow(logo)
plt.show()

plt.figure()
hist = cv2.calcHist([grayOrig],[0], None, [256],[0,256])  # plt.hist(grayOrig.ravel(), 256, [0,100]) #number
plt.plot(np.arange(0,256), hist)
plt.show()

height = grayOrig.shape[0]
width = grayOrig.shape[1]
roi_vertices = np.array([[(0,height), (width/2, height/2), (width,height)]], dtype = np.int32)
mask = np.zeros((height, width), dtype = np.uint8)
cv2.fillPoly(mask,roi_vertices,255)
plt.figure()
plt.imshow(mask)
plt.show()

# cv2.imshow('lenna', A)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
