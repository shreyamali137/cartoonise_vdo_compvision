import cv2
import numpy as np



cap= cv2.VideoCapture(0)


while(True):
	ret,img = cap.read()

	
	Z = img.reshape((-1,3))
	
	Z = np.float32(Z)

	
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 0.9)

	K = 12
	ret,label,center=cv2.kmeans(Z,K,None,criteria,2,cv2.KMEANS_RANDOM_CENTERS)

	
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	
	res2 = res2+15

	cv2.imshow("Cartoonised", res2)
	
	if(cv2.waitKey(9)==ord('q')):
		break

cap.release()
cv2.destroyAllWindows()