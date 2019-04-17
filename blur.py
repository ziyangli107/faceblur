import math  
import numpy as np  
from numpy import fft  
import cv2 
from skimage import img_as_uint 

def mosaic(img,xmin,ymin,xmax,ymax):

    kernel=np.ones((5,5),np.uint16)

    x = xmax - xmin
    y = ymax - ymin
    if (x % 3 == 1):
        xmax=xmax-1
    if (x % 3 == 2):
        xmax=xmax-2

    if (y % 3 == 1):
        ymax=ymax-1
    if (y % 3 == 2):
        ymax=ymax-2
  
    corner=img[ymin:ymax,xmin:xmax]
    im_h = corner.shape[0]
    im_w = corner.shape[1]
    
    h = im_h/3
    w = im_w/3
    
    temp=np.zeros((im_h,im_w,3))
  
    temp[h:2*h,2*w:3*w] = corner[0:h,0:w]
    temp[h:2*h,0:w] = corner[0:h,w:2*w]
    temp[2*h:3*h,w:2*w] = corner[0:h,2*w:3*w]

    temp[2*h:3*h,2*w:3*w] = corner[h:2*h,0:w]
    temp[0:h,2*w:3*w] = corner[h:2*h,w:2*w]
    temp[0:h,w:2*w] = corner[h:2*h,2*w:3*w]

    temp[0:h,0:w] = corner[2*h:3*h,0:w]
    temp[2*h:3*h,0:w] = corner[2*h:3*h,w:2*w]
    temp[h:2*h,w:2*w] = corner[2*h:3*h,2*w:3*w]

    img[ymin:ymax,xmin:xmax] = temp

    tmp=np.zeros((im_h+6,im_w+6,3),np.uint16)
    tmp[3:3+im_h,3:3+im_w]=img[ymin:ymax,xmin:xmax]
   
    tmp=cv2.filter2D(tmp,-1,kernel)*10
    img=img_as_uint(img)
    
    img[ymin:ymax,xmin:xmax]=tmp[3:3+im_h,3:3+im_w]
    print(tmp[3:3+im_h,3:3+im_w])
    cv2.imwrite('test.png',img)
def demosaic(img,xmin,ymin,xmax,ymax):

    corner=img[ymin:ymax,xmin:xmax]
    print(img[ymin:ymax,xmin:xmax])
    im_h = corner.shape[0]
    im_w = corner.shape[1]
    new_h = im_h
    new_w = im_w
    whole=new_h*new_w

    res = np.zeros((whole,whole),np.uint16)

    b,g,r = cv2.split(corner)

    b= b.flatten()
    g= g.flatten()
    r= r.flatten()

    count = 0

    for i in range(new_h):
        for j in range(new_w):
            if (i == 0) :
	        if (j == 0):
	            res[0,0]=1
                    res[0,1]=1
                    res[0,2]=1
		    res[0,new_w]=1
		    res[0,new_w+1]=1
	            res[0,new_w+2]=1
		    res[0,2*new_w]=1
		    res[0,2*new_w+1]=1
		    res[0,2*new_w+2]=1
                    count = count + 1
	        elif (j == 1):
		    res[1,0]=1
		    res[1,1]=1
		    res[1,2]=1
		    res[1,3]=1
		    res[1,new_w]=1
		    res[1,new_w+1]=1
		    res[1,new_w+2]=1
		    res[1,new_w+3]=1
		    res[1,2*new_w]=1
		    res[1,2*new_w+1]=1
		    res[1,2*new_w+2]=1
		    res[1,2*new_w+3]=1
		    count = count + 1
	        elif (j == new_w-2):
		    res[count,j-2]=1
		    res[count,j-1]=1
		    res[count,j]=1
		    res[count,j+1]=1
		    res[count,new_w+j-2]=1
		    res[count,new_w+j-1]=1
		    res[count,new_w+j]=1
                    res[count,new_w+j+1]=1
                    res[count,2*new_w+j-2]=1
                    res[count,2*new_w+j-1]=1
                    res[count,2*new_w+j]=1
                    res[count,2*new_w+j+1]=1
		    count = count + 1
                elif (j == new_w-1):
                    res[count,j-2]=1
                    res[count,j-1]=1
		    res[count,j]=1
		    res[count,new_w+j-2]=1
		    res[count,new_w+j-1]=1
                    res[count,new_w+j]=1
                    res[count,2*new_w+j-2]=1
                    res[count,2*new_w+j-1]=1
                    res[count,2*new_w+j]=1
		    count = count + 1
                else:
                    res[count,j-2]=1
                    res[count,j-1]=1
                    res[count,j]=1
                    res[count,j+1]=1
                    res[count,j+2]=1
                    res[count,new_w+j-2]=1
                    res[count,new_w+j-1]=1
                    res[count,new_w+j]=1
                    res[count,new_w+j+1]=1
                    res[count,new_w+j+2]=1
                    res[count,2*new_w+j-2]=1
                    res[count,2*new_w+j-1]=1
                    res[count,2*new_w+j]=1
                    res[count,2*new_w+j+1]=1
                    res[count,2*new_w+j+2]=1
		    count = count + 1

            elif (i == 1):
                if (j == 0):
	            res[count,0]=1
	            res[count,1]=1
	            res[count,2]=1
	            res[count,new_w]=1
	            res[count,new_w+1]=1
	            res[count,new_w+2]=1
	            res[count,2*new_w]=1
	            res[count,2*new_w+1]=1
	            res[count,2*new_w+2]=1
	            res[count,3*new_w]=1
	            res[count,3*new_w+1]=1
	            res[count,3*new_w+2]=1
		    count = count + 1
		elif (j == 1):
	            res[count,0]=1
	            res[count,1]=1
	            res[count,2]=1
	            res[count,3]=1
	            res[count,new_w]=1
	            res[count,new_w+1]=1
	            res[count,new_w+2]=1
	            res[count,new_w+3]=1
	            res[count,2*new_w]=1
	            res[count,2*new_w+1]=1
	            res[count,2*new_w+2]=1
	            res[count,2*new_w+3]=1
	            res[count,3*new_w]=1
	            res[count,3*new_w+1]=1
	            res[count,3*new_w+2]=1
	            res[count,3*new_w+3]=1
		    count = count + 1
	        elif (j == new_w-2):
		    res[count,j-2]=1
		    res[count,j-1]=1
		    res[count,j]=1
		    res[count,j+1]=1
		    res[count,new_w+j-2]=1
		    res[count,new_w+j-1]=1
		    res[count,new_w+j]=1
                    res[count,new_w+j+1]=1
		    res[count,2*new_w+j-2]=1
		    res[count,2*new_w+j-1]=1
		    res[count,2*new_w+j]=1
                    res[count,2*new_w+j+1]=1
		    res[count,3*new_w+j-2]=1
		    res[count,3*new_w+j-1]=1
		    res[count,3*new_w+j]=1
                    res[count,3*new_w+j+1]=1
		    count = count + 1
	        elif (j == new_w-1):
		    res[count,j-2]=1
		    res[count,j-1]=1
		    res[count,j]=1
		    res[count,new_w+j-2]=1
		    res[count,new_w+j-1]=1
		    res[count,new_w+j]=1
		    res[count,2*new_w+j-2]=1
		    res[count,2*new_w+j-1]=1
		    res[count,2*new_w+j]=1
		    res[count,3*new_w+j-2]=1
		    res[count,3*new_w+j-1]=1
		    res[count,3*new_w+j]=1
		    count = count + 1
                else :
                    res[count,j-2]=1
                    res[count,j-1]=1
                    res[count,j]=1
                    res[count,j+1]=1
                    res[count,j+2]=1
                    res[count,new_w+j-2]=1
                    res[count,new_w+j-1]=1
                    res[count,new_w+j]=1
                    res[count,new_w+j+1]=1
                    res[count,new_w+j+2]=1
                    res[count,2*new_w+j-2]=1
                    res[count,2*new_w+j-1]=1
                    res[count,2*new_w+j]=1
                    res[count,2*new_w+j+1]=1
                    res[count,2*new_w+j+2]=1
                    res[count,3*new_w+j-2]=1
                    res[count,3*new_w+j-1]=1
                    res[count,3*new_w+j]=1
                    res[count,3*new_w+j+1]=1
                    res[count,3*new_w+j+2]=1
		    count = count + 1

            elif (i == new_h-2) :
                if (j == 0) :
	            res[count,(i-2)*new_w]=1
	            res[count,(i-2)*new_w+1]=1
	            res[count,(i-2)*new_w+2]=1
	            res[count,(i-1)*new_w]=1
	            res[count,(i-1)*new_w+1]=1
	            res[count,(i-1)*new_w+2]=1
	            res[count,i*new_w]=1
	            res[count,i*new_w+1]=1
	            res[count,i*new_w+2]=1
	            res[count,(i+1)*new_w]=1
	            res[count,(i+1)*new_w+1]=1
	            res[count,(i+1)*new_w+2]=1
		    count = count + 1
		elif (j == 1) :
	            res[count,(i-2)*new_w]=1
	            res[count,(i-2)*new_w+1]=1
	            res[count,(i-2)*new_w+2]=1
	            res[count,(i-2)*new_w+3]=1
	            res[count,(i-1)*new_w]=1
	            res[count,(i-1)*new_w+1]=1
	            res[count,(i-1)*new_w+2]=1
	            res[count,(i-1)*new_w+3]=1
	            res[count,i*new_w]=1
	            res[count,i*new_w+1]=1
	            res[count,i*new_w+2]=1
	            res[count,i*new_w+3]=1
	            res[count,(i+1)*new_w]=1
	            res[count,(i+1)*new_w+1]=1
	            res[count,(i+1)*new_w+2]=1
	            res[count,(i+1)*new_w+3]=1
		    count = count + 1
		elif (j == new_w-2) :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-2)*new_w+j+1]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,(i-1)*new_w+j+1]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
	            res[count,i*new_w+j+1]=1
	            res[count,(i+1)*new_w+j-2]=1
	            res[count,(i+1)*new_w+j-1]=1
	            res[count,(i+1)*new_w+j]=1
	            res[count,(i+1)*new_w+j+1]=1
		    count = count + 1
		elif (j == new_w-1) :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
	            res[count,(i+1)*new_w+j-2]=1
	            res[count,(i+1)*new_w+j-1]=1
	            res[count,(i+1)*new_w+j]=1
		    count = count + 1
		else :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-2)*new_w+j+1]=1
	            res[count,(i-2)*new_w+j+2]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,(i-1)*new_w+j+1]=1
	            res[count,(i-1)*new_w+j+2]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
	            res[count,i*new_w+j+1]=1
	            res[count,i*new_w+j+2]=1
	            res[count,(i+1)*new_w+j-2]=1
	            res[count,(i+1)*new_w+j-1]=1
	            res[count,(i+1)*new_w+j]=1
	            res[count,(i+1)*new_w+j+1]=1 
	            res[count,(i+1)*new_w+j+2]=1 
		    count = count + 1
  
            elif (i == new_h-1) :
                if (j == 0) :
	            res[count,(i-2)*new_w]=1
	            res[count,(i-2)*new_w+1]=1
	            res[count,(i-2)*new_w+2]=1
	            res[count,(i-1)*new_w]=1
	            res[count,(i-1)*new_w+1]=1
	            res[count,(i-1)*new_w+2]=1
	            res[count,i*new_w]=1
	            res[count,i*new_w+1]=1
	            res[count,i*new_w+2]=1
		    count = count + 1        
		elif (j == 1) :
	            res[count,(i-2)*new_w]=1
	            res[count,(i-2)*new_w+1]=1
	            res[count,(i-2)*new_w+2]=1
	            res[count,(i-2)*new_w+3]=1
	            res[count,(i-1)*new_w]=1
	            res[count,(i-1)*new_w+1]=1
	            res[count,(i-1)*new_w+2]=1
	            res[count,(i-1)*new_w+3]=1
	            res[count,i*new_w]=1
	            res[count,i*new_w+1]=1
	            res[count,i*new_w+2]=1
	            res[count,i*new_w+3]=1
		    count = count + 1
		elif (j == new_w-2) :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-2)*new_w+j+1]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,(i-1)*new_w+j+1]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
	            res[count,i*new_w+j+1]=1
		    count = count + 1
		elif (j == new_w-1) :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
		    count = count + 1
		else :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-2)*new_w+j+1]=1
	            res[count,(i-2)*new_w+j+2]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,(i-1)*new_w+j+1]=1
	            res[count,(i-1)*new_w+j+2]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
	            res[count,i*new_w+j+1]=1
	            res[count,i*new_w+j+2]=1
		    count = count + 1

            else :
                if (j == 0) :
	            res[count,(i-2)*new_w]=1
	            res[count,(i-2)*new_w+1]=1
	            res[count,(i-2)*new_w+2]=1
	            res[count,(i-1)*new_w]=1
	            res[count,(i-1)*new_w+1]=1
	            res[count,(i-1)*new_w+2]=1
	            res[count,i*new_w]=1
	            res[count,i*new_w+1]=1
	            res[count,i*new_w+2]=1
	            res[count,(i+1)*new_w]=1
	            res[count,(i+1)*new_w+1]=1
	            res[count,(i+1)*new_w+2]=1
	            res[count,(i+2)*new_w]=1
	            res[count,(i+2)*new_w+1]=1
	            res[count,(i+2)*new_w+2]=1
		    count = count + 1             
		elif (j == 1) :
	            res[count,(i-2)*new_w]=1
	            res[count,(i-2)*new_w+1]=1
	            res[count,(i-2)*new_w+2]=1
	            res[count,(i-2)*new_w+3]=1
	            res[count,(i-1)*new_w]=1
	            res[count,(i-1)*new_w+1]=1
	            res[count,(i-1)*new_w+2]=1
	            res[count,(i-1)*new_w+3]=1
	            res[count,i*new_w]=1
	            res[count,i*new_w+1]=1
	            res[count,i*new_w+2]=1
	            res[count,i*new_w+3]=1
	            res[count,(i+1)*new_w]=1
	            res[count,(i+1)*new_w+1]=1
	            res[count,(i+1)*new_w+2]=1
	            res[count,(i+1)*new_w+3]=1
	            res[count,(i+2)*new_w]=1
	            res[count,(i+2)*new_w+1]=1
	            res[count,(i+2)*new_w+2]=1
	            res[count,(i+2)*new_w+3]=1
		    count = count + 1
		elif (j == new_w-2) :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-2)*new_w+j+1]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,(i-1)*new_w+j+1]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
	            res[count,i*new_w+j+1]=1
	            res[count,(i+1)*new_w+j-2]=1
	            res[count,(i+1)*new_w+j-1]=1
	            res[count,(i+1)*new_w+j]=1
	            res[count,(i+1)*new_w+j+1]=1
	            res[count,(i+2)*new_w+j-2]=1
	            res[count,(i+2)*new_w+j-1]=1
	            res[count,(i+2)*new_w+j]=1
	            res[count,(i+2)*new_w+j+1]=1
		    count = count + 1
		elif (j == new_w-1) :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
	            res[count,(i+1)*new_w+j-2]=1
	            res[count,(i+1)*new_w+j-1]=1
	            res[count,(i+1)*new_w+j]=1
	            res[count,(i+2)*new_w+j-2]=1
	            res[count,(i+2)*new_w+j-1]=1
	            res[count,(i+2)*new_w+j]=1
		    count = count + 1
                else :
	            res[count,(i-2)*new_w+j-2]=1
	            res[count,(i-2)*new_w+j-1]=1
	            res[count,(i-2)*new_w+j]=1
	            res[count,(i-2)*new_w+j+1]=1
	            res[count,(i-2)*new_w+j+2]=1
	            res[count,(i-1)*new_w+j-2]=1
	            res[count,(i-1)*new_w+j-1]=1
	            res[count,(i-1)*new_w+j]=1
	            res[count,(i-1)*new_w+j+1]=1
	            res[count,(i-1)*new_w+j+2]=1
	            res[count,i*new_w+j-2]=1
	            res[count,i*new_w+j-1]=1
	            res[count,i*new_w+j]=1
	            res[count,i*new_w+j+1]=1
	            res[count,i*new_w+j+2]=1
	            res[count,(i+1)*new_w+j-2]=1
	            res[count,(i+1)*new_w+j-1]=1
	            res[count,(i+1)*new_w+j]=1
	            res[count,(i+1)*new_w+j+1]=1
	            res[count,(i+1)*new_w+j+2]=1
	            res[count,(i+2)*new_w+j-2]=1
	            res[count,(i+2)*new_w+j-1]=1
	            res[count,(i+2)*new_w+j]=1
	            res[count,(i+2)*new_w+j+1]=1
	            res[count,(i+2)*new_w+j+2]=1
		    count = count + 1 


    b1=np.uint16(np.linalg.solve(res,b))/10
    b1=b1.reshape((im_h,im_w))

    g1=np.uint16(np.linalg.solve(res,g))/10
    g1=g1.reshape((im_h,im_w))

    r1=np.uint16(np.linalg.solve(res,r))/10
    r1=r1.reshape((im_h,im_w))

    corner=cv2.merge([b1,g1,r1])
    img[ymin:ymax,xmin:xmax] = corner


    cv2.imwrite('restore.png',img)

#eg：im=cv2.imread('im.jpg',3)
#eg: mosaic(im,954,706,974,732)

#dst=cv2.imread('test.png',3)
#demosaic(dst,954,706,974,732)

