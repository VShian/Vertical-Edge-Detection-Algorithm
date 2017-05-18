import numpy as np
import cv2
from matplotlib import pyplot as plt


def at(img):
    print('at')
    s2 = int(s / 2)
    th = np.zeros((image_height, image_width), np.uint8)
    intr_img = np.zeros((image_height, image_width))
    for i in range(image_width):
        value = 0
        for j in range(image_height):
            value = value + img[j][i]
            if (j == 0):
                intr_img[j][i] = value
            else:
                intr_img[j][i] = intr_img[j][i - 1] + value
    for i in range(image_width):
        for j in range(image_height):
            index = j * image_width + i
            x1, x2, y1, y2 = i - s2, i + s2, j - s2, j + s2

            if (x1 < 0):
                x1 = 0
            if (x2 >= image_width):
                x2 = image_width - 1
            if (y1 < 0):
                y1 = 0
            if (y2 >= image_height):
                y2 = image_height - 1

            count = (x2 - x1) * (y2 - y1)
            # I(x,y)=s(x2,y2)-s(x1,y2)-s(x2,y1)+s(x1,x1)
            value = intr_img[y2][x2] - intr_img[y1][x2] - intr_img[y2][x1] + intr_img[y1][x1]
            if (img[j][i] * count < value * (1.0 - t)):
                th[j][i] = 0
            else:
                th[j][i] = 255
    return th

def inv(img):
	im=np.zeros((image_height, image_width), np.uint8)
	for i in range(image_width):
		for j in range(image_height):
			if(img[j][i]):
				im[j][i]=0
			else:
				im[j][i]=255
	return im

def ulea(img):
    print('ulea')
    im = np.zeros((image_height, image_width), np.uint8)
    for i in range(1, image_width - 1):
        for j in range(1, image_height - 1):
            if (img[j][i]):
                im[j][i] = img[j][i]
            elif ((img[j - 1][i - 1] and img[j + 1][i + 1]) or (img[j - 1][i] and img[j + 1][i]) or (
                        img[j][i - 1] and img[j][i + 1]) or (img[j - 1][i + 1] and img[j + 1][i - 1])):
                im[j][i] = 255
                # print((j,i))
    return im


def ved(img):
    print('veda')
    im = np.zeros((image_height, image_width), dtype=np.uint8)
    # im=img
    for j in range(0, image_height - 1, 2):
        for i in range(1, image_width - 2):
            col1, col2, col3, col0 = 1, 1, 1, 1
            if (not (img[j][i] and img[j + 1][i])):
                col1 = 0
            if (not (img[j][i + 1] and img[j + 1][i + 1])):
                col2 = 0
            if (not (img[j][i + 2] and img[j + 1][i + 2])):
                col3 = 0
            if (not (img[j][i - 1] and img[j + 1][i - 1])):
                col0 = 0
            if (not (col0 or col3 or col2 or col1)):
                im[j][i] = 255
                im[j + 1][i] = 255
            # im[j][i+1]=255
            # im[j+1][i+1]=255
            else:
                im[j][i] = img[j][i]
                im[j + 1][i] = img[j + 1][i]
    return im


# def sobel(img):
#     sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
#     # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
#     sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
#     abs_sobel64f = np.absolute(sobelx64f)
#     sobel_8u = np.uint8(abs_sobel64f)
#     # plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
#     # plt.title('Original'), plt.xticks([]), plt.yticks([])
#     # plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
#     # plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
#     # plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
#     # plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
#
#     # plt.show()
#     return sobel_8u


def horizontal_distance(img, x, y):
    x1, x2 = 0, image_width - 1
    for i in range(y + 1, image_width - 1):
        if (img[x][i] == 0 and img[x][i + 1]):
            # return i-y+1
            x2 = i
            break
    for i in range(y - 1, 0, -1):
        if (img[x][i] == 0 and img[x][i - 1] == 0):
            x1 = i
            break;
    return x1, x2


def hdd(ulea, veda):
    print('hdd')
    hdd = np.zeros((image_height, image_width), np.uint8)
    for j in range(image_height):
        b = 0
        for i in range(image_width):
            if (ulea[j][i] and veda[j][i]):
                if (b <= i):
                    a, b = horizontal_distance(veda, j, i)
                if ((b-a)<25 and b > i):
                    hdd[j][i] = 0  # nand
                else:
                    hdd[j][i] = 255
            else:
                if (ulea[j][i] or veda[j][i]):
                    hdd[j][i] = 255  # nand
                else:
                    hdd[j][i] = 0  # no change
    return hdd

def rubr(points):
    print(points)
    e=list()
    s=list()
    # print(points)
    s.append(points[0])
    for k in range(1,len(points)):
        if(points[k]-points[k-1]>1):
            e.append(points[k-1])
            s.append(points[k])
    e.append(points[-1])
    # points=list(filter(lambda a: a!=0,points))
    # len2=len(points)
    print('first')
    print(s)
    print(e)
    for k in range(len(s)):
        if(e[k]-s[k]<4):
            e[k],s[k]=-1,-1

    e=list(filter(lambda a: a != -1,e))
    s=list(filter(lambda a: a!=-1,s))
    print('before marks')
    print(s)
    print(e)
    marks=np.zeros(len(s))
    for i in range(1,len(s)-1):
        if s[i+1]-e[i]>30:
            marks[i]-=1
        else:
            marks[i]+=1
        if s[i]-e[i-1]>30:
            marks[i]-=1
        else:
            marks[i]+=1
    print(marks)
    if len(marks)>2:
	    for i in range(1,len(marks)-1):
	        if(marks[i]==0 and marks[i+1]+marks[i-1]==0):
	            e[i],s[i]=-1,-1
	    if marks[1]!=2: e[0], s[0]=-1, -1
	    if marks[-2]!=2: e[-1], s[-1]=-1, -1;
	    e = list(filter(lambda a: a != -1, e))
	    s = list(filter(lambda a: a != -1, s))
    print('After marks')
    print(s)
    print(e)
    return s,e

def cre(img):
    print('cre')
    cr = np.full((image_height, image_width), 255, dtype=np.uint8)
    can_reg = []
    no_of_line = np.zeros([image_height])
    for j in range(0, image_height):
        for i in range(image_width):
            if (not img[j][i] and img[j][i - 1]) or (img[j][i] and not img[j][i - 1]):
                no_of_line[j] += 1
        if (no_of_line[j] > 15):
            # print(no_of_line[j], j)
            can_reg += [j]
    # print(can_reg)
    a, b = -15, can_reg[0]
    plt_reg = [(0, 0, 0)]
    for k in can_reg:
        if (a == -15):
            a = k
        if(k-b>=4 or k==can_reg[-1]):
            if(b-a>15):
                plt_reg += [(a, b, b-a)]
            a = -15
        b=k
    plt_reg.sort(key=lambda x:x[2],reverse=True)
    # print(plt_reg)
    plt_height=0
    x1=x2=int()
    s=e=list()
    cr = np.matrix.transpose(cr)
    for k in range(len(plt_reg)-1):
        x1, x2 = plt_reg[k][:2]
        if(plt_height> x2-x1):
            continue
        r1,r2=x1,x2
        blck_regn=list()
        for i in range(image_width):
            bprs = 0
            for j in range(x1, x2 + 1):
                if img[j][i] == 0:
                    bprs += 1
            if bprs >= 0.4 * (x2 - x1):
                blck_regn+=[i]
        s,e=rubr(blck_regn)
        plt_height=e[-1]-s[0]
        for l in range(len(s)):
            y1,y2=s[l],e[l]
            print("y1,y2",str(y1),str(y2))
            for i in range(y1,y2+1):
                cr[i][x1:x2] = 0
    cr = np.matrix.transpose(cr)
    # print(plt_reg)

    return cr,s[0],e[-1],r1,r2

def plt(r1,c1,r2,c2,img):
	plt=np.zeros((r2-r1+1,c2-c1+1),np.uint8)
	for i in range(c1,c2+1):
		for j in range(r1,r2+1):
			plt[j-r1][i-c1]=img[j][i]
	return plt

def characters(plt):
	plt = cv2.Canny(plt, 180, 200)
	plt, contours, hierarchy = cv2.findContours(plt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnt=contours[0]
	plate_height,plate_width=plt.shape[:2]
	area=plate_height*plate_width
	chars=[]
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)
		if w/h>0.3 and w/h<1 and w*h>area/30:
			chars=chars+[(x,y,w,h)]
			cv2.rectangle(plt, (x,y), (x+w,y+h), (255,0,255) ,-1)

	print(chars)
	return chars

# im = cv2.imread("C:/Users/iiitg/Desktop/python/h.jpg")
im = cv2.imread("C:/Users/iiitg/Desktop/Python/modified data set/IMG-20170321-WA0103.jpg")
print(im.shape)
# img=cv2.GaussianBlur(img,(3,3),0)
img=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
image_height,image_width = img.shape
s = image_width / 8
t = 0.15

th = at(img)
th2 = ulea(th)
inv = inv(th2)
ve = ved(inv)
hdd = hdd(inv, ve)
cr,c1,c2,r1,r2 = cre(hdd)
cv2.rectangle(im,(c1,r1),(c2,r2),(255,0,0),1)

plt = plt(r1,c1,r2,c2,img)
chars = characters(plt)

# ve=sobel(th2)
# ve=cv2.Canny(th2,200,240)

# plt.subplot(3, 2, 1), plt.imshow(img, cmap='gray')
# plt.subplot(3, 2, 2), plt.imshow(th, cmap='gray')
# plt.subplot(3, 2, 3), plt.imshow(th2, cmap='gray')
# plt.subplot(3, 2, 4), plt.imshow(ve, cmap='gray')
# plt.subplot(3, 2, 5), plt.imshow(hdd, cmap='gray')
# plt.subplot(3, 2, 6), plt.imshow(cr, cmap='gray')
#
# plt.show()
# cv2.imshow("Canny",ca)
# print(c1,r1)
# print(c2,r2)
cv2.imshow("original", im)
cv2.imshow("threshold", th)
cv2.imshow("inv", inv)
cv2.imshow("ulea", th2)
cv2.imshow("veda", ve)
cv2.imshow("hdd", hdd)
cv2.imshow("crs", cr)
cv2.imshow("plate", plt)
cv2.waitKey(0)
cv2.destroyAllWindows()
