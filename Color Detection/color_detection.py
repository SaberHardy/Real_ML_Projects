import cv2
import pandas as pd

clicked=False
img_path = 'google.jpg'
#Reading image with opencv
img = cv2.imread(img_path)

#read csv file
index= ["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv',names=index,header=None)
# Set a mouse callback event on a window
"""
First, we created a window in which the input image will display.
Then, we set a callback function which will be called when a
mouse event happens.
"""

def draw_function(event,x,y,flag,key):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b,g,r,xpos,ypos,clicked
        clicked = True
        xpos=x
        ypos = y
        b,g,r = img[y,x]
        b = int(b)
        g = int(g)
        r = int(r)


# Calculate distance to get color name
#return us the color name from RGB values

#Our distance is calculated by this formula:

#d = abs(Red – ithRedColor) + (Green – ithGreenColor) + (Blue – ithBlueColor)
def getColorName(R,G,B):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i,"R"]))+abs(G-int(csv.loc[i,"G"]))+\
            abs(B-int(csv.loc[i,"B"]))
        if (d<=minimum):
            minimum=d
            name_pic = csv.loc[i,"color_name"]

    return name_pic


cv2.namedWindow('Image Detection')
cv2.setMouseCallback('Image Detection',draw_function)
#Display image on the window

while(1):
    cv2.imshow("Image Detection",img)
    if (clicked):
        cv2.rectangle(img,(20,20),(750,60),(b,g,r),-1)
        text = getColorName(r,g,b) + " R="+str(r)+ " G="+ str(g)+' B='+str(b)
        cv2.putText(img,text,(50,50),2,0.8,(0,0,0),2,cv2.LINE_AA)
        print(f"Selected successfully!--> it\'s  {getColorName(r,g,b)}")
        clicked=False

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()



