# import the cv2 library
import cv2
import numpy as np

img = cv2.imread('/home/gabro/GrapheDetectProject/b74c9ce2-graphene_218641_bonds.png')
print(img.shape) # Print image shape
cv2.imshow("original", img)
 
# Cropping an image
cropped_image = img[18:51, 4:35] #img[start_row:end_row, start_col:end_col]

# Display cropped image
cv2.imshow("cropped", cropped_image)
 
# Save the cropped image
cv2.imwrite("Cropped Image.jpg", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()