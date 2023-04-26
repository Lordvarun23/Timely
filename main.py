import cv2

# Save image in set directory
# Read RGB image
img = cv2.imread('C:\\Users\\Public\\pythonProject\\FastapiDemo\\output.png')

# Output img with window name as 'image'
cv2.imshow('image', img)
