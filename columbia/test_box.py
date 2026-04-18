import cv2

img = cv2.imread("test_full_frame.jpg")
fh, fw = img.shape[:2]

TLx, TLy, width = 0.0, 48.0, 143.0

# 1. RED BOX: Absolute pixels (Annotations are raw pixel values)
cv2.rectangle(img, (int(TLx), int(TLy)), (int(TLx+width), int(TLy+width)), (0,0,255), 2)

# 2. GREEN BOX: Scaled from 640x360 (What we just tried)
px1, py1 = int((TLx / 640.0) * fw), int((TLy / 360.0) * fh)
px2, py2 = int(((TLx + width) / 640.0) * fw), int(((TLy + width) / 360.0) * fh)
cv2.rectangle(img, (px1, py1), (px2, py2), (0,255,0), 2)

# 3. BLUE BOX: Scaled from 720x480 (Common 4:3 SD video resolution)
px1, py1 = int((TLx / 720.0) * fw), int((TLy / 480.0) * fh)
px2, py2 = int(((TLx + width) / 720.0) * fw), int(((TLy + width) / 480.0) * fh)
cv2.rectangle(img, (px1, py1), (px2, py2), (255,0,0), 2)

cv2.imwrite("test_boxes.jpg", img)
print("Saved test_boxes.jpg! Open it and check the colors.")