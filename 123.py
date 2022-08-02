import cv2
import dlib
from matplotlib import pyplot as plt

cut_before_img="C:/Users/pages/Desktop/home/aistudio/work/HANDSOME.png"
cut_after_img="C:/Users/pages/Desktop/home/aistudio/work/ps_source_face.png"
img = cv2.imread(cut_before_img)
height, width = img.shape[:2]

face_detector = dlib.get_frontal_face_detector()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detector(gray, 1)

def get_boundingbox(face, width, height, scale=1.6, minsize=None):

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)

    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


if len(faces):
    face = faces[0]

x,y,size = get_boundingbox(face, width, height)
cropped_face = img[y-50:y+size,x:x+size]
cv2.imwrite(cut_after_img, cropped_face)

im1 = cv2.imread(cut_after_img)
im2 = cv2.resize(im1,(im1.shape[1],im1.shape[1]),)
cv2.imwrite(cut_after_img,im2)


before = cv2.imread(cut_before_img)
after = cv2.imread(cut_after_img)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('cut-before')
plt.imshow(before[:, :, ::-1])

plt.subplot(1, 2, 2)
plt.title('cut-after')
plt.imshow(after[:, :, ::-1])


plt.show()


