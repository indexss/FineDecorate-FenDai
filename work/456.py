import cv2
from matplotlib import pyplot as plt

ps_source = cv2.imread("C:/Users/pages/Desktop/home/aistudio/work/ps_source_face.png")
ps_ref = cv2.imread('C:/Users/pages/Desktop//home/aistudio/work/ref/ps_ref.png')
transfered_ref_ps_ref = cv2.imread('C:/Users/pages/Desktop//home/aistudio/PaddleGAN/output/transfered_ref_ps_ref.png')

plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Origanl')
plt.imshow(ps_source[:, :, ::-1])
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 2)
plt.title('MakeUp')
plt.imshow(ps_ref[:, :, ::-1])
plt.xticks([])
plt.yticks([])

plt.subplot(1, 3, 3)
plt.title('After')
plt.imshow(transfered_ref_ps_ref[:, :, ::-1])
plt.xticks([])
plt.yticks([])

save_path='C:/Users/pages/Desktop/home/aistudio/work/output.png'
plt.savefig(save_path)

plt.show()