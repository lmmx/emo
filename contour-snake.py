import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import sys

from basic_image_tool import give_me_the_original

def save_contours(img_name, gamma):
    img = give_me_the_original('birdog.jpg')
    img = rgb2gray(img)

    s = np.linspace(0, 2*np.pi, 200)
    x = 310 + 70*np.cos(s)
    y = 230 + 70*np.sin(s)
    init = np.array([x, y]).T

    snake = active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=10, gamma=float(gamma))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])
    figpath = 'img/edits/'
    figname = f"{img_name}_gamma-{str(gamma)[0:6].replace('.',',')}.jpg"
    plt.savefig(figpath + figname)
    print(f"Saved {figname}")
    return

gamma_range = np.linspace(0.0001,0.01,20)
for i in gamma_range:
    save_contours('birdog', i)
