from basic_image_tool import *

img = give_me_the_original('screaming_in_fear.png')
sample = gimme_sample(img)
fig, ax = plt.subplots(figsize=(7,7))
ax.imshow(sample)
ax.set_facecolor((1.0, 0.47, 0.42))
plt.show()
