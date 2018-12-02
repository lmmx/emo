from basic_image_tool import *

img = give_me_the_original('angry.png')
sample = gimme_angry_sample(img)
fig, ax = plt.subplots(figsize=(7,7))
ax.imshow(img, alpha=0.2)
ax.imshow(sample)
ax.set_facecolor((1.0, 0.47, 0.42))
plt.axis('off')
plt.savefig('img/edits/angry_mouth_sample.png', bbox_inches='tight')
