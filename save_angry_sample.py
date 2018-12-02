from basic_image_tool import *

img = give_me_the_original('angry.png')
sample = gimme_angry_sample(img)
mouthless = remove_angry_mouth(img)
fig, ax = plt.subplots(figsize=(7,7))
fig.add_subplot(1,2,1)
plt.imshow(img, alpha=0.2)
plt.imshow(sample)
fig.add_subplot(1,2,2)
plt.imshow(mouthless)
ax.axis('off')
plt.savefig('img/edits/angry_mouth_sample.png', bbox_inches='tight')
