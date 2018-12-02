from basic_image_tool import *
img = give_me_the_original('confounded.png')
mouthless = remove_confound_mouth(img)
plt.imshow(mouthless)
plt.show()
