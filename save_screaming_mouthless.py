from basic_image_tool import *
img = give_me_the_original('screaming_in_fear.png')
mouthless = remove_mouth(img)
plt.imshow(mouthless)
plt.axis('off')
plt.savefig('img/edits/screaming_in_fear_mouthless.png', bbox_inches='tight')
