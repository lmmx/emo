from basic_image_tool import *

img = give_me_the_original('screaming_in_fear.png')
grad = remove_mouth(img, inspect_grad=True)
