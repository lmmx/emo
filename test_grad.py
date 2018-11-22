from basic_image_tool import *

img = give_me_the_original('screaming_in_fear.png')
grad = remove_mouth(img, inspect_grad=True)

for k in grad.keys(): print(f'{k}:\tn: {len(grad[k])}\tmean distance: {np.around(np.mean(grad[k]), 2)},\trange: {np.around(np.ptp(grad[k]), 2)},\tmax: {np.around(np.max(grad[k]), 2)},\tmin: {np.around(np.min(grad[k]), 2)}')
