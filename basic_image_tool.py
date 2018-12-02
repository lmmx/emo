import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from imageio import imread
from skimage.color import rgb2grey
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import polygon as DrawPolygon

def read_image(img_name):
    """
    Read an image file (.jpg) into a numpy array in which each entry is
    a row of pixels (i.e. ``len(game_img)`` is the image height in px.
    """
    #data_dir = Path(__file__).parent() / 'img'
    data_dir = Path('.') / 'img'
    #data_dir = '/home/louis/spin/m/components/opt/mac/img/'
    # - imread doesn't accept pathlib PosixPath objects?
    img = imread(data_dir / img_name)
    return img

def give_me_the_original(img_name):
    """
    Debugging/development: produce an original image
    """
    img = read_image(img_name)
    return img

def show_me_the_original(img_name):
    """
    Debugging/development: produce and display an original image
    """
    img = give_me_the_original(img_name)
    show_me_the_image(img)
    return img

def show_me_the_image(img):
    """
    Show an image using a provided pixel row array.
    """
    plt.imshow(img)
    plt.show()

def take_outline(img):
    """
    Take an outline of an image
    """
    # TODO edge detection, then apply (smooth?) alias fill/border
    show_me_the_image(img)
    return

def plot_img_and_hist(img, axes, bins=256):
    """
    Plot an image along with its histogram and cumulative histogram.
    """
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def pad_border(poly, pad=0.05, external=True):
    """
    Pad a border polygon `poly` (e.g. an active contour snake spline)
    by a given number of pixels (default = 5%, set by `pad`).
    """
    # Centroid of polygon i.e. centre of mass
    c = np.mean(poly, axis=0)
    if not external: pad *= -1
    padded = ((poly - c) * (1 + pad)) + c
    return padded

def v_ordered_grad(centre, sample):
    """
    Take a linear gradient by sampling values vertically via the
    centre point. This can create an ordered set for a gradient to
    subsequently be discerned from (e.g. for a radial gradient).
    """
    seen_px = []
    # for r, row in enumerate(sample):
    for y, row in enumerate(sample):
        if not np.any(row):
            # Ignore rows not in sample (i.e. all RGBA=[0,0,0,0])
            continue
        for offset in range(0, len(row) // 2):
            # Pick the pixel nearest to centre in the row
            x = centre[1] - offset
            centred_px = row[x]
            if np.any(centred_px):
                break
            if offset == 0:
                next
            # Otherwise take the offset to the right
            x = centre[1] + offset
            centred_px = row[x]
            if np.any(centred_px):
                break
        # centred_px must be a non-empty pixel at this point
        assert np.any(centred_px)
        if len(seen_px) == 0:
            seen_px.append((tuple(centred_px), (y,x)))
            next
        # Else slice out the seen RGBA values and compare current px
        seen_rgba = np.array(seen_px)[:,0]
        if not np.any([x == tuple(centred_px) for x in seen_rgba]):
            # Append a 2-tuple of (RGBA 4-tuple, YX 2-tuple)
            # i.e. the RGBA value and the location it was recorded at
            seen_px.append((tuple(centred_px), (y,x)))
    return seen_px

def radial_gradient(centre, sample):
    """
    Estimate linear RGB gradient using distance from the
    image centre for the points in a given sample region,
    which can then be used to fill a region with `grad_fill`.
    N.B. will ignore blank pixels (RGBA values of [0,0,0,0]).
    """
    # Get an ordered, non-degenerate list of gradient RGBA values
    grad = v_ordered_grad(centre, sample) # (vertically)
    # Calculate all distances from centre point in the sample
    dists = {}
    ## [np.linalg.norm(px - centre, ord=2) for px in sample]
    ################################################################
    ### UNCLEAR HOW TO ORDER THE FOLLOWING SO CHANGE TO ordered_grad
    ################################################################
    for px in grad:
        y, x = px[1]
        dists[tuple(px[0])] = np.linalg.norm([y,x] - centre, ord = 2)
    #for y, row in enumerate(sample):
    #    for x, px in enumerate(row):
    #        if not np.array_equal(px, [0,0,0,0]):
    #            # pixel is in sample, so check distance
    #            dist = np.linalg.norm([y,x] - centre, ord=2)
    #            if tuple(px) not in dists.keys():
    #                dists[tuple(px)] = []
    #            dists[tuple(px)].append(((y, x), dist))
    #return ordered, dists
    return dists

def grad_fill(centre, gradient, coord):
    """
    Calculate pixel according to distance from radial centre,
    using a dictionary of gradients with known distances from
    running the `radial_gradient` function on an image sample.
    """
    y, x = coord
    dist = np.linalg.norm([y,x] - centre, ord=2)
    # use distance to determine a pixel colour
    for rgba in gradient.keys():
        if dist > gradient[rgba]:
            last_rgba = rgba
            next
        else:
            return last_rgba
    # This should not be reached
    raise ValueError(f'RGBA values in gradient exhausted!')

def poly2pxmask(poly_coords, shape, xy=True):
    """
    Turn a polygon (numpy array of edge pixel coordinates) into
    a labelled pixel mask (masked with 1 on a background of 0).
    
    N.B. polygon coordinates should be supplied as (x, y) even
    though the `skimage.draw.polygon` function takes (y, x) or
    (r, c) parameters. If parameter `xy` is set to False, you
    may supply polygon coords as (y, x).
    """
    assert len(shape) == 2
    mask = np.zeros(shape, dtype=np.uint)
    # When xy = True (default), (r,c) = (1,0)
    c = 1 - int(xy)
    r = 1 - c
    rr, cc = DrawPolygon(poly_coords[:,r], poly_coords[:,c])
    mask[rr, cc] = 1
    return mask

#######################################################################
###                       SCREAMING IN FEAR                         ###
#######################################################################

def init_mouth(round=False):
    """
    Circle the general mouth region (reused for both contouring and
    gradient fill sampling). `round` parameter enforces integer
    coercion of the values to be used as pixel coordinates, not in
    reference to the shape (which is a circle).
    """
    # The circular init area from which the snake shrinks is
    # useful for gradient fill sampling, so function is reusable
    s = np.linspace(0, 2*np.pi, 400)
    x = 225 + 85*np.cos(s)
    y = 330 + 85*np.sin(s)
    init = np.array([x, y]).T
    if round:
        init = np.round(init).astype(np.uint)
    return init

def contour_mouth(img, visualise=False):
    """
    Run active contour model (snake) segmentation for screaming_in_fear
    emoji's mouth (then pad because the snake is too bendy).
    """
    init = init_mouth()
    # round coords as they will be used to mask image pixels
    snake = np.round(pad_border(active_contour(gaussian(img, 8),
                init, alpha=0.015, beta=1, gamma=0.001),
                pad = 0.08)).astype(int)
    if visualise:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()
    return snake

def gimme_sample(img, inspect_grad=False):
    # Get outline of mouth:
    mouth_poly = contour_mouth(img)
    # Turn outline of mouth into mask of pixel coordinates
    mouth_bitmask = poly2pxmask(mouth_poly, img.shape[0:2])
    centre = np.divide(img.shape[0:2], 2).astype(int)
    # Alternative: use top of mouth rather than emoji centroid...
    ymin = mouth_poly[mouth_poly[:,0] == np.min(mouth_poly[:,0])]
    mouth_top_centre = ymin[ymin[:,1] == np.min(ymin[:,1])][0]
    # Get area directly around the mouth for gradient sampling
    mouth_init_bitmask = poly2pxmask(init_mouth(), img.shape[0:2])
    # N.B. the following line produces a bool mask cf. a bit mask
    sample_mask = np.logical_xor(mouth_init_bitmask, mouth_bitmask)
    sample = np.copy(img)
    sample[~sample_mask] = [0,0,0,0]
    return sample

def remove_mouth(img, inspect_grad=False):
    # Get outline of mouth:
    mouth_poly = contour_mouth(img)
    # Turn outline of mouth into mask of pixel coordinates
    mouth_bitmask = poly2pxmask(mouth_poly, img.shape[0:2])
    centre = np.divide(img.shape[0:2], 2).astype(int)
    # Alternative: use top of mouth rather than emoji centroid...
    ymin = mouth_poly[mouth_poly[:,0] == np.min(mouth_poly[:,0])]
    mouth_top_centre = ymin[ymin[:,1] == np.min(ymin[:,1])][0]
    # Get area directly around the mouth for gradient sampling
    mouth_init_bitmask = poly2pxmask(init_mouth(), img.shape[0:2])
    # N.B. the following line produces a bool mask cf. a bit mask
    sample_mask = np.logical_xor(mouth_init_bitmask, mouth_bitmask)
    sample = np.copy(img)
    sample[~sample_mask] = [0,0,0,0]
    grad = radial_gradient(centre, sample)
    if inspect_grad:
        ranges = []
        for k in grad.keys():
            ranges.append(np.ptp(grad[k]))
            print(
            f'{k}\t{len(grad[k])}\t{np.around(np.mean(grad[k]), 2)}\t'
            + f'{np.around(np.ptp(grad[k]), 2)}\t'
            + f'{np.around(np.min(grad[k]), 2)}\t'
            + f'{np.around(np.max(grad[k]), 2)}')
        print(f'Mean range: {np.mean(ranges).astype(int)}')
        print(f'Range range: {np.ptp(ranges).astype(int)}')
        print('----------------------------------------------------------')
        print('----------------------------------------------------------')
        print('----------------------------------------------------------')
    #return grad
    mouthless = np.copy(img)
    for y, r in enumerate(mouth_bitmask.astype(bool)): 
        for x, c in enumerate(r):
            if c:
                mouthless[y,x] = grad_fill(centre, grad, (y,x))
    return mouthless
