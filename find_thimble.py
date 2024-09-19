"""
Sample code for finding the Lefschetz thimble using the constant phase contour method 

Reference: 
Acquiring the Lefschetz thimbles: efficient evaluation of the diffraction integral for lensing in wave optics
Xun Shi 2024
https://doi.org/10.1093/mnras/stae2127
"""

import numpy as np
import matplotlib.pyplot as plt

import PL_functions as plf

from datetime import datetime
startTime = datetime.now()

# lensing parameters: kappa (lens amplitude), beeta (source location) 
kappa = -500; beeta = 100  # parameters for Jow+23 Figure A2

outpath = '/Users/xun/Desktop/ims/'

# lens shape -- 1D rational lens
def psi(x, kappa):
    return kappa * 1. / (1. + x**2) 

def func(x):
    """
    i phi / nu
    exponent4integration_over_nu
    separate into exponent = h + iH
    """
    psi_ = psi(x, kappa) 
    res = (x - beeta)**2 / 2. - psi_
    return res * 1j


poly_coeff_phidot = [1, -beeta, 2, -2*beeta, 2*kappa+1, -beeta]

images = np.roots(poly_coeff_phidot) 
print(kappa, beeta, images)

images.sort()
bcode_real_image = np.where(images.imag==0)[0]

poles = np.roots([1, 0, 1])
poles.sort()

dx = 0.01 # grid resolution  
r_crossing = dx * 3 # thresshold for identifying crossing 

# get xs, ys for H and h
xs = plf.get_xs_3level(beeta, np.array(list(set(images.real))), XLIMFF=1, XLIMF=15, XLIM=500, deltaff=dx, deltaf=0.1, delta=5)
ys = plf.get_xs_3level(beeta, np.array(list(set(images.imag))), XLIMFF=1, XLIMF=15, XLIM=500, deltaff=dx, deltaf=0.1, delta=5)

xx, yy = np.meshgrid(xs, ys)
HH = plf.amp_imag(xx + 1j * yy, func)
hh = plf.amp_real(xx + 1j * yy, func)

H_images = list(set(plf.amp_imag(images, func)))

# get contour for each H value at images
contours_H = [] # contours with H values = H(image) 
for i, imag_i in enumerate(H_images): 
    contours_H_i = plf.get_contour_pieces(imag_i, xs, ys, func, images, poles, r_crossing) # 1.5 sometimes not enough
    contours_H = contours_H + contours_H_i


# get relevant contours (for all images with descending contours, keep those with some (ascending) contour crossing the real axis)
PL_thimble_candidates, dummy_images_effective = plf.get_thimble_pieces(contours_H, images)  

# from graph to path
path, G, thimble_end_codes = plf.find_path(PL_thimble_candidates, bcode_real_image, RETURN_MORE=True)

# from path, link thimbles
image_thimbles = plf.get_image_thimbles(path, images, PL_thimble_candidates)


print('time for getting thimbles', datetime.now() - startTime)

# show
fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = fig.add_axes([0.2, 0.6, 0.2, 0.2])
for c in image_thimbles:
    v = c['thimble']
    ax1.plot(v[:,0], v[:,1], 'k-', lw=0.2)
    ax2.plot(v[:,0], v[:,1], 'k-', lw=0.5)
ax1.set_xlim(v[:,0].min(), 400)
ax1.set_ylim(v[:,1].min(), v[:,1].max())
ax1.set_xlabel('Real(x)')
ax1.set_ylabel('Im(x)')
ax2.set_xlim(-5, 5)
ax2.set_ylim(-10, 5)
plt.show()




