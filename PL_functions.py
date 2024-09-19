"""
functions for identifying pl thimbles using the facts:
1. PL thimbles belong to contours with the same H value as the images (including imaginary images)
2. PL thimbles have h smaller than h at the associated image
3. ...

STEPS:
1. get the relavant H values (= those at images) 
2. get the contour associated with each H value
3. cut the contour at image, pole, and bounary

bcode: number recording which image/pole/boundary a crossing is associated with. 0 to nimage-1: images; nimage to nimage+npoles-1: poles; -1: bottom/left boundary, -2: top/right boundary

func: i * phase_function == h + i * H

Xun Shi 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import splrep, splev

from mergeIntervals import mergeIntervals, excludeIntervals

def bcodes_assigner(TYPE, i, nimage=None):
    """
    TYPE: 
        LB (left/bottom boundary) 
        RT (right/top boundary)
        IMAGE (image)
        POLE (pole)
    """
    if TYPE=='LB': return -1
    elif TYPE=='RT': return -2
    elif TYPE=='IMAGE': return i
    elif TYPE=='POLE': return nimage + i

def amp_imag(p, func):
    """
    H of func
    """
    return func(p).imag

def amp_real(p, func):
    """
    h of func
    """
    return func(p).real

def v2c(vs): # (,2) vec to complex numbers
    return vs[:,0] + 1j * vs[:,1]

def c2v(cs): # complex numbers to (,2) vec
    return np.array([cs.real, cs.imag]).T

def dist(v, vi):
    return ((v[:,0] - vi[0])**2 + (v[:,1] - vi[1])**2)**0.5

def if_cross_y0(v): 
    """
    check if thimble v crosses the real axis
    """
    y = v[:,1]
    if (y.max()>0) & (y.min()<0):
        return 1
    else:
        return 0

def get_xs_ys(beeta, XLIMF=10, XLIM=1000, deltaf=0.02, delta=1):
    """
    get x, y sample points for getting contour for thimble extraction
    refined resolution around lens (x=0, y=0) and source (x=beeta, y=0)  
    """
    integrands_fine = mergeIntervals([[-XLIMF, XLIMF], [-XLIMF+beeta, XLIMF+beeta]])
    integrands_rough_exclusive = excludeIntervals([[-XLIM, XLIM+beeta]], integrands_fine)
    xs1 = np.hstack([np.arange(interv[0], interv[1], deltaf) for interv in integrands_fine])
    xs2 = np.hstack([np.arange(interv[0], interv[1], delta) if delta<(interv[1]-interv[0]) else np.array([interv[0], (interv[0]+interv[1])/2.]) for interv in integrands_rough_exclusive])
    xs = list(set(np.hstack([xs1, xs2])))
    xs.sort()
    xs = np.array(xs)
    if XLIM<=XLIMF:
        ys = np.arange(-XLIMF, -XLIMF, deltaf)
    else:
        ys = np.hstack([np.arange(-XLIM, -XLIMF, delta), np.arange(-XLIMF, XLIMF, deltaf), np.arange(XLIMF, XLIM, delta)])
    return xs, ys

def get_xs_3level(beeta, images, XLIMFF=1, XLIMF=10, XLIM=1000, deltaff=0.01, deltaf=0.1, delta=1):
    """
    finest around images; fine around 0 (lens) and beeta; otherwise coarse
    """
    integrands_finest = mergeIntervals([[-XLIMFF+x_im, x_im+XLIMFF] for x_im in images])
    integrands_fine = mergeIntervals([[-XLIMF+x_im, x_im+XLIMF] for x_im in [0, beeta]])
    integrands_fine = excludeIntervals(integrands_fine, integrands_finest)
    integrands_rough_exclusive = excludeIntervals([[-XLIM, XLIM+beeta]], integrands_fine)
    xs0 = np.hstack([np.arange(interv[0], interv[1], deltaff) for interv in integrands_finest])
    xs1 = np.hstack([np.arange(interv[0], interv[1], deltaf) for interv in integrands_fine])
    xs2 = np.hstack([np.arange(interv[0], interv[1], delta) if delta<(interv[1]-interv[0]) else np.array([interv[0], (interv[0]+interv[1])/2.]) for interv in integrands_rough_exclusive])
    xs = list(set(np.hstack([xs0, xs1, xs2])))
    xs.sort()
    xs = np.array(xs)
    xs = xs[np.where(np.diff(xs)>1e-5)]
    return xs

def get_contours_H(imag_i, xs, ys, HH):
    """
    get contour at a certain H = imag_i
    xx, yy = np.meshgrid(xs, ys)
    HH = amp_imag(xx + 1j * yy, func)
    """
    fig = plt.figure()
    cs = plt.contour(xs, ys, HH, [imag_i]) # get contour with H = H_i
    paths = cs.collections[0].get_paths()
    plt.close(fig) 
    return paths

def get_crossing_points(v, x, y, r_crossing):
    """
    #get where contour v is closest to (x, y)
    #return if the closest distance is smaller than r_crossing
    get where contour v crosses (x, y) within r_crossing
    // do not use argmin since this can lead to contour pieces crossing (x, y)
    """
    dists = dist(v, np.array([x, y]))
    #i_crossing = np.argmin(dists)
    i_crossing = np.where(dists < r_crossing)[0]
    return i_crossing 

def get_info_crossing_imagei(v, images_i, images, xs, ys, r_crossing):
    """
    determine where contour v crosses images (i.e. with distance < r_crossing)
    images_i: if known, images with H being the one that was used to generate contour v; otherwise just input images 
    """
    info_crossings = [] # (2, Ncutting points) locate where to cut the contour
    for image in images_i:
        bcode = np.where(images==image)[0] # number recording which image/pole/boundary this crossing is associated with. 0 to nimage-1: images; nimage to nimage+npoles-1: poles; -1: bottom/left boundary, -2: top/right boundary
        crossings_image = np.where(dist(v, c2v(image)) < r_crossing)[0]  
# where path crosses this image // here cannot use 'get_crossing_point' since one contour may cross an image for multiple times
        if len(crossings_image) > 0:
            info_crossings_image = np.vstack([crossings_image, np.ones(len(crossings_image)) * bcode]) # (2,) array for recording crossing location and the image crossed
            info_crossings.append(info_crossings_image)
    if len(info_crossings)==0:
        info_crossings = np.array(info_crossings)
    else:
        info_crossings = np.hstack(info_crossings)
        info_crossings = info_crossings[:,info_crossings[0].argsort()].astype('int')
        #dx_min = np.diff(xs).min()
        dx_min = np.max([np.diff(xs)[0], np.diff(xs)[-1], np.diff(ys)[0], np.diff(ys)[-1]])
        #print('dx_min', dx_min)
        #if ((v[0][0] not in [xs.min(), xs.max()]) and (v[0][1] not in [ys.min(), ys.max()])):
        if np.min([abs(v[0][0] - xs.min()), abs(v[0][0] - xs.max())]) > (dx_min/2.) and np.min([abs(v[0][1] - ys.min()), abs(v[0][1] - ys.max())]) > (dx_min/2.):
# not close to boundary, i.e. closed path
            ## get location of an image and put it in the beginning
            dists = dist(v[info_crossings[0]], c2v(images[info_crossings[1,0]]))
            crossing1 = info_crossings[0][np.array(dists).argmin()]
            v = np.roll(v, -crossing1, axis=0) # shift image to beginning of path
            info_crossings[0] = (info_crossings[0] - crossing1) % len(v)
            info_crossings = info_crossings[:,info_crossings[0].argsort()]
            if v.shape[0]-1 not in info_crossings[0]: # both start and end of the loop need to be marked
                info_crossings = np.hstack([info_crossings, np.array([v.shape[0]-1, info_crossings[1,0]])[:,np.newaxis]])
    return v, info_crossings

def get_info_crossing_pole_boundary(v, xs, ys, images, poles=[], r_crossing=None):
    """
    determine where contour v crosses boundary, poles (i.e. with distance < r_crossing)
    """
    dx_min = np.max([np.diff(xs)[0], np.diff(xs)[-1], np.diff(ys)[0], np.diff(ys)[-1]])
    #print('dx_min', dx_min)
    # add the poles
    if len(poles) > 0:
        ll = [get_crossing_points(v, pole.real, pole.imag, r_crossing) for pole in poles]
        info_crossings_poles = np.hstack([np.vstack([ll[ipole], np.ones(len(ll[ipole]))*(ipole+images.size)]) for ipole in range(len(poles))])
        info_crossings = info_crossings_poles
    else:
        info_crossings = np.array([[],[]])
    # add the boundaries
    if 0 not in info_crossings[0]:
        if np.min([abs(v[0][0] - xs.min()), abs(v[0][1] - ys.min())]) < (dx_min/2.):  # reaching the boundary
            bcode = -1 # for bottom-left boundary
        elif np.min([abs(v[0][0] - xs.max()), abs(v[0][1] - ys.max())]) < (dx_min/2.):
            bcode = -2 # for upper-right boundary
        else:
            bcode = -3 # check for numerical problem
        if bcode != -3:
            info_crossings = np.hstack([np.array([0,bcode])[:,np.newaxis], info_crossings])
    if v[:,0].size-1 not in info_crossings[0]:
        if np.min([abs(v[-1][0] - xs.min()), abs(v[-1][1] - ys.min())]) < (dx_min/2.):
            bcode = -1 # for bottom-left boundary
        elif np.min([abs(v[-1][0] - xs.max()), abs(v[-1][1] - ys.max())]) < (dx_min/2.):
            bcode = -2 # for upper-right boundary
        else:
            bcode = -3 # check for numerical problem
        if bcode != -3:
            info_crossings =  np.hstack([info_crossings, np.array([v[:,0].size-1, bcode])[:,np.newaxis]])
    info_crossings = info_crossings.astype('int')
    return info_crossings


def get_contour_pieces(imag_i, xs, ys, func, images, poles=[], r_crossing=None):
    """
    get contour at a certain H = imag_i (for some image), cut at crossing images_i, poles, bounaries
    """
    xx, yy = np.meshgrid(xs, ys)
    HH = amp_imag(xx + 1j * yy, func)
    if r_crossing==None: r_crossing = np.diff(xs).mean() * 2 #1.5
    dicts_imag_i = []
    imag_images = amp_imag(images, func)
    list_setH = list(set(imag_images))
    images_i = images[np.where(imag_images==imag_i)]
    fig = plt.figure()
    cs = plt.contour(xs, ys, HH, [imag_i]) # get contour with H = H_i
    paths = cs.collections[0].get_paths() # extract contour paths
    plt.close(fig)
    for p in paths:
        v = p.vertices # contour
        #### cut contour into pieces starting/end at image, pole or boundary
        v, info_crossings_images = get_info_crossing_imagei(v, images_i, images, xs, ys, r_crossing)
        if len(info_crossings_images) > 0: # contours not connecting to images are discarded
            info_crossings_pole_boundary = get_info_crossing_pole_boundary(v, xs, ys, images, poles, r_crossing) 
            info_crossings = np.hstack([info_crossings_images, info_crossings_pole_boundary])
            info_crossings = info_crossings[:,info_crossings[0].argsort()]
            #### done with identifying cutting points
            #### now cut the contours and record each piece
            for isect in range(len(info_crossings[0])-1):
                if info_crossings[0][isect+1] > (info_crossings[0][isect]+1): # length > 1
                    v_isect = v[info_crossings[0,isect]:info_crossings[0,isect+1],:]
                    descend = 1 if amp_real(v_isect[:,0]+v_isect[:,1]*1j, func).max() <= amp_real(images_i, func).min() else 0 # maximum h on the contour piece has to be smaller than the minimum h of all images lying on the whole contour
                    print(descend, 'end_codes:', [info_crossings[1,isect], info_crossings[1,isect+1]], 'len:', len(v_isect))
                    dict_imag = {'vertices': v_isect, 'H': imag_i, 'i_H': list_setH.index(imag_i), 'image': images_i, 'end_codes': [info_crossings[1,isect], info_crossings[1,isect+1]], 'descend': descend}
                    dicts_imag_i.append(dict_imag)    
    return dicts_imag_i

def get_thimble_candidates(contours_H):
    """
    from all contour pieces, select those with 'descend'==1 as thimble candidates
    now replaced by 'get_thimble_pieces'
    """
    PL_thimble_candidates = []
    for dict_imag in contours_H:
        if dict_imag['descend']==1:
            PL_thimble_candidates.append(dict_imag)
    return PL_thimble_candidates

def get_thimble_pieces(contours_H, images):
    """
    from all contour pieces, select those that should lie on the thimble
    using the criteria: (1) contour piece is descending (2) contour piece is linked to an image connected to some contour piece that crosses the real axis  
    """
    # select images that are connected to some descending contour piece
    PL_thimble_candidates = get_thimble_candidates(contours_H)
    i_selected_images = np.array(list(set([i for p in PL_thimble_candidates for i in p['end_codes']])))
    i_selected_images = i_selected_images[np.where((i_selected_images>=0) & (i_selected_images<images.size))]
    # add real images to the final image list
    i_selected_images_final = list(np.where(images.imag==0.)[0])
    # check the imaginary images in this list, see if they have an ascending contour piece that crosses the real axis
    i_imag_image_candidates = i_selected_images[np.where(images[i_selected_images].imag!=0)]
    for i_imag_image in i_imag_image_candidates:
        contours_imag_image = [contours_H_i for contours_H_i in contours_H if i_imag_image in contours_H_i['end_codes']] # (4) 2 descending + 2 ascending contour pieces lined to this image
        if_crossing_y0 = [if_cross_y0(contour_i['vertices']) for contour_i in contours_imag_image] # four 0s for unrelevant imaginary image; three 0s and one 1 for relevant ones 
        if 1 in if_crossing_y0:
            i_selected_images_final.append(i_imag_image) 
    PL_thimble_pieces = [thimble_candidate for thimble_candidate in PL_thimble_candidates if bool(set(thimble_candidate['end_codes']) & set(i_selected_images_final))] # len: 2 * len(i_selected_images_final) 
    return PL_thimble_pieces, images[i_selected_images_final]

def link_thimble(path, PL_thimble_candidates, thimble_end_codes=None, SEPARATE=False):
    """
    connect vertices in PL_thimble_candidates accordoing to path
    if SEPARATE=False: return whole thimble; otherwise, return the pieces in right order
    """
    if thimble_end_codes==None: thimble_end_codes = [dict_imag['end_codes'] for dict_imag in PL_thimble_candidates] # bcode of the two ends for each thimble piece
    indices_thimble_end_codes_4path = []
    for inode in range(len(path) -1):
        ed1 = path[inode]
        ed2 = path[inode+1]
        try:
            indice_thimble = thimble_end_codes.index([ed1,ed2])
            flip = 0
        except:
            indice_thimble = thimble_end_codes.index([ed2,ed1])
            flip = 1
        indices_thimble_end_codes_4path.append([indice_thimble, flip])

    thimble_path = []
    for item in indices_thimble_end_codes_4path:
        thimble_piece = PL_thimble_candidates[item[0]]['vertices']
        if item[1] == 1: thimble_piece = thimble_piece[::-1]
        thimble_path.append(thimble_piece)
    if SEPARATE:
        return thimble_path
    else:
        thimble_path = np.vstack(thimble_path)
        return thimble_path

def integrate_vy_dvx(v, IF_ABSY=False):
    """
    get area of thimble from x axis in the complex plane.
    for path selection
    """
    vx = v[:,0]
    vy = v[:,1]
    if IF_ABSY: vy = abs(vy)
    dvx = np.diff(vx)
    vy_mid = (vy[1:] + vy[:-1]) / 2.
    return (vy_mid * dvx).sum()

def check_numbder_order(ll, numbers):
    """
    if some of numbers are in list ll, check if they are sorted as in list numbers (from small to large)
    return 0 only when len(numbers)>=2 and some of the numbers are on wrong order in ll
    caution: this replies on that the images are sorted by their location  
    """
    index_included = [ll.index(n) for n in numbers if n in ll]
    #print(index_included)
    res = 1 if index_included == sorted(index_included) else 0
    return res

def get_elements_set(lists, except_nums=[-1]):
    res = [i for i in l for l in lists if i not in except_nums]
    return set(res)

def if_clockwise(path_candidate, PL_thimble_candidates):
    """
    integration over closed cycle
    clockwise: positive
    counter-clockwise: negative
    """
    cycle_candidate = path_candidate+[path_candidate[0]]
    v = link_thimble(cycle_candidate, PL_thimble_candidates) # (Nstep, 2) 
    v_cycle = np.vstack([v, v[0][np.newaxis,:]]) # link two '-1's 
    cycle_integral = integrate_vy_dvx(v_cycle)
    return 1 if cycle_integral > 0 else 0 

def get_area_to_real_axis(path_candidate, PL_thimble_candidates, EXTRAPOLATE=False):
    """
    get the aera of path_candidate to the real axis
    as path selection criterion
    if EXTRAPOLATE==True: extrapolate the paths so that they all start at the same x
    """
    v = link_thimble(path_candidate, PL_thimble_candidates) # (Nstep, 2)
    area = integrate_vy_dvx(v, IF_ABSY=True)
    return area

def extrapolate_v_path(v, xmin, xmax, npoints=5):
    """
    extrapolate v to cover the whole [xmin, xmax]
    """
    vxs = v[:,0]
    vys = v[:,1]
    if xmin < vxs[0]: 
        vxs_ = [xmin] + list(vxs[:npoints])
        ymin = splev(np.array(vxs_), splrep(vxs[:npoints], vys[:npoints]))[0]
        v = np.vstack([np.array([xmin, ymin])[np.newaxis,:], v])
    if xmax > vxs[-1]: 
        vxs_ = list(vxs[-npoints:]) + [xmax]
        ymax = splev(np.array(vxs_), splrep(vxs[-npoints:], vys[-npoints:]))[-1]
        v = np.vstack([v, np.array([xmax, ymax])[np.newaxis,:]])
    return v 
 
def select_min_area(path_candidates, PL_thimble_candidates, EXTRAPOLATE=True):
    v_paths = [link_thimble(path_candidate, PL_thimble_candidates) for path_candidate in path_candidates]
    xmin = np.min([v_path[0][0] for v_path in v_paths])
    xmax = np.max([v_path[-1][0] for v_path in v_paths])
    if EXTRAPOLATE:
        v_paths = [extrapolate_v_path(v, xmin, xmax) for v in v_paths]   
    areas = [integrate_vy_dvx(v, IF_ABSY=True) for v in v_paths]
    isel_ = np.array(areas).argmin()
    return path_candidates[isel_] 

def if_separate_lists(list_of_lists, except_nums=[-1]):
    """
    if all l in list_of_lists have no shared elements except for those in except_nums
    """   
    flattened_list = [i for l in list_of_lists for i in l]
    shared = set([x for x in flattened_list if flattened_list.count(x) > 1]) 
    real_shared = [x for x in shared if x not in except_nums]
    res = 1 if len(real_shared)==0 else 0
    return res 

def get_non_repetitive_groups(lists, except_nums=[-1]):
    """
    get groups of lists without repetitive elements (except elements in except_nums)
    letting them include as many elements as possible
    """
    n_list = len(lists)
    groups = [[l] for l in lists] # add groups of 1 lists
    for nlist_in_group in range(1,n_list): # add groups with more number of lists 
        groups_w_nlist = [group for group in groups if len(group)==nlist_in_group]
        for i in range(len(groups_w_nlist)):
            for j in range(n_list):
                new_group = groups_w_nlist[i] + [lists[j]]
                if if_separate_lists(new_group, except_nums=except_nums)==1:  
                    groups.append(new_group) 
    return groups

def link_cycles_in_groups(groups, PL_thimble_candidates): # for -1
    cycle_paths = []
    for group in groups:
        thimble_end_codes = [dict_imag['end_codes'] for dict_imag in PL_thimble_candidates]
        i_thimbles = [thimble_end_codes.index([path[0], path[1]]) for path in group]  
        x_v_start_thimbles = [PL_thimble_candidates[i_thimble]['vertices'][0][0] for i_thimble in i_thimbles] # x location of beginning of path
        if x_v_start_thimbles == sorted(x_v_start_thimbles): # only when cycles are placed in the right order 
            cycle_paths.append([i for path in group for i in path]) 
    return cycle_paths 

def link_cycles_in_groups2(groups, PL_thimble_candidates): # for -2
    cycle_paths = []
    for group in groups:
        thimble_end_codes = [dict_imag['end_codes'] for dict_imag in PL_thimble_candidates]
        i_thimbles = [thimble_end_codes.index([path[-1], path[0]]) for path in group]
        x_v_start_thimbles = [PL_thimble_candidates[i_thimble]['vertices'][0][0] for i_thimble in i_thimbles] # x location of beginning of path
        if x_v_start_thimbles == sorted(x_v_start_thimbles): # only when cycles are placed in the right order
            cycle_paths.append([i for path in group for i in path])
    return cycle_paths

def find_path(PL_thimble_candidates, bcode_real_image, RETURN_MORE=False):
    """
    decomposing graph G (composed from descending contour pieces)
    into a short path from bottom-left to upper-right bounday (bcode -1 to -2)
    and (multiple) cycles (bcode -1 to -1)
    then determine which cycle(s) to add to the path
    choose from alternative cycles: which path has smaller abs integrated area over the real axis x.

    ** when input PL_thimble_pieces instead of PL_thimble_candidates, there should be no alternative cycles.
    """
    thimble_end_codes = [dict_imag['end_codes'] for dict_imag in PL_thimble_candidates]
    G = nx.Graph()
    G.add_edges_from(thimble_end_codes) 
    list_short_path = list(nx.all_shortest_paths(G, -1, -2)) # direct pathfrom left-bottom boundary to upper-right boundary
    if len(list_short_path) == 1:
        short_path = list_short_path[0]
    else:
        print('check: multiple short paths from -1 to -2', list_short_path)    
    #if len(np.intersect1d(bcode_real_image, short_path)) == len(bcode_real_image): # all real images included in short path
    #    print('all real images are in short path')
    #    path = short_path
    if 1: #else: 
        # get cycles: starting and ending with bcode=-1 
        H = G.to_directed()
        simple_cycles = list(nx.simple_cycles(H))
        cycles_1 = [np.array(cycle) for cycle in simple_cycles if ((-1 in cycle) & (len(cycle)>2))] # remove false cycles with only 2 nodes and those not including -1
        cycles_2 = [np.array(cycle) for cycle in simple_cycles if ((-2 in cycle) & (len(cycle)>2))] # remove false cycles with only 2 nodes and those not including -2
        if len(cycles_1) > 0: 
            cycles_1 = [list(np.roll(cycle, -np.where(cycle==-1)[0][0])) for cycle in cycles_1] # cycles starting at bottom-left boundary -1
            #cycles_1 = [cycle for cycle in cycles_1 if check_numbder_order(cycle, bcode_real_image)==1] # remove cycles that include real images in the wrong order
            cycles_1 = [cycle for cycle in cycles_1 if if_clockwise(cycle, PL_thimble_candidates)==1] # include only clockwise cycles
            groups = get_non_repetitive_groups(cycles_1, except_nums=[-1]) # get all cycle combinations without repetitive image/poles
            cycle_paths1 = link_cycles_in_groups(groups, PL_thimble_candidates) # select composite cycles and link them
            cycle_paths1 = cycle_paths1 + [[]] # add no-cycle option
        else:
            cycle_paths1 = [[]] 

        if len(cycles_2) > 0:
            cycles_2 = [list(np.roll(np.roll(cycle, -np.where(cycle==-2)[0][0]), -1)) for cycle in cycles_2] # cycles ends at top-right boundary -2
            cycles_2 = [cycle for cycle in cycles_2 if if_clockwise(cycle, PL_thimble_candidates)==0] # include only anti-clockwise cycles
            groups = get_non_repetitive_groups(cycles_2, except_nums=[-2]) # get all cycle combinations without repetitive image/poles
            cycle_paths2 = link_cycles_in_groups2(groups, PL_thimble_candidates) # select composite cycles and link them
            cycle_paths2 = cycle_paths2 + [[]] # add no-cycle option
        else:
            cycle_paths2 = [[]]

        path_candidates = [cycle_path1 + short_path + cycle_path2 for cycle_path1 in cycle_paths1 for cycle_path2 in cycle_paths2]
        path = select_min_area(path_candidates, PL_thimble_candidates, EXTRAPOLATE=True)
 
        #bcode_real_image_in_cycle = [bc for bc in bcode_real_image if bc not in short_path] # real images in cycles
        #len_intersect = [np.intersect1d(bcode_real_image_in_cycle, ll).size for ll in cycles_1]
        #if np.max(len_intersect) != len(bcode_real_image_in_cycle):
        #    print('real images are spread in more than 1 cycles')
        #    print('need manual selection of paths')
        #    cycles_1 = [cycles_1[i] for i in range(len(cycles_1)) if len_intersect[i]>0]
        #else: # one cycle can contain all cycle images
        #    cycles_1 = [cycles_1[i] for i in range(len(cycles_1)) if len_intersect[i]==np.max(len_intersect)] # select cycles with max number of real images
        #    E_candidates = [integrate_vy_dvx(link_thimble(path_candidate, PL_thimble_candidates)) for path_candidate in cycles_1]
        #    path_cycle = cycles_1[abs(np.array(E_candidates)).argmin()]
        #    path = path_cycle + short_path
    if RETURN_MORE:
        return path, G, thimble_end_codes
    else:
        return path

def integrate(simplices, nu, func, WITHOUTNORM=False):
    #vals = np.exp(func(simplices) * nu)
    z_mid = (simplices[:-1] + simplices[1:]) / 2.
    vals = np.exp(func(z_mid) * nu)
    stepsizes = np.diff(simplices)
    #res = (vals[:-1] + vals[1:]) / 2. * stepsizes  # Trapezoidal rule
    res = vals * stepsizes
    if WITHOUTNORM:  
        E = res.sum()
    else:
        E = res.sum() * np.sqrt(nu / np.pi / 2j) # !! 2
    return E

def get_effective_images(path, images):
    """
    given path, identify images that contribute
    based on bcode rule!
    """
    i_effective_images = np.array(path)
    i_effective_images = i_effective_images[np.where((i_effective_images>=0) & (i_effective_images < images.size))]
    effective_images = images[i_effective_images]
    return effective_images

def get_image_thimbles(path, images, PL_thimble_candidates):
    """
    given path, from thimble candidates form complete thimble, then split thimble into pieces each associated with an image
    code in path based on bcode rule!
    """
    i_effective_images = np.array(path)
    i_effective_images = i_effective_images[np.where((i_effective_images>=0) & (i_effective_images < images.size))]
    
    thimble_paths = link_thimble(path, PL_thimble_candidates, SEPARATE=True)
    image_thimbles = []
    for i_ei in i_effective_images:
        i_in_path = path.index(i_ei)
        thimble_image_ei = np.vstack([thimble_paths[i_in_path-1], c2v(images[i_ei])[np.newaxis,:], thimble_paths[i_in_path]])
        image_thimbles.append({'image': images[i_ei], 'bcode': i_ei, 'i_in_path': i_in_path, 'thimble': thimble_image_ei, 'end_codes': [path[i_in_path-1], path[i_in_path+1]]})
    return image_thimbles

def integrate_image_thimble(image_thimble, func, nu, WITHOUTNORM=False):
    simplices = v2c(image_thimble['thimble'])
    E_image = integrate(simplices, nu, func, WITHOUTNORM)
    return E_image    

def integrate_image_thimble_all(image_thimbles, func, nu, WITHOUTNORM=False):
    Es = [integrate_image_thimble(image_thimble, func, nu, WITHOUTNORM) for image_thimble in image_thimbles]
    bcodes = [image_thimble['bcode'] for image_thimble in image_thimbles]
    return bcodes, Es

def contour2ic(paths_x, x, func, xs, ys, images, poles, r_crossing):
    """
    get the piece of contour connecting (x,0) to the nearest pole / boundary along the descending direction 
    #v = path_x.vertices
    """
    print('x=',x)
    dists = [dist(path_x.vertices, [x,0]).min() for path_x in paths_x]
    path_x = paths_x[np.array(dists).argmin()]
    v = path_x.vertices
    i_x = dist(v, [x,0]).argmin()
    info_crossings_pole_boundary = get_info_crossing_pole_boundary(v, xs, ys, images, poles, r_crossing)
    #h_x = amp_real(x, func) # 0 for x on the real axis
    #h_crossings = amp_real(v2c(v)[info_crossings_pole_boundary[0]], func)
    #info_crossings_pole_boundary_descend = info_crossings_pole_boundary[:, np.where(h_crossings < h_x)]
    if amp_real(v[i_x+1][0] + 1j * v[i_x+1][1], func) < amp_real(x, func): # descending direction is positive along contour index
        inds = np.where(info_crossings_pole_boundary[0] > i_x)[0]
        if len(inds) > 0: # pole/boundary exist along descending direction of contour
            iind = info_crossings_pole_boundary[0][inds].argmin() # take the first
            ind = inds[iind]
            i_pb = info_crossings_pole_boundary[0][ind]
            v_ic = v[i_x:i_pb]
        else:
            ind = info_crossings_pole_boundary[0].argmin() # the first after folding
            i_pb = info_crossings_pole_boundary[0][ind]
            v_ic = np.vstack([v[i_x:], v[:i_pb]])
    else:
        inds = np.where(info_crossings_pole_boundary[0] < i_x)[0]
        if len(inds) > 0: # pole/boundary exist along descending direction of contour
            iind = info_crossings_pole_boundary[0][inds].argmax() # take the first
            ind = inds[iind]
            i_pb = info_crossings_pole_boundary[0][ind]
            v_ic = v[i_pb:i_x][::-1,:]
        else:
            ind = info_crossings_pole_boundary[0].argmax() # the first after folding
            i_pb = info_crossings_pole_boundary[0][ind]
            v_ic = np.vstack([v[i_pb:], v[:i_x]])[::-1,:]
    if v_ic[0][1] != 0.:
        v_ic = np.vstack([np.array([x,0]), v_ic]) # add point (x, 0) 
    bcode = info_crossings_pole_boundary[1, ind]
    return {'vertices': v_ic, 'end_code': bcode}

def get_ic_xs(xs2int, xs, ys, hh, func, images, poles, r_crossing):
    """
    get relevant integration contours of hh corresponding to all x in xs2int  
    // hh created with: 
    xx, yy = np.meshgrid(xs, ys)
    hh = amp_imag(xx + 1j * yy, func)

    steps of getting relevant integration contours corresponding to x:
    1. get contour hh = h(x)  
    2. cut contour at infinity / pole / image
    3. keep ic_x: the descending contour connected to x (starting at x), record its endpoint (ic for integration_contour)
    
    note: ideally, xs2int should be a sub-set of xs 

    next: integration path from x1 to x2:
    ic_x1 + ic_x2[::-1]
    if ic_x1, ic_x2 do not have the same endpoint i.e. they cross an image contour, add the relevant image thimble (2 pieces linked to the image in the positive order in path) 
    """
    h_xs = amp_imag(xs2int, func)
    #h_xs_argsort = h_xs.argsort() 
    #xs2int_sorted = xs2int[h_xs.argsort()]
    #inv_argsort = xs2int_sorted.argsort()
    h_xs_set_sorted = list(set(h_xs)) # this can cope with xs with same h values
    h_xs_set_sorted.sort()
    idx = {x:i for i,x in enumerate(h_xs_set_sorted)} 
    inv_argsort = np.array([idx[x] for x in h_xs]) # if x in idx] 
 
    #cs = plt.contour(xs, ys, hh, h_xs[h_xs_argsort]) # get contour with h = h_i
    cs = plt.contour(xs, ys, hh, h_xs_set_sorted) # get contour with h = h_i
    ic_dicts = [contour2ic(cs.collections[inv_argsort[i]].get_paths(), xs2int[i], func, xs, ys, images, poles, r_crossing) for i in range(len(xs2int))]
    return ic_dicts 

def get_ic_x0(x0, xs, ys, hh, func, images, poles, r_crossing):
    """
    get relevant integration contours of hh corresponding x0
    """
    
    h_x0 = amp_imag(x0, func)

    cs = plt.contour(xs, ys, hh, [h_x0]) # get contour with h = h_i
    ic_dict = contour2ic(cs.collections[0].get_paths(), x0, func, xs, ys, images, poles, r_crossing) 
    return ic_dict

def get_xcross(images_effective, xs, ys, func, images, poles, r_crossing):
    """
    get xcross positions where the ascending contour of the imaginary images intersect with the real axis
    return xcross for all images 
    """
    xcross = images_effective.real.copy()
    imag_images_effective = images_effective[np.where(images_effective.imag != 0)[0]]
    for imag_image in imag_images_effective:
        i_imag_image = np.where(images==imag_image)[0]
        imag_i = amp_imag(imag_image, func)
        contours_h_i = get_contour_pieces(imag_i, xs, ys, func, images, poles, r_crossing)
        contours_imag_image = [contour_h_i for contour_h_i in contours_h_i if i_imag_image in contour_h_i['end_codes']] # (4) 2 descending + 2 ascending contour pieces lined to this image
        if_crossing_y0 = [if_cross_y0(contour_i['vertices']) for contour_i in contours_imag_image]
        try:
            contour_ascending_crossing = contours_imag_image[if_crossing_y0.index(1)]    
        except:
            print('error getting xcross for:')
            print(imag_image, if_crossing_y0)
        v = contour_ascending_crossing['vertices']
        if contour_ascending_crossing['end_codes'][0] != i_imag_image:
            v = v[::-1] # ensure that v starts at the image
        first_zero_crossing = np.where(np.diff(np.sign(v[:,1])))[0][0]
        sel_ = np.arange(first_zero_crossing-5, first_zero_crossing+5)
        vv = v[sel_]
        if vv[0,1] > vv[-1,1]: vv = vv[::-1,:]
        xcross_image = splev(0, splrep(vv[:,1], vv[:,0]))
        i_imag_image_effective = np.where(images_effective==imag_image)[0]
        print(i_imag_image_effective, xcross_image)
        xcross[i_imag_image_effective] = xcross_image
    return xcross

def get_E_int(xs2int, nu, func, ic_dicts, image_thimbles, xcrosses, WITHOUTNORM=False):
    """
    E_-infty^x
    ics linking x to -1, -2 or poles
    image thimbles linking boundary -1 to -2
    """
    E_ics = np.array([integrate(v2c(ic_dict['vertices']), nu, func, WITHOUTNORM=WITHOUTNORM) for ic_dict in ic_dicts])
    #ic_end_codes = np.array([ic_dict['end_code'] for ic_dict in ic_dicts])
    E_image = [integrate(v2c(t['thimble']), nu, func, WITHOUTNORM=WITHOUTNORM) for t in image_thimbles]
    #E_int = np.array([E_ics[i] if ic_end_codes[i]==-2 else -E_ics[i] for i in range(len(E_ics))])
    E_int = - E_ics
    def get_ixross(xcross):
        try:
            res = np.argwhere(np.array(xs2int)>xcross)[0][0]-1
        except:
            res = xs2int.size-1
        return res
    ixcrosses = [get_ixross(xcross) for xcross in xcrosses]
    for i,ixcross in enumerate(ixcrosses):
        E_int[ixcross+1:] = E_int[ixcross+1:] + E_image[i]
    return E_int 

def get_dIdx(nu, xs2int, xs, ys, HH, func, images, poles, r_crossing, image_thimbles=[], xcrosses=[], ic_dicts=[]):
    """
    given xs2int (points to sum for evaluating the integral), get the I contribution from each dx step. 
    Steps: 
    1. if not given, get image thimble for all effective images using 'get_image_thimbles'
    2. get xcross positions where the ascending contour of the imaginary images intersect with the real axis -> form a list of xcross for all images 
    3. get the thimble piece ic starting from each point x (not in xcross list) and ending at its the closest descending pole/boundary using 'get_ic_xs'
    4. for each dx = x2 - x1, integrate along positive direction for ic_x1 and negative direction for ic_x2; if there is an xcross between x1 and x2, then integrate also along the thimble of that image  
    Note: dE/dx is just exp(func) which is highly oscillatory and thus sensitively dependent on the binning in x -- not worth computing  
    Note: dI != |E_-infty^x2 - E_-infty^x1|^2, but dI = I_-infty^x2 - I_-infty^x1
    """ 
    if len(image_thimbles)==0: # get image thimbles
        H_images = list(set(amp_imag(images, func)))
        contours_H = [] # contours with H values = H(image)
        for i, imag_i in enumerate(H_images):
            contours_H_i = get_contour_pieces(imag_i, xs, ys, func, images, poles, r_crossing=dx*2) # 1.5 sometimes not enough
            contours_H = contours_H + contours_H_i
        # get relevant contours
        PL_thimble_candidates, images_effective = get_thimble_pieces(contours_H, images)
        # from graph to path
        path, G, thimble_end_codes = find_path(PL_thimble_candidates, bcode_real_image, RETURN_MORE=True)        
        # from path, link thimbles
        image_thimbles = get_image_thimbles(path, images, PL_thimble_candidates)

    images_effective = np.array([t['image'] for t in image_thimbles])
    if len(xcrosses)==0:
        xcrosses = get_xcross(images_effective, xs, ys, func, images, poles, r_crossing)
    if len(ic_dicts)==0:
        ic_dicts = get_ic_xs(xs2int, xs, ys, HH, func, images, poles, r_crossing)
    # integrate for dxs = np.diff(xs2int)
    E_int = get_E_int(xs2int, nu, func, ic_dicts, image_thimbles, xcrosses)
    I_int = abs(E_int)**2
    dI = np.diff(I_int)
    dx = np.diff(xs2int)
    return dI / dx

def get_smoothedB(width, nu, xs2int_regular, func, image_thimbles, xcrosses, ic_dicts, WITHOUTNORM=False):
    # here, to support convolution, xs2int must have regular spacing
    E_int = get_E_int(xs2int_regular, nu, func, ic_dicts, image_thimbles, xcrosses, WITHOUTNORM=WITHOUTNORM)
    if np.isnan(E_int).sum()==1:
        print('1 nan to correct')
        inan = np.where(np.isnan(E_int)==True)[0][0]
        E_int[inan] = (E_int[inan-1] + E_int[inan+1]) / 2.
    Ediff = np.diff(E_int)
    xs = (xs2int_regular[1:] + xs2int_regular[:-1]) / 2. 
    kernel = np.exp(-xs**2/2/width**2) / width / (2*np.pi)**0.5
    Ediff_smooth = np.convolve(Ediff, kernel, mode='same')
    B_smooth = abs(Ediff_smooth)**2 
    #dx = np.diff(xs2int_regular)
    return B_smooth #/ dx



