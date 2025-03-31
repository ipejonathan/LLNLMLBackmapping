import itertools
from collections import defaultdict
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# a DataUtils class with some key functionality
# ------------------------------------------------------------------------------
class DataUtils:

    # --------------------------------------------------------------------------
    # utils related to loading data (file loading and decomposing across ranks)
    # --------------------------------------------------------------------------
    @staticmethod
    def write_filelists(filename, files_by_ranks):
        LOGGER.debug(f'Writing file list to ({filename})')
        with open(filename, 'w') as fp:
            fp.write('rank, filename\n')
            for file_dict in files_by_ranks:
                rank, files = file_dict['rank'], file_dict['files']
                for f in files:
                    fp.write(f'{rank}, {f}\n')

    @staticmethod
    def fetch_from_npz(filename, keyname):
        LOGGER.debug(f'Loading ({keyname}) from ({filename})')
        with np.load(filename) as npz:
            data = npz.get(keyname, None)
        return data

    @staticmethod
    def write_to_npz(filename, data):
        LOGGER.debug(f'Writing ({list(data.keys())}) to ({filename})')
        np.savez_compressed(filename, **data)

    @staticmethod
    def domain_decomposition(nranks, nlist):

        assert isinstance(nranks, int) and isinstance(nlist, int)
        assert nranks > 0 and nlist > 0

        n = int(np.ceil(nlist / nranks))
        tasks = [[n * i, n * (i + 1)] for i in range(nranks)]
        tasks[-1][1] = nlist

        return tasks

    @staticmethod
    def reduce_files(files, nfiles, randomly=True):

        if nfiles == 0 or nfiles >= len(files):
            return files

        LOGGER.info(f'Reducing {len(files)} files to {nfiles} (randomly={randomly})')
        if randomly:
            return np.random.choice(files, nfiles, replace=False).tolist()
        else:
            return files[-1 * nfiles:]

    # --------------------------------------------------------------------------
    # utils related to rescaling data and quick analysis
    # --------------------------------------------------------------------------
    @staticmethod
    def compute_normalization(data, label=''):
        LOGGER.info(f'Computing min/max for {data.shape} {label}')
        axs = tuple(range(data.ndim - 1))
        dmin = data.min(axis=axs)
        dmax = data.max(axis=axs)
        return dmin, dmax

    @staticmethod
    def compute_standardization(data, label=''):
        LOGGER.info(f'Computing mean/std for {data.shape} {label}')
        n = data.shape[-1]
        dmean = np.zeros(n, dtype=data.dtype)
        dstd = np.zeros(n, dtype=data.dtype)
        for i in range(n):
            dmean[i] = data[..., i].mean(dtype=np.float)
            dstd[i] = data[..., i].std(dtype=np.float)
        return dmean, dstd

    @staticmethod
    def normalize(data, label, dmin=None, dmax=None):

        if dmin is None or dmax is None:
            dmin, dmax = DataUtils.compute_normalization(data)

        LOGGER.info(f'Normalizing {data.shape} {label}')
        data -= dmin
        data /= (dmax - dmin)
        return data, dmin, dmax

    @staticmethod
    def denormalize(data, label, dmin=None, dmax=None):

        # LOGGER.info(f'De-normalizing {data.shape} {label}')
        data *= (dmax - dmin)
        data += dmin
        return data

    @staticmethod
    def standardize(data, label, dmean=None, dstd=None):

        if dmean is None or dstd is None:
            dmean, dstd = DataUtils.compute_standardization(data, label)

        # LOGGER.info(f'Standardizing {data.shape} {label}')
        data -= dmean
        data /= dstd
        return data, dmean, dstd

    @staticmethod
    def standardize_zero_std(data, label, ind, dmean=None, dstd=None):

        if dmean is None or dstd is None:
            dmean, dstd = DataUtils.compute_standardization(data, label)

        LOGGER.info(f'Standardizing {data.shape} {label}')
        data -= dmean

        # for i in range(data.shape[0]):
        #     for j in range(data.shape[1]):
        #         if j == ind:
        #             data[i,j,0:2] = data[i,j,0:2] / dstd[j,0:2]
        #         else:
        #             data[i,j,:] = data[i,j,:] / dstd[j,:]
        for j in range(data.shape[1]):
            if j == ind:
                data[:,j,0:2] = data[:,j,0:2] / dstd[j,0:2]
            else:
                data[:,j,:] = data[:,j,:] / dstd[j,:]
        return data, dmean, dstd

    @staticmethod
    def summary(data, filename=''):
        assert len(data.shape) == 4

        LOGGER.info(f'Summarizing the data {data.shape}')
        dmin, dmax = DataUtils.compute_normalization(data)
        dmean, dstd = DataUtils.compute_standardization(data)

        for i in range(data.shape[-1]):
            LOGGER.info(f'channel {i}: min = {dmin[i]}; max = {dmax[i]}; '
                        f'mean = {dmean[i]}; std = {dstd[i]}')

        if len(filename) > 0:
            np.savez_compressed(filename,
                                min = np.array(dmin), max = np.array(dmax),
                                mean = np.array(dmean), std = np.array(dstd),
                                data_shp = data.shape)

    # --------------------------------------------------------------------------
    @staticmethod
    def compute_histograms(data, nbins=80):
        assert isinstance(data, np.ndarray)
        assert len(data.shape) == 4
        assert data.shape[-1] == 14

        histograms = [np.histogram(data[:, :, :, c], bins=nbins)
                      for c in range(data.shape[-1])]
        hists = np.array([h[0] for h in histograms])
        bins = np.array([h[1] for h in histograms])
        return hists, bins

    # --------------------------------------------------------------------------
    # utils related to shuffling and splitting data
    # --------------------------------------------------------------------------
    @staticmethod
    def shuffle(data):
        np.random.shuffle(data)
        return data

    @staticmethod
    def split(data, factor):

        assert isinstance(factor, float)
        assert 0 <= factor <= 1
        tsplit = int(data.shape[0] * factor)
        return data[:tsplit], data[tsplit:]

    @staticmethod
    def split_all(train_ratio, data, *args):

        assert 0.5 < train_ratio < 1.0

        ndata = data.shape[0]
        ntrain = int(ndata * train_ratio)

        # first, shuffle the data
        np.random.seed(1223)
        shuffled_idxs = np.arange(ndata)
        np.random.shuffle(shuffled_idxs)
        _tidx = shuffled_idxs[:ntrain]
        _vidx = shuffled_idxs[ntrain:]

        LOGGER.info(f'Split data: {ndata} into '
                    f'training = {_tidx.shape[0]}, validation = {_vidx.shape[0]}')

        data = [(data[_tidx], data[_vidx])]
        for d in args:
            data.append((d[_tidx], d[_vidx]))

        return data

    # --------------------------------------------------------------------------
    # utils related to graph operations
    # --------------------------------------------------------------------------
    @staticmethod
    def edgelist_to_outgoing_and_angles(edges):

        # create a set of outgoing edges from each node
        outgoing = defaultdict(list)
        for (a, b) in edges:
            outgoing[a].append(b)
            outgoing[b].append(a)

        # create a list of all angles along the edges in the graph
        angle_ids = []
        for n, nbrs in outgoing.items():
            _ = [[n, a, b] for a, b in itertools.product(nbrs, nbrs) if a < b]
            angle_ids.extend(_)

        return dict(outgoing), np.array(angle_ids)

    @staticmethod
    def find_connected_components(nnodes, edges):

        # simplified union-find
        LOGGER.debug (f'Finding connected components (nnodes = {nnodes}, nedges = {edges.shape})')

        # each node is its own component
        components = np.arange(nnodes, dtype=int)
        for [a,b] in edges:
            ca, cb = components[a], components[b]
            components[a] = components[b] = min(ca,cb)

        components, sizes = np.unique(components, return_counts=True)
        LOGGER.debug(f'Found {components.shape[0]} components: {components} = {sizes}')
        return components, sizes

    # --------------------------------------------------------------------------
    # fix the periodic boundary in z (z ONLY)
    # --------------------------------------------------------------------------
    @staticmethod
    def fix_periodic_z_bfs(seed_idxs, outgoing, bbox_z, node_pos):

        #print (f'Fixing periodic-z artifacts using BFS (seed = {seed_idxs}, bbox_z = {bbox_z:.3f})')

        # perform a bfs on the graph!
        visited = [s for s in seed_idxs]
        queue = [s for s in seed_idxs]

        while queue:
            s = queue.pop(0)
            for nbr in outgoing[s]:
                # check and fix periodic boundary artifact
                d = node_pos[nbr] - node_pos[s]
                if d[2] > 0.5*bbox_z:
                    node_pos[nbr, 2] -= bbox_z
                elif d[2] < -0.5*bbox_z:
                    node_pos[nbr, 2] += bbox_z

                if nbr not in visited:
                    visited.append(nbr)
                    queue.append(nbr)

        return visited

    # --------------------------------------------------------------------------
    # utils related to computing key metrics in data and graphs
    # --------------------------------------------------------------------------
    @staticmethod
    def rmsd(x, y):
        assert x.shape == y.shape
        assert x.ndim == 3, 'Expect (n, m, d) arrays: n = num_data, m = num_points, d = dim_points'
        d = np.square(x - y).sum(axis=-1)   # compute the squared distance
        return np.sqrt(d.mean(axis=-1))     # compute the mean over all points

    @staticmethod
    def rmsd_dist(x, y):
        assert x.shape == y.shape
        assert x.ndim == 2, 'Expect (n, m) arrays: n = num_data, m = num_points, d = dim_points'
        d = np.square(x - y).sum(axis=-1)   # compute the squared distance
        return np.sqrt(d.mean(axis=-1))     # compute the mean over all points

    @staticmethod
    def distance(a,b):
        return np.linalg.norm(a-b, axis=-1)

    @staticmethod
    def angle(o,a,b):
        va, vb = a-o, b-o
        va_dot_vb = (va*vb).sum(axis=-1)
        va_cross_vb = np.cross(va, vb, axis=-1)
        va_cross_vb = np.linalg.norm(va_cross_vb, axis=-1)
        return np.arctan2(va_cross_vb, va_dot_vb)

    @staticmethod
    def compute_distances_and_angles(node_pos, edges, angle_ids):

        assert edges.ndim == 2 and edges.shape[1] == 2
        assert angle_ids.ndim == 2 and angle_ids.shape[1] == 3
        assert node_pos.ndim in [2,3] and node_pos.shape[-1] == 3

        print(f'Computing edge distances: pos = {node_pos.shape}, edges = {edges.shape}')
        a,b = edges[:,0], edges[:,1]
        dists = DataUtils.distance(node_pos[:,a], node_pos[:,b])
        print(f'    distances = {dists.shape}: [{dists.min():.3f}, {dists.max():.3f}]')

        print(f'Computing node angles: pos = {node_pos.shape}, angle_ids = {angle_ids.shape}')
        o,a,b = angle_ids[:,0], angle_ids[:,1], angle_ids[:,2]
        angles = DataUtils.angle(node_pos[:,o], node_pos[:,a], node_pos[:,b])
        print(f'    angles = {angles.shape}: [{angles.min():.3f}, {angles.max():.3f}]')

        return dists, angles

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# handle complexes and reorientaton
# ------------------------------------------------------------------------------
def complex_idxs(protein_ids, complex_ids):

    if complex_ids is None:
        return None

    _, pid2pidx = np.unique(protein_ids, return_inverse=True)
    assert protein_ids.shape == _.shape
    assert 1 == _.min() and protein_ids.shape[0] == _.max()
    return pid2pidx[complex_ids-1]


def complex_midpoints(protein_positions, concentrations,
                      protein_states, complex_ids, protein_ids,
                      force_center_to_first_ras=False):

    # no protein
    if protein_positions is None:
        assert protein_states is None
        assert complex_ids is None
        return None, None

    nprots = protein_positions.shape[0]
    assert nprots == protein_states.shape[0]

    # no protein
    if 0 == nprots:
        assert complex_ids is None
        return None, None

    # first protein is already at the center
    dx = 30./concentrations.shape[0]
    d = protein_positions[0, :2]
    # if it is not already centered
    if d[0] > 0.5*dx and d[1] > 0.5*dx:
        d = np.abs(protein_positions[0, :2] - np.array([15., 15.]))
        assert d[0] < 0.5*dx and d[1] < 0.5*dx, 'Should be centered around first protein'

    # ----------------------------------------------------------------------
    def mid_point(idxa, idxb):
        if force_center_to_first_ras and idxa == 0:
            return protein_positions[idxa]
        elif force_center_to_first_ras and idxb == 0:
            return protein_positions[idxb]
        else:
            # handle periodic
            ab_vec = protein_positions[idxb] - protein_positions[idxa]
            for i in range(2):
                if ab_vec[i] > 15.:
                    ab_vec[i] -= 30.
                elif ab_vec[i] < -15.:
                    ab_vec[i] += 30.
            return protein_positions[idxa] + 0.5*ab_vec

    # ----------------------------------------------------------------------
    # 1 protein
    if 1 == nprots:
        assert complex_ids is None or complex_ids.shape[0]==0
        return protein_positions[0], None

    # 2 proteins that are both RAS
    if 2 == nprots and complex_ids is None:
        return protein_positions[0], protein_positions[1]

    # 2 proteins that are a complex
    if 2 == nprots and complex_ids is not None:
        assert (1,2) == complex_ids.shape
        return mid_point(0, 1), None

    # 3 proteins (1 RAS and 1 RAS-RAF complex)
    if 3 == nprots:
        assert (1,2) == complex_ids.shape

        cidxs = complex_idxs(protein_ids, complex_ids)
        pcid = cidxs[0]
        ncid = np.setdiff1d(np.array([0, 1, 2]), pcid)
        assert ncid.shape[0] == 1

        posnc = protein_positions[ncid[0]]
        posc = mid_point(pcid[0], pcid[1])
        return (posnc, posc) if ncid[0] == 0 else (posc, posnc)

    # 4 proteins (2 RAS-RAF complex)
    if 4 == nprots:
        assert (2,2) == complex_ids.shape

        cidxs = complex_idxs(protein_ids, complex_ids)
        pa,pb = cidxs[0], cidxs[1]
        posa = mid_point(pa[0], pa[1])
        posb = mid_point(pb[0], pb[1])
        return (posa, posb) if 0 in pa else (posb, posa)

    assert False, f'Unknown configuration: {protein_states} {complex_ids}'
    return None, None


# ------------------------------------------------------------------------------
# reorientation and alignment functionality
# ------------------------------------------------------------------------------
def reorient_patch(protein_positions, concentrations,
                   protein_states, complex_ids, protein_ids, data_label):
    """This function reorients a patch with respect to protein positions.
    -  0 proteins: do nothing
    - >0 proteins: center around the first RAS
    - >1 proteins: center around the first RAS and
                    put RAS-RAS vector on [0,45] degrees
    """
    # --------------------------------------------------------------------------
    # no proteins. nothing to do
    if protein_positions is None:
        return concentrations, protein_positions

    if len(protein_positions.shape) == 0:
        return concentrations, protein_positions

    nprots = protein_positions.shape[0]
    ndims = protein_positions.shape[1]
    if nprots == 0:
        return concentrations, protein_positions

    # --------------------------------------------------------------------------
    # start working
    psize, gsize = 30., concentrations.shape[0]
    p2g = psize / gsize
    g2p = gsize / psize

    # lets check the frame of reference of the patch
    cx = np.abs(protein_positions[0])
    if cx[0] <= 0.5*p2g and cx[1] <= 0.5*p2g:
        pcent = np.array([0., 0., 0.])
        pext = np.array([-0.5 * psize, 0.5 * psize, 0.])
    else:
        pcent = np.array([0.5 * psize, 0.5 * psize, 0.])
        pext = np.array([0., psize, 0.])

    if ndims == 2:
        pcent = pcent[:2]
        pext = pext[:2]

    # --------------------------------------------------------------------------
    def _center_at(_img, _pos, _p):
        disg = np.round(g2p * (pcent - _p)).astype(np.int)
        if disg[0] == 0 and disg[1] == 0:
            return _img, _pos
        _pos += p2g * disg
        for i in range(2):
            _pos[_pos[:, i] < pext[0], i] += psize
            _pos[_pos[:, i] > pext[1], i] -= psize
        _img = np.roll(_img, shift=(disg[1], disg[0]), axis=(0, 1))
        return _img, _pos

    def _flip(_img, _pos, _dir):
        if _dir == 'x':
            _pos[:, 0] = psize - _pos[:, 0]
            _img = np.flip(_img, axis=1)
        elif _dir == 'y':
            _pos[:, 1] = psize - _pos[:, 1]
            _img = np.flip(_img, axis=0)
        else:
            assert 0
        return _img, _pos

    def _rot(_img, _pos, _dir):
        _pos[:, [1, 0]] = _pos[:, [0, 1]] - pcent[:2]
        if _dir == 'ccw':
            _pos[:, 0] *= -1
            _img = np.rot90(_img, axes=(1, 0))
        elif _dir == 'cw':
            _pos[:, 1] *= -1
            _img = np.rot90(_img, axes=(0, 1))
        else:
            assert 0
        _pos += pcent
        return _img, _pos

    # --------------------------------------------------------------------------
    # create a copy
    img = np.copy(concentrations)
    ppositions = np.copy(protein_positions)

    # consider only 1 protein or 1 complex
    # center with respect to the two proteins
    if True:
        assert nprots in [0,1,2], f'Assuming there are only two proteins, but found {nprots}'
        pa = protein_positions[0]
        pb = protein_positions[1] if nprots == 2 else None

    # consider multiprotein and multicomplex patches
    # here, we want to center with respect to the midpoints of complex
    else:
        pa, pb = complex_midpoints(protein_positions, concentrations,
                                   protein_states, complex_ids, protein_ids,
                                   force_center_to_first_ras=True)

    # --------------------------------------------------------------------------
    # 1 protein (center around first RAS)
    if pb is None:
        img, ppositions = _center_at(img, ppositions, pa)
        return img, ppositions

    # --------------------------------------------------------------------------
    # 2 proteins

    # center the first
    img, ppositions = _center_at(img, ppositions, pa)

    # compute where the vector falls
    disp = pb - pa
    ang = np.rad2deg(np.arctan2(disp[1], disp[0]))
    if ang < 0:
        ang += 360
    q = int(np.floor(ang / 45))
    assert 0 <= q < 8

    # based on the half-quadrant, different transformations are needed
    if q == 1:
        img, ppositions = _flip(img, ppositions, 'y')
        img, ppositions = _rot(img, ppositions, 'ccw')

    elif q == 2:
        img, ppositions = _rot(img, ppositions, 'cw')

    elif q == 3:
        img, ppositions = _flip(img, ppositions, 'x')

    elif q == 4:
        img, ppositions = _flip(img, ppositions, 'x')
        img, ppositions = _flip(img, ppositions, 'y')

    elif q == 5:
        img, ppositions = _flip(img, ppositions, 'y')
        img, ppositions = _rot(img, ppositions, 'cw')

    elif q == 6:
        img, ppositions = _rot(img, ppositions, 'ccw')

    elif q == 7:
        img, ppositions = _flip(img, ppositions, 'y')

    # --------------------------------------------------------------------------
    return img, ppositions


# ------------------------------------------------------------------------------
def align_protein(x, anchor0, anchor1, anchor2):
    from scipy.spatial.transform import Rotation as R

    '''
    # define anchors generically. we want
        # anchor0 to be at the origin: (0,0,0)
        # anchor1 to be on +x axis:    (+x, 0, 0)
        # anchor2 to be on +y axis:    (x, +y, 0)
    anchor0 = np.arange(183, 184, dtype=int)        # farnesyl!
    anchor1 = np.arange(0, 165, dtype=int)          # gdomain!
    anchor2 = np.arange(0, 1, dtype=int)            # first bead!
    '''
    xaxis = np.array([1,0,0])

    def get_anchors(x):
        return np.mean(x[anchor0], axis=0), \
               np.mean(x[anchor1], axis=0), \
               np.mean(x[anchor2], axis=0)

    do_norm = lambda x: x / np.linalg.norm(x)

    # --------------------------------------------------------------------------
    # step 1: move anchor 0 to the center
    x0, x1, x2 = get_anchors(x)
    x -= x0

    # --------------------------------------------------------------------------
    # step2: rotate at x+ (1,0,0)
    x0, x1, x2 = get_anchors(x)

    # axis of rotation
        # a x b = |a| |b| sin (theta) n
        # if we cross normalize a and b, we can get theta out
    norm = np.cross(do_norm(x1), do_norm(xaxis))
    if norm.all() == 0:
        ang = 0
    else:
        ang = np.arcsin(np.linalg.norm(norm))

    r = R.from_rotvec(ang * do_norm(norm))
    #r = R.from_rotvec(ang * do_norm(norm), degrees=False)
    x = r.apply(x)

    # --------------------------------------------------------------------------
    # step2: rotate x2 to be (1) along +y axis and (2) in yz plane
        # i.e., z component should be zero
    x0, x1, x2 = get_anchors(x)

    # need to compute the angle of rotation in yz plane
    x2_2d = x2 - np.dot(x2, xaxis) * xaxis
    x2_2d = x2_2d[1:]       # (y,z)
    ax_2d = xaxis[:2]       # (y,z)

    ang = np.arccos(np.dot(do_norm(x2_2d), ax_2d))

    r = R.from_rotvec(-1 * ang * xaxis)
    #r = R.from_rotvec(-1 * ang * do_norm(xaxis), degrees=False)
    x = r.apply(x)

    return x

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
