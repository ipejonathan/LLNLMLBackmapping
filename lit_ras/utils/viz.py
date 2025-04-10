import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation
import matplotlib.ticker

import logging
LOGGER = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# from .lipids import LIPID_NAMES

# ------------------------------------------------------------------------------
def replace_nans(data):
    for d in range(data.shape[1]):
        k = np.isnan(data[:,d])
        if k.sum() > 0:
            LOGGER.info(' -- dimension {}: replacing {} nans with zeros'.format(d,k.sum()))
            data[np.isnan(data[:,d]),d] = 0
    return data


# ------------------------------------------------------------------------------
class MidpointNormalize(matplotlib.colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        super(MidpointNormalize, self).__init__(vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


# ------------------------------------------------------------------------------
def add_colorbar_to_empty_axis(ax, img, rng, fontsize=12):

    ax.axis('off')
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="30%", pad=0, pack_start=True)
    ax.get_figure().add_axes(ax_cb)

    ticks = [rng[0], 0.5*(rng[0]+rng[1]), rng[1]]
    ticklabels = ['{:.3f}'.format(x) if not np.isclose(x, 0) else '0' for x in ticks]

    cbar = plt.colorbar(img, cax=ax_cb, orientation='vertical', ticks=ticks)
    cbar.ax.set_yticklabels(ticklabels, fontsize=fontsize)


# ------------------------------------------------------------------------------
# def plot_images(imgs, labels, color_scale, filename, show):

#     isz = 2
#     show_cbar = True
#     show_lipids_name = True

#     # --------------------------------------------------------------------------
#     assert isinstance(imgs, np.ndarray)
#     assert isinstance(color_scale, str)
#     assert isinstance(filename, str)
#     assert isinstance(show, bool)
#     assert color_scale in ['diverg', 'quant']

#     nimages, nchannels = imgs.shape[0], imgs.shape[-1]
#     if labels is not None:
#         assert isinstance(labels, np.ndarray)
#         assert labels.shape == (nimages, )

#     # --------------------------------------------------------------------------
#     # setup for diverging colormaps
#     if color_scale == 'diverg':
#         cmax = np.abs(imgs).max(axis=(0, 1, 2))
#         cmin = -cmax
#         cnorms = [MidpointNormalize(midpoint=0, vmin=cmin[c], vmax=cmax[c]) for c in range(nchannels)]
#         cmap = matplotlib.cm.RdBu_r

#     else:
#         cmax = imgs.max(axis=(0, 1, 2))
#         cmin = imgs.min(axis=(0, 1, 2))
#         cnorms = [None for c in range(nchannels)]
#         cmap = 'viridis'

#     # --------------------------------------------------------------------------
#     fig, axs = plt.subplots(nrows=nimages+int(show_cbar), ncols=nchannels,
#                             figsize=(isz * nchannels, isz * nimages))

#     for c in range(nchannels):
#         for r in range(nimages):
#             if nchannels == 1:
#                 ax = axs[r]
#             else:
#                 ax = axs[r, c]
#             ax.axis('off')
#             pim = ax.imshow(imgs[r, :, :, c], vmin=cmin[c], vmax=cmax[c],
#                             norm=cnorms[c], cmap=cmap)

#             if c == 0 and labels is not None:
#                 ax.text(-6,19, '{}'.format(labels[r]))
#             #if r == 0 and len(labels) > 0:
#             #    ax.set_title(labels[c])
#             if r == 0 and show_lipids_name == True:
#                 ax.text(0,-6, '{}'.format(LIPID_NAMES[c]))

#         if show_cbar:
#             if nchannels == 1:
#                 ax = axs[-1]
#             else:
#                 ax = axs[-1, c]
#             add_colorbar_to_empty_axis(ax, pim, (cmin[c], cmax[c]))

#     # --------------------------------------------------------------------------
#     plt.savefig(filename, bbox_inches='tight')
#     if show:
#         plt.show()
#     plt.close()


# ------------------------------------------------------------------------------
def plot_distribution(ax, hist, bins, xtype, label='',
                      fcolor='#a6cee3', ecolor='#1f78b4', alpha=1):

    assert isinstance(hist, np.ndarray) and isinstance(bins, np.ndarray)
    assert len(hist.shape) == len(bins.shape)
    assert hist.shape[0] == bins.shape[0]-1
    assert xtype in ['symmetric', 'positive', 'default']

    h = hist / hist.sum()
    b = 0.5 * (bins[1:] + bins[:-1])

    ax.fill_between(b, h, 0, label=label, step='mid',
                             facecolor=fcolor, edgecolor=ecolor, alpha=alpha)
    ax.set_ylim(bottom=0)

    if xtype == 'positive':
        ax.set_xlim(left=0)

    elif xtype == 'symmetric':
        x = max(abs(b[0]), abs(b[-1]))
        ax.set_xlim(left=-x, right=x)

    return h.max()


def plot_labeled_scatters(pos, labels, colors, ax=None):

    if np.unique(labels).shape[0] > len(colors):
        LOGGER.info('please add more colors')

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    for _l, _c in zip(np.unique(labels), colors):
        _ids = labels == _l
        _lcnt = np.count_nonzero(_ids)
        # ax.scatter(pos[_ids, 0], pos[_ids, 1], c=_c, alpha=0.5, label='{:02d}={}'.format(_l, _lcnt))
        classes = ['A', 'A\'', 'A\'\'','new','A\'\'\'']
        ax.scatter(pos[_ids, 0], pos[_ids, 1], c=_c, alpha=0.5, label='{}'.format(classes[_l], _lcnt))
    # ax.legend()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          ncol=3, markerscale=3., fontsize="30")
    return ax

def plot_labeled_scatters3(pos, labels, colors, ax=None):

    if np.unique(labels).shape[0] > len(colors):
        LOGGER.info('please add more colors')

    if ax is None:
        fig= plt.subplots(1, 1, figsize=(12, 12))
        ax = plt.axes(projection ="3d")

    for _l, _c in zip(np.unique(labels), colors):
        _ids = labels == _l
        _lcnt = np.count_nonzero(_ids)
        classes = ['A', 'A\'', 'A\'\'','new', 'A\'\'\'' ]
        # ax.scatter3D(pos[_ids, 0], pos[_ids, 1], pos[_ids, 2], c=_c, alpha=0.5, label='{:02d}={}'.format(_l, _lcnt))
        ax.scatter3D(pos[_ids, 0], pos[_ids, 1], pos[_ids, 2], c=_c, alpha=0.5, label='{}={}'.format(classes[_l],_lcnt))
    ax.legend()
    return ax

def plot_histo2d_hexbin(ax, x, y, xlabel, ylabel, scale='linear'):
    assert scale in ['log', 'linear']

    if scale == 'log':
        hb = ax.hexbin(x, y, gridsize=50, mincnt=1, bins='log', cmap='inferno_r')
    else:
        hb = ax.hexbin(x, y, gridsize=50, mincnt=1, cmap='inferno_r')

    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return hb


# ------------------------------------------------------------------------------
# PLOT CONVOLUTION
# ------------------------------------------------------------------------------
def plot_feature_maps(data, title, filename, show):

    show_cbar = True
    isz = 2
    assert isinstance(data, np.ndarray)
    assert isinstance(title, str) and isinstance(filename, str)
    assert len(data.shape) == 4

    dmin, dmax = data.min(axis=(0,1,2)), data.max(axis=(0,1,2))

    # get the shape of the data
    ndata, x, y, nfilters = data.shape

    fig, axs = plt.subplots(nrows=ndata+int(show_cbar), ncols=nfilters,
                            figsize=(isz * (nfilters + 2), isz * (ndata + 2)))
    if nfilters == 1:
        axs = axs[:, np.newaxis]
    for f in range(nfilters):
        for d in range(ndata):
            ax = axs[d][f]
            pim = ax.imshow(data[d, :, :, f], vmin=dmin[f], vmax=dmax[f])
            ax.axis('off')

        if show_cbar:
            add_colorbar_to_empty_axis(axs[-1,f], pim, (dmin[f], dmax[f]))

    fig.suptitle(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_conv_layer(layer, title, filename, show):

    # get the filters and biases
    filters, biases = layer.get_weights()
    assert filters.shape[-1] == biases.shape[0]
    nfilters = filters.shape[-1]
    fdepth = filters.shape[-2]

    # normalize filter values to 0-1 so we can visualize them
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    f_min, f_max = filters.min(), filters.max()

    LOGGER.info('filters =', filters.shape, biases.shape)
    LOGGER.info(f_min, f_max)

    isz = 2
    fig, axs = plt.subplots(nrows=fdepth, ncols=nfilters,
                            figsize=(isz * (nfilters + 2), isz * (fdepth + 2)))

    for f in range(nfilters):
        for d in range(fdepth):
            ax = axs[d][f]
            ax.imshow(filters[:, :, d, f], vmin=f_min, vmax=f_max, cmap='gray')
            ax.axis('off')

            if d == 0:
                ax.set_title('{}'.format(f), fontsize=8)
            if f == 0:
                ax.axis('on')
                ax.set_ylabel('c{}'.format(d), fontsize=8)
                ax.set_yticks([])
                ax.set_xticks([])


    fig.suptitle(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_activations(data, title, filename, show):

    assert isinstance(data, np.ndarray)

    h, b = np.histogram(data, bins=80)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    plot_distribution(axs, h, b, 'default')
    axs.set_ylabel('Distribution')
    axs.set_xlabel('Activations ({})'.format(data.shape[1]))
    axs.set_title(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# ------------------------------------------------------------------------------
# PLOT ERROR CURVES
# ------------------------------------------------------------------------------
colors = {'total': 'k',
          'symmetry': '#1b9e77', 'category': '#d95f02',
          'rdf': '#7570b3', 'psd': '#e7298a', 'adf': '#a6d854'}

ccolors = {'loss': 'r','class0': '#a6cee3',
          'class1': '#1b9e77', 'class2': '#d95f02',
          'class3': '#7570b3', 'class4': '#e7298a', 'class5': '#a6d854',
          'class6': '#fdbf6f', 'class7': '#ff7f00', 'class8': '#bebada',
          'class9': '#80b1d3', 'class10': '#ccebc5', 'class11': '#ffed6f',
          'loss-total': 'r', 'loss-pos' : '#a6cee3', 'loss-loops': '#1b9e77',
          'loss-cons': '#d95f02', 'kl_loss': '#7570b3',
          'loss-potential': '#e7298a', 'loss-cons2': '#a6d854',}

def plot_error_curves(history, filename, show):

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))

    for k,v in history.items():
        v = np.array(v)
        v = v.flatten()
        if v.ndim > 1:
            continue

        c = ccolors.get(k.split('_')[-1], 'k')
        t = '--' if k.split('_')[0] == 'val' else '-'

        h = history[k]
        x = np.arange(len(h))+1
        axs.plot(x, h, label=k, color=c, linestyle=t)

    axs.set_xlim(left=0)
    axs.set_ylim(bottom=0)
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Metric Value')
    axs.tick_params(labelright=True)

    axs.grid(which='major', color='#dddddd', linewidth=0.8)
    axs.grid(which='minor', color='#eeeeee', linestyle=':', linewidth=0.5)
    axs.minorticks_on()

    plt.legend(ncol=2, loc="upper left")
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_training_error_curves(history, title, units, filename, show):

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))

    metrics= {}
    for k in history.keys():
        # if k == "val_loss_loops":
        #     k = "val_loss-loops"
        # if k == "loss_loops":
        #     k = "loss-loops"
        c = ccolors.get(k.split('_')[-1], 'k')
        # if k == "val_loss-loops":
        #     k = "val_loss_loops"
        # if k == "loss-loops":
        #     k = "loss_loops"
        t = '--' if k.split('_')[0] == 'val' else '-'
        if 'loss' in k:
            mn = np.nanmean(history[k], axis=0)
            x = np.arange(mn.shape[0])+1
            axs.plot(x, mn, label=k, color=c, linestyle=t)
            metrics[k] = mn

    axs.set_xlim(left=0)
    axs.set_ylim(bottom=0)
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Metric Value (' + units +')')
    axs.tick_params(labelright=True)

    axs.grid(which='major', color='#dddddd', linewidth=0.8)
    axs.grid(which='minor', color='#eeeeee', linestyle=':', linewidth=0.5)
    axs.minorticks_on()
    # plt.ylim(0, 10)
    plt.ylim(0, 4)
    plt.legend(ncol=2, loc="upper left")
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_residual_plot(original, reconstructed, filename, title, type, show):

    fig,axs = plt.subplots(1, 1)
    s = 100
    # plt.scatter(original, original - reconstructed, color="navy", s=s, lw=0)
    if type == "distances (Å)":
        plt.scatter(original[0:164,0:164], (original - reconstructed)[0:164,0:164], color="navy", s=s, lw=0, label='G-domain')
        plt.scatter(original[164:,164:], (original - reconstructed)[164:,164:], color="red", s=s, lw=0, label='No G-domain')
        axs.set_xlabel('Actual distances (Å)')
    else:
        plt.scatter(original[:,0], (original - reconstructed)[:,0], color='navy', s=s, lw=0, label='x')
        plt.scatter(original[:,1], (original - reconstructed)[:,1], color='green', s=s, lw=0, label='y')
        plt.scatter(original[:,2], (original - reconstructed)[:,2], color='red', s=s, lw=0, label='z')
        axs.set_xlabel('Actual positions')
    axs.set_ylabel('Residual')
    axs.set_title(title)
    plt.axhline(y=0, ls='--')
    plt.legend(loc='upper left')
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_histogram_positions(positions, positions2, filename, title, nbins=30,show=True):

    fig,axs = plt.subplots(1, 1)
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    arr = plt.hist(positions.flatten(), alpha=0.5, density=False, bins=nbins, label="RMSD < 3.5")
    arr2 = plt.hist(positions2.flatten(), alpha=0.5, density=False, bins=nbins, label="RMSD > 8")
    # for i in range(nbins):
    #     plt.text(arr[1][i],arr[0][i],"%.f" % (arr[0][i]), fontsize=8)
    #     plt.text(arr2[1][i],arr2[0][i],"%.f" % (arr2[0][i]), fontsize=8)

    axs.set_ylabel('Count')
    axs.set_xlabel('Positions ({})'.format((positions.shape[0]+positions2.shape[0])*751*3))
    axs.set_title(title)
    plt.yscale("log")
    plt.axhline(y=0, ls='--')
    plt.legend(loc='upper left')
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_positions(positions, filename, title, show):

    fig = plt.figure()
    ax = plt.axes(projection ="3d")
    l = 100
    ax.scatter3D(positions[:, 0], positions[:, 1], positions[:, 2], color="navy")
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def minmax_dists(x):
    import scipy.spatial
    d = scipy.spatial.distance.pdist(x)
    return d.min(), d.max()

def plot_positions_single_gro(positions,  boxsize, parts, anchors, filename, title, inpath, show=False):

    # n,3
    assert len(positions.shape) == 2

    fig = plt.figure()
    # fig = plt.figure(figsize=(12,5))
    ax = plt.axes(projection ="3d")
    plot_positions_gro_3d(ax, positions,  boxsize, parts, anchors, title, inpath)

    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_positions_gro_3d(ax, positions, boxsize, parts, anchors, title, inpath, ids2colors={}):
    '''Input needed: path to file, box size, and a list of protein parts.
    Names to include in the protein parts list:
    ALL = display all of the protein
    G = RAS G-domain and HVR
    CYF = RAS farnesyl
    RBD = RAF RBD
    CRD = RAF RBD-CRD linker, CRD, and CRD-cys-loop
    LINK1 = RAF Linker1
    PS365 = RAF PS365 region
    LINK2 = RAF Linker2
    PS729 = RAF PS729
    KD = RAF Kinase domain
    1433a = First 14-3-3
    1433b = Second 14-3-3
    '''
    import MDAnalysis as mda

    d = minmax_dists(positions)

    # inpath="/p/vast1/konsgeor/helgi_files/new_structure"
    gromacs_ref_file = os.path.join(inpath, 'ras-raf-14-3-3-ref-CG.gro')
    # file_traj = os.path.join(inpath, 'traj_comp.xtc')
    # file_topo = os.path.join(inpath, 'topol.tpr')
    u_all = mda.Universe(gromacs_ref_file)
    # u_in = u_all.select_atoms('not (resname W NA CL)')
    u_in = u_all.atoms
    u = u_in[:2897]
    u.atoms.positions = positions


    BB = u.select_atoms('name BB')
    CYF = u.select_atoms('resname CYF and not name SC*')
    GX = BB.atoms.positions[0:184,0]
    GY = BB.atoms.positions[0:184,1]
    GZ = BB.atoms.positions[0:184,2]

    RBD_X = BB.atoms.positions[184:256,0]
    RBD_Y = BB.atoms.positions[184:256,1]
    RBD_Z = BB.atoms.positions[184:256,2]

    Linker_X = BB.atoms.positions[256:264,0]
    Linker_Y = BB.atoms.positions[256:264,1]
    Linker_Z = BB.atoms.positions[256:264,2]

    CRD_X = BB.atoms.positions[264:303,0]
    CRD_Y = BB.atoms.positions[264:303,1]
    CRD_Z = BB.atoms.positions[264:303,2]

    CYFX = CYF.atoms.positions[0:,0]
    CYFY = CYF.atoms.positions[0:,1]
    CYFZ = CYF.atoms.positions[0:,2]

    # print(f'CYFX:{CYFX} CYFY:{CYFY} CYFZ:{CYFZ[0]}')
    # print(f'BB shape:{BB.indices}')
    # filename = os.path.join(os.path.join("./", f'bb_ids.npz'))
    # np.savez_compressed(filename, bb_ids = BB.indices)
    # exit(0)

    CRD_CYS_X = BB.atoms.positions[303:309,0]
    CRD_CYS_Y = BB.atoms.positions[303:309,1]
    CRD_CYS_Z = BB.atoms.positions[303:309,2]

    LINK1_X = BB.atoms.positions[309:387,0]
    LINK1_Y = BB.atoms.positions[309:387,1]
    LINK1_Z = BB.atoms.positions[309:387,2]

    pS365_X = BB.atoms.positions[387:398,0]
    pS365_Y = BB.atoms.positions[387:398,1]
    pS365_Z = BB.atoms.positions[387:398,2]

    LINK2_X = BB.atoms.positions[398:484,0]
    LINK2_Y = BB.atoms.positions[398:484,1]
    LINK2_Z = BB.atoms.positions[398:484,2]

    KD_X = BB.atoms.positions[484:751,0]
    KD_Y = BB.atoms.positions[484:751,1]
    KD_Z = BB.atoms.positions[484:751,2]

    pS729_X = BB.atoms.positions[751:766,0]
    pS729_Y = BB.atoms.positions[751:766,1]
    pS729_Z = BB.atoms.positions[751:766,2]

    a1433_X = BB.atoms.positions[767:996,0]
    a1433_Y = BB.atoms.positions[767:996,1]
    a1433_Z = BB.atoms.positions[767:996,2]

    b1433_X = BB.atoms.positions[997:1226,0]
    b1433_Y = BB.atoms.positions[997:1226,1]
    b1433_Z = BB.atoms.positions[997:1226,2]

    xCOM = BB.atoms.center_of_geometry()[0]
    yCOM = BB.atoms.center_of_geometry()[1]
    zCOM = BB.atoms.center_of_geometry()[2]
    markersize = 3.5
    line = 1.5
    box_size = boxsize #100
    projection = 'persp'
    Gcolor = [0,0,1,1]
    CYFcolor = [0.7,0.7,0.7,1]
    RBDcolor = [0,1,0,1]
    Linkcolor = [0.25,0.75,0.75,1]
    CRDcolor = [1,1,0,1]
    CRD_CYScolor = [1,0.5,0,1]
    LINK1color = [0.9,0,0.9,1]
    pS365color = [0.27,0,0.98,1]
    LINK2color = [1,0.6,0.6,1]
    KDcolor = [1,0,0,1]
    pS729color = [0.65,0,0.65,1]
    a1433color = [0.235,0.235,0.235,1]
    b1433color = [0.7,0.7,0.7,1]




    # ax = plt.axes(projection ="3d")
    # ax.set_aspect('equal')
    ax.set_box_aspect([1,1,1])

    if [i for i in ['G','ALL'] if i in parts]:
        ax.plot(   GX,   GY,   GZ,color=Gcolor,linewidth=line)
        ax.scatter(GX,   GY,   GZ,color=Gcolor,s=markersize, label="RAS G-domain and HVR")

    if [i for i in ['CYF','ALL'] if i in parts]:
        ax.plot(   CYFX, CYFY, CYFZ,color=CYFcolor,linewidth=line)
        ax.scatter(CYFX, CYFY, CYFZ,color=CYFcolor,s=markersize, label="CYF:RAS farnesyl")

    if [i for i in ['RBD','ALL'] if i in parts]:
        ax.plot(   RBD_X, RBD_Y, RBD_Z,color=RBDcolor,linewidth=line)
        ax.scatter(RBD_X, RBD_Y, RBD_Z,color=RBDcolor,s=markersize, label="RAF RBD")

    if [i for i in ['CRD','ALL'] if i in parts]:
        ax.plot(   Linker_X, Linker_Y, Linker_Z,color=Linkcolor,linewidth=line)
        ax.scatter(Linker_X, Linker_Y, Linker_Z,color=Linkcolor,s=markersize, label="RAF RBD-CRD linker")
        ax.plot(   CRD_X, CRD_Y, CRD_Z,color=CRDcolor,linewidth=line)
        ax.scatter(CRD_X, CRD_Y, CRD_Z,color=CRDcolor,s=markersize, label="CRD")
        ax.plot(   CRD_CYS_X, CRD_CYS_Y, CRD_CYS_Z,color=CRD_CYScolor,linewidth=line)
        ax.scatter(CRD_CYS_X, CRD_CYS_Y, CRD_CYS_Z,color=CRD_CYScolor,s=markersize, label="CRD-cys-loop")

    if [i for i in ['LINK1','ALL'] if i in parts]:
        ax.plot(   LINK1_X, LINK1_Y, LINK1_Z,color=LINK1color,linewidth=line)
        ax.scatter(LINK1_X, LINK1_Y, LINK1_Z,color=LINK1color,s=markersize, label="RAF Linker1")

    if [i for i in ['PS365','ALL'] if i in parts]:
        ax.plot(   pS365_X, pS365_Y, pS365_Z,color=pS365color,linewidth=line)
        ax.scatter(pS365_X, pS365_Y, pS365_Z,color=pS365color,s=markersize, label="RAF PS365 region")

    if [i for i in ['LINK2','ALL'] if i in parts]:
        ax.plot(   LINK2_X, LINK2_Y, LINK2_Z,color=LINK2color,linewidth=line)
        ax.scatter(LINK2_X, LINK2_Y, LINK2_Z,color=LINK2color,s=markersize, label="RAF Linker2")

    if [i for i in ['KD','ALL'] if i in parts]:
        ax.plot(   KD_X, KD_Y, KD_Z,color=KDcolor,linewidth=line)
        ax.scatter(KD_X, KD_Y, KD_Z,color=KDcolor,s=markersize,label="RAF Kinase domain")

    if [i for i in ['PS729','ALL'] if i in parts]:
        ax.plot(   pS729_X, pS729_Y, pS729_Z,color=pS729color,linewidth=line)
        ax.scatter(pS729_X, pS729_Y, pS729_Z,color=pS729color,s=markersize, label="RAF PS729")

    if [i for i in ['1433a','ALL'] if i in parts]:
        ax.plot(   a1433_X, a1433_Y, a1433_Z,color=a1433color,linewidth=line)
        ax.scatter(a1433_X, a1433_Y, a1433_Z,color=a1433color,s=markersize,label="First 14-3-3")

    if [i for i in ['1433b','ALL'] if i in parts]:
        ax.plot(   b1433_X, b1433_Y, b1433_Z,color=b1433color,linewidth=line)
        ax.scatter(b1433_X, b1433_Y, b1433_Z,color=b1433color,s=markersize, label="Second 14-3-3")


    ax.plot([xCOM+box_size/2,xCOM-box_size/2], [yCOM-box_size/2,yCOM+box_size/2],[CYFZ[0],CYFZ[0]],color=[0,1,1,1],linestyle='--')
    ax.set_proj_type(projection)
    ax.azim=45
    ax.elev=0
    ax.roll=0

    ax.set_zlabel('Z ($\AA$)')
    ax.set_ylabel('Y ($\AA$)')
    ax.set_xlabel('X ($\AA$)')
#     ax.set_zlim(CYFZ[0]-box_size/10.0*9,CYFZ[0]+box_size/10.0)

    # ax.set_zlim(0,CYFZ[0]+box_size/10.0)
    # ax.set_ylim(yCOM-box_size/2,yCOM+box_size/2)
    # ax.set_xlim(xCOM-box_size/2,xCOM+box_size/2)

    # ax.set_xlim(-50,130)
    # ax.set_ylim(-90,90)
    ax.set_zlim(-125,10)
    title = title + '\n' + f'[{d[0]:.02f}, {d[1]:.02f}]'
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), markerscale=3)

def plot_distance_matrix(matrix, filename, title, show):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    plt.imshow(matrix, interpolation='none', cmap='viridis')

    plt.colorbar()
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def plot_difference_distance_matrix(matrix, filename, title, show):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    plt.imshow(matrix, interpolation='none', cmap='bwr', vmin=-15, vmax=15)

    plt.colorbar()
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_rmsds_nice(data, title, filename, nbins=30, ref=None, show=True):
    assert isinstance(data, np.ndarray)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Use log scale on y-axis
    ax.set_yscale('log')

    # Histogram
    counts, bins, _ = ax.hist(data, bins=nbins, density=False, color='steelblue', edgecolor='black')

    # Add text labels (optional, less cluttered for log scale)
    for i in range(nbins):
        if counts[i] > 0:
            ax.text(bins[i], counts[i], f"{int(counts[i])}", fontsize=7, rotation=90, va='bottom')

    # Reference line
    if ref is not None:
        ax.axvline(x=ref, color='r', linestyle='--', linewidth=1.5)

    # Labels and Title
    ax.set_xlabel('Error – RMSD (Ångström)\n({} structures)'.format(data.shape[0]), fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title(title, fontsize=16)

    # Format y-axis ticks as powers of 10
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f'$10^{{{int(np.log10(y))}}}$' if y > 0 else '0'))

    # Save + Show
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_rmsds(data, title, filename, nbins=30, ref=None, show=True):

    assert isinstance(data, np.ndarray)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    # plt.yscale("log")
    arr = plt.hist(data, density=False, bins=nbins)
    for i in range(nbins):
        plt.text(arr[1][i],arr[0][i],"%.f" % (arr[0][i]), fontsize=8)

    if ref !=None:
        plt.axvline(x = ref, color = 'r')
    axs.set_ylabel('Frequency', fontsize=24)
    axs.set_xlabel('Error - RMSD (Ångström) ({} structures)'.format(data.shape[0]), fontsize=24)
    # axs.set_ylabel('Count')
    # axs.set_xlabel('Error - RMSD (Ångström) ({} structures)'.format(data.shape[0]))
    axs.set_title(title)

    # # plt.ylim((pow(10,-1),pow(10,5)))

    # plt.rcParams.update({'font.size': 20})
    plt.savefig(filename, bbox_inches='tight')
    # plt.savefig(filename +".eps", bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def  plot_mses(data, title, filename, nbins=30, ref=None, show=True):

    assert isinstance(data, np.ndarray)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    arr = plt.hist(data, density=False, bins=nbins)
    for i in range(nbins):
        plt.text(arr[1][i],arr[0][i],"%.f" % (arr[0][i]), fontsize=8)

    if ref !=None:
        plt.axvline(x = ref, color = 'r')
    axs.set_ylabel('Count')
    axs.set_xlabel('Error - MSE ({})'.format(data.shape[0]))
    axs.set_title(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
# ------------------------------------------------------------------------------
# training history
def plot_training_history(history, title, filename, show):

    fig, [ax1,ax2] = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    xmax = 0
    metrics= {}
    for k in history.keys():
        c = ccolors.get(k.split('_')[-1], 'k')
        t = '--' if k.split('_')[0] == 'val' else '-'
        if 'accuracy' in k:
            mn = np.nanmean(history[k], axis=0)
            x = np.arange(mn.shape[0])+1
            ax1.plot(x, mn, label=k, color=c, linestyle=t)
            metrics[k] = mn
        elif 'loss' in k:
            mn = np.nanmean(history[k], axis=0)
            x = np.arange(mn.shape[0])+1
            ax2.plot(x, mn, label=k, color=c, linestyle=t)
            metrics[k] = mn

    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax1.tick_params(labelright=True)

    ax1.grid(which='major', color='#dddddd', linewidth=0.8)
    ax1.grid(which='minor', color='#eeeeee', linestyle=':', linewidth=0.5)
    ax1.legend(loc="upper left", ncol=2)
    ax1.minorticks_on()

    ax1.set_title(title)

    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    ax2.tick_params(labelright=True)

    ax2.grid(which='major', color='#dddddd', linewidth=0.8)
    ax2.grid(which='minor', color='#eeeeee', linestyle=':', linewidth=0.5)
    ax2.legend(loc="upper left")
    ax2.minorticks_on()

    fig.text(0.5,0.04, "Epochs", ha="center", va="center")
    fig.text(0.08,0.5, "Metric Value", ha="center", va="center", rotation=90)

    plt.legend(ncol=2)
    plt.savefig(filename + ".png", bbox_inches='tight')
    np.savez_compressed(filename + ".npz", **metrics)

    if show:
        plt.show()
    plt.close()

# training history precision and recall
def plot_training_precision_recall(history, title, filename, show):

    fig, [ax1,ax2] = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    xmax = 0
    metrics= {}
    for k in history.keys():
        c = ccolors.get(k.split('_')[-1], 'k')
        t = '--' if k.split('_')[0] == 'val' else '-'
        if 'precision' in k:
            mn = np.nanmean(history[k], axis=0)
            x = np.arange(mn.shape[0])+1
            ax1.plot(x, mn, label=k, color=c, linestyle=t)
            metrics[k] = mn
        elif 'recall' in k:
            mn = np.nanmean(history[k], axis=0)
            x = np.arange(mn.shape[0])+1
            ax2.plot(x, mn, label=k, color=c, linestyle=t)
            metrics[k] = mn

    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    ax1.tick_params(labelright=True)

    ax1.grid(which='major', color='#dddddd', linewidth=0.8)
    ax1.grid(which='minor', color='#eeeeee', linestyle=':', linewidth=0.5)
    ax1.legend(loc="upper left", ncol=2)
    ax1.minorticks_on()

    ax1.set_title(title)

    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    ax2.tick_params(labelright=True)

    ax2.grid(which='major', color='#dddddd', linewidth=0.8)
    ax2.grid(which='minor', color='#eeeeee', linestyle=':', linewidth=0.5)
    ax2.legend(loc="upper left")
    ax2.minorticks_on()

    fig.text(0.5,0.04, "Epochs", ha="center", va="center")
    fig.text(0.08,0.5, "Metric Value", ha="center", va="center", rotation=90)

    plt.legend(ncol=2)
    plt.savefig(filename + ".png", bbox_inches='tight')
    np.savez_compressed(filename + ".npz", **metrics)

    if show:
        plt.show()
    plt.close()


# ------------------------------------------------------------------------------
# PLOT DISTRIBUTIONS
# ------------------------------------------------------------------------------
def plot_distribution_of_channels(hists, bins, xtype, title, filename, show):

    assert isinstance(hists, np.ndarray) and isinstance(bins, np.ndarray)
    assert len(hists.shape) == 2 and len(bins.shape) == 2
    assert hists.shape[1] == bins.shape[1] - 1

    assert xtype in ['symmetric', 'positive']
    assert hists.shape[0] == 14
    assert hists.shape[0] == bins.shape[0]

    fig, axs = plt.subplots(nrows=2, ncols=8, figsize=(12,4))
    for i in range(hists.shape[0]):
        plot_distribution(axs[i//8, i%8], hists[i], bins[i], xtype)

    axs[1,6].axis('off')
    axs[1,7].axis('off')

    if len(title) > 0:
        fig.suptitle(title)

    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_distribution_in_latentspace(data, filename, title, show):

    assert isinstance(data, np.ndarray)
    ndims = data.shape[1]
    dmax = np.abs(data).max()

    isz = 2
    if ndims < 20:
        ncols, nrows = ndims, 1
    else:
        ncols, nrows = 20, int(np.ceil(ndims/20.))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                            figsize=(isz * ncols, isz * nrows))

    for d in range(ndims):
        h, b = np.histogram(data[:, d], range=(-dmax, dmax), bins=40)
        if nrows == 1 and ncols == 1:
            ax = axs
        elif nrows == 1:
            ax = axs[d % ncols]
        else:
            ax = axs[d // ncols, d % ncols]
        plot_distribution(ax, h, b, 'symmetric')

    plt.suptitle(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_distribution_of_latent_distances(dists, d, filename, title, show):

    h, b = np.histogram(dists, range=(0, dists.max()), bins=80)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    plot_distribution(axs, h, b, 'positive')
    axs.set_ylabel('Distribution')
    axs.set_xlabel('Distance in {}-D Latent Space'.format(d))
    axs.set_title(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_distribution_of_rotation_errors(pdists, rotation_dists, rotation_labels,
                                         d, filename, title, show):

    fcolors = ['#aaaaaa', '#a6cee3', '#fdc086', '#beaed4', '#7fc97f', 'r', 'r']
    ecolors = ['#333333', '#1f78b4', '#bf5b17', '#f0027f', '#4daf4a', 'g', 'g']

    LOGGER.info('plotting rotation_errors')

    # find the max of all values to get consistent binning
    dmax = pdists.max()
    for i in range(len(rotation_dists)):
        dmax = max(dmax, rotation_dists[i].max())

    hmax = 0
    # plot on the same axes
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))

    h, b = np.histogram(pdists, range=(0, dmax), bins=80)
    h = plot_distribution(axs, h, b, 'positive',
                          'pairwise distances', fcolors[0], ecolors[0], 0.5)
    hmax = max(h, hmax)

    for l in range(len(rotation_dists)):
        h, b = np.histogram(rotation_dists[l], range=(0, dmax), bins=80)
        h = plot_distribution(axs, h, b, 'positive', rotation_labels[l]+' distances',
                              fcolors[l+1], ecolors[l+1], 0.5)
        hmax = max(h, hmax)

    axs.set_ylim(bottom=0) #[0, 1.1*hmax])
    axs.set_xlim(left=0)
    axs.set_ylabel('Distribution')
    axs.set_xlabel('Distance in {}-D Latent Space'.format(d))
    axs.set_title(title)

    plt.legend()
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_correlation_of_latentcoords(lcoords, filename, title, show):
    n = lcoords.shape[1]

    fig, axs = plt.subplots(nrows=n, ncols=n, figsize=(12, 12))

    for i in range(n):
        for j in range(n):
            ax = axs[i][j]

            if j <= i:
                ax.axis('off')
            else:
                hb = ax.hexbin(lcoords[:, i], lcoords[:, j],
                               gridsize=50, mincnt=1, cmap='inferno_r')

    if len(title) > 0:
        fig.suptitle(title)

    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# ------------------------------------------------------------------------------
# PLOT TSNE
# ------------------------------------------------------------------------------
def plot_tsne(encodings, labels1, filename, title, show):
    colors1 = ['k',
               '#a6cee3', 'orange', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
               '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
               '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
               '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
               'orange', 'r']
    colors2 = ['k', '#a6cee3', '#b2df8a', '#fb9a99']
    colors = ['k', '#a6cee3', '#b2df8a']

    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=0)
    x = tsne.fit_transform(encodings)

    # fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
    #                         figsize=(16, 8))
    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
    plt.rcParams.update({'font.size': 30})
    plot_labeled_scatters(x, labels1, colors1)
    # plot_labeled_scatters(x, labels1, colors, ax=axs[0])
    # plot_labeled_scatters(x, labels1, colors, ax=axs[1])

    # for i in range(2):
        # axs[i].set_xlabel('tsne dim 1')
        # axs[i].set_ylabel('tsne dim 2')

    # axs.set_xlabel('tsne dim 1')
    # axs.set_ylabel('tsne dim 2')

    # plt.rcParams.update({'font.size': 30})
    plt.xlabel('t-SNE dim 1', fontsize=30)
    plt.ylabel('t-SNE dim 2', fontsize=30)
    if len(title) > 0:
        fig.suptitle(title)
    # plt.xlim(-100, 100)
    # plt.ylim(-100, 100)


    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

# ------------------------------------------------------------------------------
# PLOT PCA
# ------------------------------------------------------------------------------
def plot_pca(encodings, labels1, filename, filename_xz, filename_xy, filename_yz, title, show):
    colors1 = ['k',
               '#a6cee3', 'orange', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
               '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
               '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
               '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
               'orange', 'r']
    colors2 = ['k', '#a6cee3', '#b2df8a', '#fb9a99']
    colors = ['k', '#a6cee3', '#b2df8a']

    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    x = pca.fit_transform(encodings)

    fig, axs = plt.subplots(1, 1, figsize=(16, 8))

    axs = plot_labeled_scatters3(x, labels1, colors1)

    var1 = round(pca.explained_variance_ratio_[0],3) * 100
    var2 = round(pca.explained_variance_ratio_[1],3) * 100
    var3 = round(pca.explained_variance_ratio_[2],3) * 100
    plt.xlabel('PC 1 ({:.2f}%)'.format(var1))
    plt.ylabel('PC 2 ({:.2f}%)'.format(var2))
    axs.set_zlabel('PC 3 ({:.2f}%)'.format(var3))

    if len(title) > 0:
        fig.suptitle(title)

    plt.savefig(filename, bbox_inches='tight')
    axs.view_init(elev=0, azim=-90)
    plt.savefig(filename_xz, bbox_inches='tight')
    axs.view_init(elev=90, azim=-90)
    plt.savefig(filename_xy, bbox_inches='tight')
    axs.view_init(elev=0, azim=0)
    plt.savefig(filename_yz, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()

# ------------------------------------------------------------------------------
# PLOT LDA
# ------------------------------------------------------------------------------
def plot_lda(encodings, labels1, filename, filename_xz, filename_xy, filename_yz, title, show):
    colors1 = ['k',
               '#a6cee3', 'orange', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
               '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
               '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
               '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
               'orange', 'r']
    colors2 = ['k', '#a6cee3', '#b2df8a', '#fb9a99']
    colors = ['k', '#a6cee3', '#b2df8a']

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=3)
    x = lda.fit(encodings, labels1).transform(encodings)

    fig, axs = plt.subplots(1, 1, figsize=(16, 8))

    # axs = plot_labeled_scatters(x, labels1, colors1)
    axs = plot_labeled_scatters3(x, labels1, colors1)


    plt.xlabel('Function 1')
    plt.ylabel('Function 2')
    axs.set_zlabel('Function 3')

    if len(title) > 0:
        fig.suptitle(title)

    plt.savefig(filename, bbox_inches='tight')
    axs.view_init(elev=0, azim=-90)
    plt.savefig(filename_xz, bbox_inches='tight')
    axs.view_init(elev=90, azim=-90)
    plt.savefig(filename_xy, bbox_inches='tight')
    axs.view_init(elev=0, azim=0)
    plt.savefig(filename_yz, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()
# ------------------------------------------------------------------------------
# PLOT correlation between two types of distances
# ------------------------------------------------------------------------------
def plot_distribution_of_x_vs_lspace_distances(ldists, rdf_dists, name,
                                               filename, title, show):

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    x, xlabel = rdf_dists, f'dist in {name} space'
    y, ylabel = ldists, 'dist in latent space'

    hb = plot_histo2d_hexbin(axs, x, y, xlabel, ylabel, 'log')
    cb = fig.colorbar(hb, ax=axs)
    cb.set_label('log10 counts')

    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# ------------------------------------------------------------------------------
# PLOT CONFUSION MATRIX
# ------------------------------------------------------------------------------
# def save_confusion_matrix(history, filename, show):

#     ## TODO: should not use tensorflow here!
#     import tensorflow as tf

#     for k in history.keys():
#         if 'confusion' in k and 'val' in k:
#             total = history[k].sum(axis=0)
#     total = total[-1]
#     numclasses = total.shape[0]

#     class_correct = tf.linalg.tensor_diag_part(total)
#     class_total = tf.reduce_sum(total, axis=1)
#     class_accuracy = class_correct / tf.maximum(1, class_total)

#     all_correct = tf.reduce_sum(class_correct)
#     all_total = tf.reduce_sum(class_total)
#     all_accuracy = all_correct / tf.maximum(1, all_total)

#     classes = []
#     for i in range(numclasses):
#         classes.append("class"+str(i))
#     LOGGER.info(f'total ({total})')
#     fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
#     plt.imshow(total, interpolation='nearest', extent=[0.5, numclasses + 0.5, numclasses + 0.5, 0.5], cmap='Blues')
#     axs.set_xlabel('\nPredicted Values')
#     axs.set_ylabel('Actual Values ')

#     tick_marks = np.arange(total.shape[0])+1
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     for (j,i),label in np.ndenumerate(total):
#         axs.text(i + 1, j + 1,label,ha='center',va='center')

#     plt.colorbar()
#     total_accuracy = all_accuracy.numpy() * 100
#     plt.title('Total accuracy %1.2f' %total_accuracy)
#     plt.savefig(filename, bbox_inches='tight')
#     if show:
#         plt.show()
#     plt.close()


# ------------------------------------------------------------------------------
# PLOT WRONGLY CLASSIFIED PATCH INTO TRANSITION PLOT FOR PROTEIN STATES
# ------------------------------------------------------------------------------
def plot_transition_protein_states(inpath, wrong_patchid, title, analysis_path, data_label, show):
    colors1 = ['k',
               '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
               '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
               '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
               '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
               'orange', 'r']

    assert data_label in ['C3', 'C4'], f'Invalid data label ({data_label}'

    actual_protein_transition = os.path.join(os.path.join(analysis_path, f'{title}_protein_transitions.png'))

    psplit = wrong_patchid.rfind('_')
    wrong_fileid = wrong_patchid[:psplit]
    wrong_patchid = int(wrong_patchid[psplit+1:])

    if data_label == 'C4':
        npatches_per_file = 2000
        filename = "patches_" + str(wrong_fileid) + ".npz"
        data = np.load(os.path.join(inpath, filename), allow_pickle=True)
    elif data_label == 'C3':
        npatches_per_file = 62318
        filename = str(wrong_fileid)
        data = np.load(filename, allow_pickle=True)

    protein_states = data['protein_states']

    y = np.empty(protein_states.shape[0], dtype='object')
    # LOGGER.info(f'filename : {filename}')
    # LOGGER.info(f'protein_states.ndim : {protein_states.ndim}')
    if data_label == 'C4':
        if protein_states.shape[1] == 1:
            y = protein_states
        else:
            for i in range(data['protein_states'].shape[0]):y[i]=data['protein_states'][i,0] + "-" + data['protein_states'][i,1]
    elif data_label == 'C3':
         # C3
        n = protein_states.shape[0]

        for i in range(n):
        # Doing this because the shape of protein states is not the same for all files in C3
            if len(protein_states.shape) == 2:
               ndims = protein_states.shape[1]
               if ndims == 1:
                    y[i] = str(protein_states[i,0])
               elif ndims == 2:
                    # change order if it is zRAFazRASa
                    if (protein_states[i,0] == "zRAFa"):
                        y[i] = str(protein_states[i,1]) + str(protein_states[i,0])
                    else:
                        y[i] = str(protein_states[i,0]) + str(protein_states[i,1])
            else:
                if protein_states[i].shape[0] == 1:
                    y[i] = str(protein_states[i][0])
                elif protein_states[i].shape[0] == 2:
                    # change order if it is zRAFazRASa
                    if (protein_states[i][0] == "zRAFa"):
                        y[i] = str(protein_states[i][1]) + str(protein_states[i][0])
                    else:
                        y[i] = str(protein_states[i][0]) + str(protein_states[i][1])

    # show 30 points before and after the wrong patch
    show_adjacent = True
    nadjacents = 30
    u, ind = np.unique(y, return_inverse=True)

    if show_adjacent:
        low_limit = wrong_patchid-nadjacents
        high_limit = wrong_patchid+nadjacents
        if low_limit < 0: low_limit = 0
        if high_limit >= len(ind): high_limit = len(ind) - 1
        time = list(range(low_limit, high_limit+1))
        ind = ind[low_limit:high_limit+1]
    else:
        time = list(range(0,npatches_per_file))

    fig, axs = plt.subplots(1, 1, figsize=(16, 8))
    if show_adjacent:
        plt.scatter(time, ind, s=100)
    else:
        plt.scatter(time, ind, s=2)
    plt.axvline(x=wrong_patchid, color='r')

    # plot_labeled_scatters(time, ind, colors1)

    plt.yticks(range(len(u)), u)
    plt.title(f'{title} (Patch_id:{wrong_patchid}) file:{filename}')
    plt.xlabel('Time History')
    plt.ylabel('Protein States')
    plt.savefig(actual_protein_transition, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


# ------------------------------------------------------------------------------
# PLOT VALIDATION DATA
# ------------------------------------------------------------------------------
# def plot_validation_data(encodings, patches, labels, predicted_labels,
#                          patchids, nlabels, analysis_path, inpath, data_label, save_analysis_data, show):

#     assert data_label in ['C3', 'C4'], f'Invalid data label ({data_label}'

#     from sklearn.manifold import TSNE
#     tsne = TSNE(n_components=2, random_state=0)
#     x = tsne.fit_transform(encodings)

#     name_correct_patches_all = os.path.join(os.path.join(analysis_path, f'correct_validation_all_classes.png'))
#     inshp = patches[1].shape
#     correct_patches_all = np.zeros((nlabels,) + inshp, dtype=np.float32)
#     correct_labels_all = np.zeros(nlabels, dtype=np.uint8)
#     if data_label == 'C4':
#         correct_patches_all_mu = np.zeros((nlabels,) + inshp, dtype=np.float32)
#         correct_labels_all_mu = np.zeros(nlabels, dtype=np.uint8)
#         correct_patches_all_mu0 = np.zeros((nlabels,) + inshp, dtype=np.float32)
#         correct_labels_all_mu0 = np.zeros(nlabels, dtype=np.uint8)
#         correct_patches_all_mu12 = np.zeros((nlabels,) + inshp, dtype=np.float32)
#         correct_labels_all_mu12 = np.zeros(nlabels, dtype=np.uint8)
#         correct_patches_all_mu18 = np.zeros((nlabels,) + inshp, dtype=np.float32)
#         correct_labels_all_mu18 = np.zeros(nlabels, dtype=np.uint8)

#     for label in range(nlabels):
#         ncorrectvalidation = 4
#         nwrongvalidation = 4
#         name_correct_patches = os.path.join(os.path.join(analysis_path, f'correct_validation_class{label}.png'))
#         name_wrong_patches = os.path.join(os.path.join(analysis_path, f'wrong_validation_class{label}.png'))
#         distance_wrong_patches = os.path.join(os.path.join(analysis_path, f'distance_wrong_patches{label}.txt'))

#         class_ids = np.where(labels == label)
#         if np.shape(class_ids)[1] > 0:
#             labels_class = labels[class_ids]
#             predicted_labels_class = predicted_labels[class_ids]
#             patches_class = patches[class_ids]
#             if patchids is not None:
#                 patchids_class = patchids[class_ids]

#             correct_indices = np.where(labels_class == predicted_labels_class)
#             if np.shape(correct_indices)[1] > 0:
#                 # calculate the centroid of the class cluster
#                 x_class = x[class_ids]
#                 mean_x = np.mean(x_class[correct_indices], axis=0)

#                 if ncorrectvalidation > np.shape(correct_indices)[1]: ncorrectvalidation = np.shape(correct_indices)[1]
#                 if ncorrectvalidation > 0:
#                     LOGGER.info(f'plotting correct validation data : {ncorrectvalidation} out of {np.shape(correct_indices)[1]} for label {label}')
#                     correct_patches = patches_class[correct_indices]
#                     correct_labels = labels_class[correct_indices]
#                     correct_patchids = patchids_class[correct_indices]
#                     plot_images(correct_patches[:ncorrectvalidation], correct_labels[:ncorrectvalidation], 'quant', name_correct_patches, show)

#                     correct_patches_all[label] = correct_patches[0]
#                     correct_labels_all[label] = correct_labels[0]
#                     if data_label == 'C4':
#                         correct_mu = [i for i in range(0,np.shape(correct_patchids)[0]) if "mu-" in correct_patchids[i]]
#                         correct_mu0 = [i for i in range(0,np.shape(correct_patchids)[0]) if "mu0-" in correct_patchids[i]]
#                         correct_mu12 = [i for i in range(0,np.shape(correct_patchids)[0]) if "mu12-" in correct_patchids[i]]
#                         correct_mu18 = [i for i in range(0,np.shape(correct_patchids)[0]) if "mu18.142-" in correct_patchids[i]]
#                         if len(correct_mu) > 0:
#                             correct_patches_all_mu[label] = correct_patches[correct_mu[0]]
#                             correct_labels_all_mu[label] = correct_labels[correct_mu[0]]
#                         if len(correct_mu0) > 0:
#                             correct_patches_all_mu0[label] = correct_patches[correct_mu0[0]]
#                             correct_labels_all_mu0[label] = correct_labels[correct_mu0[0]]
#                         if len(correct_mu12) > 0:
#                             correct_patches_all_mu12[label] = correct_patches[correct_mu12[0]]
#                             correct_labels_all_mu12[label] = correct_labels[correct_mu12[0]]
#                         if len(correct_mu18) > 0:
#                             correct_patches_all_mu18[label] = correct_patches[correct_mu18[0]]
#                             correct_labels_all_mu18[label] = correct_labels[correct_mu18[0]]
#                     # calculate the distance of correctly classified patches from the centroid of the class cluster
#                     x_correct = x_class[correct_indices]
#                     x_correct_sel = x_correct[:ncorrectvalidation]
#                     dist_cor = np.zeros(ncorrectvalidation)
#                     for i in range(ncorrectvalidation):
#                         dist_cor[i] = np.linalg.norm(x_correct_sel[i]-mean_x)
#                     mn_correct = np.mean(dist_cor, axis=0)
#                     std_correct = np.std(dist_cor, axis=0)

#                 wrong_indices = np.where(labels_class!=predicted_labels_class)
#                 if nwrongvalidation > np.shape(wrong_indices)[1]: nwrongvalidation = np.shape(wrong_indices)[1]
#                 if nwrongvalidation > 0:
#                     LOGGER.info(f'plotting wrong validation data : {nwrongvalidation} out of {np.shape(wrong_indices)[1]}')
#                     wrong_patches = patches_class[wrong_indices]
#                     wrong_labels = labels_class[wrong_indices]
#                     wrong_predicted_labels = predicted_labels_class[wrong_indices]
#                     if patchids is not None:
#                         wrong_patchids = patchids_class[wrong_indices]
#                     plot_images(wrong_patches[:nwrongvalidation], wrong_labels[:nwrongvalidation], 'quant', name_wrong_patches, show)

#                     # plot the wrong classified samples into transition plots for protein_states
#                     #if save_analysis_data:
#                     #    wrong_validation_data = os.path.join(analysis_path, f'wrong_validation_data_{label}.npz')
#                     #    np.savez_compressed(wrong_validation_data, wrong_patchids=wrong_patchids[0:nwrongvalidation], wrong_labels=wrong_labels[0:nwrongvalidation], wrong_predicted_labels=wrong_predicted_labels[0:nwrongvalidation] )

#                     if patchids is not None:
#                         for i in range(nwrongvalidation):
#                             title = f"Class{label}-{i}wrong (Actual:{wrong_labels[i]} Predicted:{wrong_predicted_labels[i]})"
#                             plot_transition_protein_states(inpath, wrong_patchids[i], title, analysis_path, show)

#                     # calculate the distance of wronlgy classified patches from the centroid of the class cluster
#                     x_wrong = x_class[wrong_indices]
#                     x_wrong_sel = x_wrong[:nwrongvalidation]

#                     dist = np.zeros(nwrongvalidation)
#                     for i in range(nwrongvalidation): dist[i] = np.linalg.norm(x_wrong_sel[i]-mean_x)
#                     with open(distance_wrong_patches, "w") as file:
#                         file.write(np.array2string(dist, formatter={'float_kind':lambda dist: "%.2f" % dist}) + "(mean:" + "{0:.2f}".format(mn_correct) + ", std:" + "{0:.2f}".format(std_correct) + ")")

#     plot_images(correct_patches_all, correct_labels_all, 'quant', name_correct_patches_all, show)
#     if save_analysis_data:
#         outfile_all = os.path.join(analysis_path, f'validation_data.npz')
#         np.savez_compressed(outfile_all, patches=correct_patches_all, labels=correct_labels_all)
#     if data_label == 'C4':
#         LOGGER.info(f'plotting correct validation data  for all labels for different mu')
#         name_correct_patches_all_mu = os.path.join(os.path.join(analysis_path, f'correct_validation_all_classes_mu-.png'))
#         name_correct_patches_all_mu0 = os.path.join(os.path.join(analysis_path, f'correct_validation_all_classes_mu0-.png'))
#         name_correct_patches_all_mu12 = os.path.join(os.path.join(analysis_path, f'correct_validation_all_classes_mu12-.png'))
#         name_correct_patches_all_mu18 = os.path.join(os.path.join(analysis_path, f'correct_validation_all_classes_mu18.142.png'))
#         plot_images(correct_patches_all_mu, correct_labels_all_mu, 'quant', name_correct_patches_all_mu, show)
#         plot_images(correct_patches_all_mu0, correct_labels_all_mu0, 'quant', name_correct_patches_all_mu0, show)
#         plot_images(correct_patches_all_mu12, correct_labels_all_mu12, 'quant', name_correct_patches_all_mu12, show)
#         plot_images(correct_patches_all_mu18, correct_labels_all_mu18, 'quant', name_correct_patches_all_mu18, show)
#         if save_analysis_data:
#             outfile_all_mu18 = os.path.join(analysis_path, f'validation_data_mu18.npz')
#             np.savez_compressed(outfile_all_mu18, patches=correct_patches_all_mu18, labels=correct_labels_all_mu18)

# ------------------------------------------------------------------------------
# PLOT CRD DISTANCE
# ------------------------------------------------------------------------------
def plot_crd_dist(data, title, filename, nbins=30, ref=None, show=True):

    assert isinstance(data, np.ndarray)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    arr = plt.hist(data, density=False, bins=nbins)
    for i in range(nbins):
        plt.text(arr[1][i],arr[0][i],"%.f" % (arr[0][i]), fontsize=8)

    if ref !=None:
        plt.axvline(x = ref, color = 'r')
    axs.set_ylabel('Count')
    axs.set_xlabel('CRD distance (Ångström) ({} structures)'.format(data.shape[0]))
    axs.set_title(title)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.savefig(filename, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


