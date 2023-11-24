import corner
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import gaussian_kde
from collections import namedtuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.lines as mpllines
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q

#-- reading raw pe samples --
def pos_samples(filename):
    
    file = h5py.File(filename, 'r')

    mchirp, q, chi_eff = np.array(file['Mc']), np.array(file['mass_ratio']), np.array(file['chi_eff'])    
    iota, ra, dec = np.array(file['iota']), np.array(file['ra']), np.array(file['dec'])
    dL =  np.array(file['dL'])
        
    return np.column_stack((mchirp, q, chi_eff, iota, ra, dec, dL))

def plot_corner(samples, fig=None, **kwargs):
    
    defaults_kwargs = dict(
            bins=20, smooth=0.9, label_kwargs=dict(fontsize=16),
            title_kwargs=dict(fontsize=16), color='#0072C1',
            truth_color='tab:orange', quantiles=None,
            #levels=[1 - np.exp(-0.74**2/2), 1 - np.exp(-1.32**2/2)],
            levels=[0.5, 0.9],
            plot_density=False, plot_datapoints=False, fill_contours=False,
            max_n_ticks=5, hist_kwargs=dict(density=True), show_titles=False, title_fmt=None)
    
    defaults_kwargs.update(kwargs)
    fig = corner.corner(samples, fig=fig, **defaults_kwargs)

    return fig

def plot_multiple(samples, colors, linestyle, measure=None, filename=None, \
                                  plot_density_arr=None, fill_contour_arr=None, save=True, **kwargs):
    
    kwargs.get('hist_kwargs').update(ls=linestyle[0])
    kwargs.get('contour_kwargs').update(linestyles=linestyle[0])
    fig = plot_corner(samples[0], fig=None, **kwargs)
    
    lines = []
    
    for i, samps in enumerate(samples):
                
        if colors:
            c = colors[i]
            
        if linestyle:
            ls = linestyle[i]
            
        kwargs.get('hist_kwargs').update(color=c)
        kwargs.get('hist_kwargs').update(ls=ls)
        kwargs.get('contour_kwargs').update(colors=c)
        kwargs.get('contour_kwargs').update(linestyles=ls)
        fig = plot_corner(samps, fig=fig, **kwargs)
        lines.append(mpllines.Line2D([0], [0], color=c, lw=1))
        
    tlabels = kwargs.get('tlabels')    
    axes = fig.get_axes()
    
    loc = ['left', 'right']
    labels = kwargs.get('labels')
        
    for j, (values1, values2) in enumerate(zip(samples[0].T, samples[1].T)):
        
        ax = axes[j + j * samples[0].shape[1]]
        
        q1 = calc_quant(values1, kwargs['bins'], measure=measure)
        ax.axvline(x=q1[0], color=colors[0], ls='--', lw=kwargs['hist_kwargs']['lw'])
        ax.axvline(x=q1[1], color=colors[0], ls='--', lw=kwargs['hist_kwargs']['lw'])
        ax.axvline(x=q1[2], color=colors[0], ls='--', lw=kwargs['hist_kwargs']['lw'])
        
        q2 = calc_quant(values2, kwargs['bins'], measure=measure)
        ax.axvline(x=q2[0], color=colors[1], ls='--', lw=kwargs['hist_kwargs']['lw'])
        ax.axvline(x=q2[1], color=colors[1], ls='--', lw=kwargs['hist_kwargs']['lw'])
        ax.axvline(x=q2[2], color=colors[1], ls='--', lw=kwargs['hist_kwargs']['lw'])
            
    #  Add the titles
    for j, (value1, value2) in enumerate(zip(samples[0].T, samples[1].T)):
        
        ax = axes[j + j * samples[0].shape[1]]
        ax.text(0, 1.3, tlabels[0][j], c=colors[0], transform=ax.transAxes, **kwargs['title_kwargs'])
        
        if(j==1 or j == 4 or j == 5):
        
            ax.text(0.23, 1.1, tlabels[1][j], c=colors[1], transform=ax.transAxes, **kwargs['title_kwargs'])
            
        if(j==3):
        
            ax.text(0.21, 1.1, tlabels[1][j], c=colors[1], transform=ax.transAxes, **kwargs['title_kwargs'])
            
        if(j==6):
        
            ax.text(0.28, 1.1, tlabels[1][j], c=colors[1], transform=ax.transAxes, **kwargs['title_kwargs'])    
        
        if(j==0):
            
            ax.text(0.29, 1.1, tlabels[1][j], c=colors[1], transform=ax.transAxes, **kwargs['title_kwargs'])
            
        if(j==2):
            
            ax.text(0.34, 1.1, tlabels[1][j], c=colors[1], transform=ax.transAxes, **kwargs['title_kwargs'])
    
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=kwargs.get('label_kwargs').get('fontsize')-5)
                
    # Rescale the axes
    for i, ax in enumerate(fig.axes):
        ax.autoscale()
    plt.draw()
        
    labels = kwargs.get('methods')    
    axes = fig.get_axes()
    ndim = int(np.sqrt(len(axes)))
    axes[ndim - 1].legend(lines, labels, fontsize=kwargs.get('label_kwargs').get('fontsize'))
    
    if save:
        
        if filename:
            
            fig.savefig(fname=filename, dpi=200, bbox_inches='tight', pad_inches=0.1)

        else:
            
            filename = 'corner_comparison.jpeg'
            fig.savefig(fname=filename, dpi=200, bbox_inches='tight', pad_inches=0.1)

    return fig

def title_formats(samps, labels, titles, fmt_arr, measure=None, bins=None, flag=None):
    
    range_vals = []
    tlabels = np.empty(7, dtype=object)

    for p in range(samps.shape[1]):
        
        kernel = gaussian_kde(samps[:,p])
        count, val = np.histogram(samps[:,p], bins)
        val_pdf = kernel.pdf(val)
        map_val = val[np.argmax(val_pdf)]
        
        q_5, q_50, q_95 = np.quantile(samps[:,p], [0.05, 0.5, 0.95])

        if(measure == 'median'):
            
            q_m, q_p = q_50-q_5, q_95-q_50
            cent_val = q_50
            
        else:
            
            q_m, q_p = map_val-q_5, q_95-map_val
            cent_val = map_val

        title_fmt=fmt_arr[p]
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(cent_val), fmt(q_m), fmt(q_p))
        
        if(flag=='first'):
            title = "{0} = {1}".format(titles[p], title)
            tlabels[p]=title
        else:
            title = "{0}".format(title)
            tlabels[p]=title
            
    return tlabels

def calc_quant(samps, bins, measure=None):
    
    q_5, q_50, q_95 = np.quantile(samps, q=[0.05, 0.5, 0.95])
    kernel = gaussian_kde(samps)
    count, val = np.histogram(samps, bins)
    val_pdf = kernel.pdf(val)
    map_val = val[np.argmax(val_pdf)]
    
    if(measure=='median'):
        
        return [q_5, q_50, q_95]
    
    else:
        
        return [q_5, map_val, q_95]

def ecdf(samps):
    
    vals = np.sort(samps)
    #calculate CDF values
    cdf = np.arange(len(samps)) / (len(samps))
    
    return vals, cdf
