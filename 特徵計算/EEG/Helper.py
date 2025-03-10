import numpy as np
from scipy import signal, stats
import pywt
from scipy.stats.stats import pearsonr

def cor_print(g1,g2):
    c_v, p_v = pearsonr(g1,g2)
    if p_v < 0.001:
        return (c_v, '***')
    elif p_v < 0.01:
        return (c_v, '**')
    elif p_v < 0.05:
        return (c_v, '*')
    else:
        return (c_v, '')

def Printpvalue(g1,g2):
    p_val = t_test(g1,g2)[1]
    if p_val < 0.001:
        print('***')
    elif p_val < 0.01:
        print('**')
    elif p_val < 0.05:
        print('*')
        
def t_test(group1, group2):
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    
    std1 = np.std(group1)
    std2 = np.std(group2)
    
    nobs1 = len(group1)
    nobs2 = len(group2)
    
    modified_std1 = np.sqrt(np.float32(nobs1)/
                    np.float32(nobs1-1)) * std1
    modified_std2 = np.sqrt(np.float32(nobs2)/
                    np.float32(nobs2-1)) * std2
    (statistic, pvalue) = stats.ttest_ind_from_stats( 
               mean1=mean1, std1=modified_std1, nobs1=nobs1,   
               mean2=mean2, std2=modified_std2, nobs2=nobs2)
    return statistic, pvalue

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.
    https://raphaelvallat.com/bandpower.html?utm_source=pocket_mylist
    Requires MNE-Python >= 0.14.

    Parameters
    ----------
    data : 1d-array
      Input signal in the time-domain.
    sf : float
      Sampling frequency of the data.
    band : list
      Lower and upper frequencies of the band of interest.
    method : string
      Periodogram method: 'welch' or 'multitaper'
    window_sec : float
      Length of each window in seconds. Useful only if method == 'welch'.
      If None, window_sec = (1 / min(band)) * 2.
    relative : boolean
      If True, return the relative power (= divided by the total power of the signal).
      If False (default), return the absolute power.

    Return
    ------
    bp : float
      Absolute or relative band power.
    """
    from scipy.integrate import simps
    #from mne.time_frequency import psd_array_multitaper

    band = np.asarray(band)
    low, high = band

    #sd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0)
    freqs, psd = signal.welch(data, sf, nperseg=window_sec * sf)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def EEGbandpower(data, sf, band, relative=True):
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band
    #sd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0)
    freqs, psd = signal.welch(data, sf, nperseg=data.shape[0])

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find index of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using parabola (Simpson's rule)
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

def GetDataPointsNum(ms_time : int, sf : int):
    '''
    GetDataPointsNum, get data points number with sample rate from ms time unit
    
    ms_time: million seconds
    '''
    
    ms_time /= 1000
    return int(ms_time * sf)

def CalVectorDist(l : dict, d: dict):
    s = 0
    s += abs(l["X"] - d["X"])
    s += abs(l["Y"] - d["Y"])
    s += abs(l["Z"] - d["Z"])
    return s

def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array.
    [python - How to calculate a Gaussian kernel matrix efficiently in numpy? - Stack Overflow](https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy)
    """
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

def normalize(x, method='standard', axis=None):
	x = np.array(x, copy=False)
	if axis is not None:
		y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
		shape = np.ones(len(x.shape))
		shape[axis] = x.shape[axis]
		if method == 'standard':
			res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
		elif method == 'range':
			res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
		elif method == 'sum':
			res = x / np.float_(np.sum(y, axis=1).reshape(shape))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	else:
		if method == 'standard':
			res = (x - np.mean(x)) / np.std(x)
		elif method == 'range':
			res = (x - np.min(x)) / (np.max(x) - np.min(x))
		elif method == 'sum':
			res = x / float(np.sum(x))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	return res

def KLD(p, q):
    EPSILON = np.finfo('float').eps
    p = normalize(p, method='sum')
    q = normalize(q, method='sum')
    return np.sum(np.where(p != 0, p * np.log((p+EPSILON) / (q+EPSILON)), 0))



def TimeFrequencyCWT(data,fs,totalscal,wavelet='morl'):
    # 采样数据的时间维度
    t = np.arange(data.shape[0])/fs
    # 中心频率
    wcf = pywt.central_frequency(wavelet=wavelet)
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam/np.arange(totalscal, 0., -0.1)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0/fs)
    
    return t, frequencies, cwtmatr