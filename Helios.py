# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:01:08 2021

@author: Daniel - Copied from Helios TA Code



# Helios Code
## *Dan Farbowitz*
General code to take CSVs pre-processed through SurfaceXplorer *(background, bad data, and chirp correction)* and C60 ground-state absorbance data *(if excitation density is desired)* and perform analysis. 

### Inputs:


1.   *Helios data*
- Folder in Google Drive with pre-processed CSVs
- Files named with identifier at beginning and metadata filled in or all relevant info (identifier, UvVis/NIR, pump power, pump WL) 
2.   *Relevant Feature Wavelengths* 
- Excitons, Bound Polarons, Free Polarons
- Overlapping WLs (use Pyglotaran?)
3. *Models for Kinetics, including known time constants*
4. *C60 data in separate folder with identifiers in name*


### Outputs:

1. Spectral Graphs, log scale time stamps
2. Kinetics Graphs w/best-fits
3. Solar cell internal efficiencies - exciton diffusion to interface, charge transfer, charge dissociation from interface, charges carried to interface (Can we estimate quantities like V_OC and J_SC?)



"""

#introducing libraries
#pip install numpy scipy pandas matplotlib lmfit colour pyglotaran
#calling up libraries
from datetime import datetime
import pandas as pd
import numpy as np
import scipy
from scipy import optimize
import scipy.integrate as integrate
from scipy.integrate import quad
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits import mplot3d
from datetime import timedelta
import statistics
from matplotlib.collections import LineCollection 
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.signal import argrelmax
#import CSV data from drive
import glob
import os

#%%
#Setup: Defining some useful functions

#https://codereview.stackexchange.com/questions/189319/nearest-neighbor-algorithm-general-neares
def find_nearest(array, number, direction): 
    if direction is None:
        idx = (np.abs(array - number)).min()
    elif direction == 'backward':
        _delta = number - array
        _delta_positive = _delta[_delta > 0]
        if not _delta_positive.empty:
            idx = _delta_positive.min()
    elif direction == 'forward':
        _delta = array - number
        _delta_positive = _delta[_delta >= 0]
        if not _delta_positive.empty:
            idx = _delta_positive.min()
    return idx

def find_nearest_dex(array, number, direction=None): 
    idx = -1
    if direction is None:
        ser = np.abs(array-number)
        idx = ser.get_loc(ser.min())
    elif direction == 'backward':
        _delta = number - array
        _delta_positive = _delta[_delta > 0]
        if not _delta_positive.empty:
            idx = _delta.get_loc((_delta_positive.min()))
    elif direction == 'forward':
        _delta = array - number
        _delta_positive = _delta[_delta >= 0]
        if not _delta_positive.empty:
            idx = _delta.get_loc(_delta_positive.min())
    return idx

#Assuming power is in filename:
def power(name):
  for word in name.split(sep=' '):
    if 'mW' in word:
      val = int(1000*float(word.replace('mW', '')))
      return val
    elif 'uW' in word:
      val = int(float(word.replace('uW', '')))
      return val
  return np.nan()

def fluence_translate(name, OD=0.4):
  #note units are uJ/cm^2
  P = power(name)
  #return P*beam_prop/(f*beam_area)  

  #Excitation Density, instead!
  identifier = name[0:2]
  OD = OD_value(identifier)
  #photon_E in eV
  photon_E = hc/pump_WL
  photons_absorbed = 1 - (10**(-OD))
  return photons_absorbed*P*beam_prop/(f*photon_E*beam_area)
  #E per pulse per area
  


def input_features(*WLs):
  for WL in WLs:
    print(WL)

#%%
#Set important values 
#hc in nm*eV
hc = 1239.842
#beam f in Hz
f=10000
#beam area in cm^2
beam_area = 0.0001
beam_prop = 1
#Pump wavelength in nm
pump_WL = 532
#dictionary to translate any other indicators in filename
name_id ={'01':'Chlorobenzene Blended Film','02':'Chloroform Blended Film','03':'ortho-Xylene Blended Film','04':'CS_2 Acetone Blended Film','05':'Chlorobenzene PBDB-T Film','06':'Chloroform PBDB-T Film','07':'ortho-Xylene PBDB-T Film','08':'CS_2 Acetone PBDB-T Film','09':'Chlorobenzene ITIC Film','10':'Chloroform ITIC Film','11':'ortho-Xylene ITIC Film','12':'CS_2 Acetone ITIC Film'}

#%%
#Import steady-state data and prep for power/area per pulse calculation
folder = 'C:/Users/Daniel/Desktop/PBDBT_ITIC Cells/Steady state (Cary60) data'
os.chdir(folder)

all_files = glob.glob(os.path.join(folder, '*.csv'))
steady_state_data = {}

#assign OD values
def OD_value(sample_name, pump_WL=pump_WL):
  sorbs = []
  for name in steady_state_data:
    if sample_name in name:
      WL = find_nearest(steady_state_data[name].index, pump_WL)
      sorbs.append(steady_state_data[name].loc[WL])
  return np.average(sorbs)


#%%
#Handling TA Data
""" Dealing with ufs file vs csv exported from SurfaceXplorer"""

target_folders = ['C:/Users/Daniel/Desktop/Pre-Processed TA data']



TA_data = {}

for folder in target_folders:
    os.chdir(folder)
    all_files = glob.glob(os.path.join(folder, '*.csv'))

    for filename in all_files:
      #Simplify name
      new_name = filename.replace(".csv", "").replace(folder+"\\","")
      #import data
      df = pd.read_csv(filename, sep=',', header=0)

      #Some cleaning
      if '.ufs' not in new_name:
          df.drop(df.tail(14).index, inplace=True)
      df = df.rename({'0.00000E+0.1':'0.000000000'}, axis=1)
      cols = df.columns.values
      cols[0] = 'Wavelength(nm)'
      df.columns = cols
      df = df.set_index('Wavelength(nm)')
      #remove rows (wavelengths) with at least 3 na values
      df = df.dropna(axis=0, thresh=3)
    
      #change strings to numbers
      df.index = pd.to_numeric(df.index)
      for name in df.columns:
        df[name] = pd.to_numeric(df[name])
      df.columns = pd.to_numeric(df.columns)
      
      """#Change filename based on name_id translation
      for part in new_name.split(sep=' '):
          if part in name_id.keys():
              new_name = new_name.replace(part, name_id(part))"""
    
      #Rewrite to determine based on Wavelength given
      if (df.index > 600).all():
        TA_data[new_name + ' NIR'] = df
      elif (df.index < 1000).all():
        TA_data[new_name + 'UV-Vis'] = df
      else:
        TA_data[new_name + '???'] = df
        

  

TA_data

#import C60 data
#solution data


"""
# Adjustments:
- Label things without 'UvVis' 
- Assign power
- Use steady-state data to find absorbance at pump WL
- Combine power, absorbance, and pump laser metadata for the day to get Pump Energy per Pulse or Excitation Density

"""













#%%
# Importing data from google drive folder, sorting each into absorbance, spectroscope runs, and general spectroscope data.

#remember to remove laser signal
merged_data = {}
for name1 in TA_data:
  for name2 in TA_data:
    if (name1 != name2) & (name1[0:2] == name2[0:2]):
      comb = pd.concat([TA_data[name1], TA_data[name2]])
      if 'UvVis' not in name1:
        new_name = name1[0:12]
      elif 'UvVis' not in name2:
        new_name = name2[0:12]
      merged_data[new_name] = comb.sort_index()

merge_interp = {}
for name in merged_data:
  merge_interp[name] = (merged_data[name].T).interpolate().T

merge_interp

TA_data

#Print out Spectral data on log scale
def spec_time_data(name, times):
  plt.subplots()
  n = len(times)
  for i in range(len(times)):
    
      plt.plot(np.asarray(TA_data[name].index), np.asarray(TA_data[name][times[i]]),  label=str(dex[i])+' ps', color=colormap(i/n))
      plt.legend(title='Time from t0', bbox_to_anchor=(1,1), loc="upper left")
      plt.title(name + " Transient Spectroscopy")
      plt.xlabel('Energy(eV)')
      plt.ylabel('ΔA')
'''#color method from stack exchange :https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
def colorFader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
c1='white' 
c2='black' '''
#better color method?
colormap = cm.get_cmap('viridis')
#find global maxes to fix TA scale
maxes=[]
mins=[]
'''bmaxes=[]
bmins=[]
pmaxes = []
pmins = []
imaxes=[]
imins=[]'''
for name in TA_data:
  '''if int(name[0:2]) <= 4:
    bmaxes.append(TA_data[name].max().max())
    bmins.append(TA_data[name].min().min())
  elif int(name[0:2]) > 4 and int(name[0:2]) <= 8:
    pmaxes.append(TA_data[name].max().max())
    pmins.append(TA_data[name].min().min())
  elif int(name[0:2]) > 8:
    imaxes.append(TA_data[name].max().max())
    imins.append(TA_data[name].min().min())'''
  maxes.append(TA_data[name].max().max())
  mins.append(TA_data[name].min().min())
'''bmax = max(bmaxes)
bmin = min(bmins)
pmax = max(pmaxes)
pmin = min(pmins)
imax = max(imaxes)
imin = min(imins)  
gmax = max(maxes)
gmin = min(mins)
print(imaxes)
print(imins)'''
#routine for switching between type comparisons
'''def switch(name, a, b, c):
  if int(name[0:2]) <= 4:
    return a
  elif int(name[0:2]) > 4 and int(name[0:2]) <= 8:
    return b
  elif int(name[0:2]) > 8:
    return c'''

for name in TA_data:
  dex = [0, 1, 5, 10, 50, 100]
  times = []
  for i in range(len(dex)):
    times.append(find_nearest(TA_data[name].columns,dex[i], 'forward')+dex[i])
  spec_time_data(name, times)
  plt.plot(np.array(TA_data[name].index), np.zeros(len(TA_data[name].index), dtype=float, order='C'), color='black')
  plt.show()


  print(name)
  plt.show()



#%%
# Can skip, but if you want a SurfaceXplorer-type view...
#method from mathplotlib documentation
"""
viridis = cm.get_cmap('coolwarm', 100)
def plot_examples(colormaps, data, name):
    
    #Helper function to plot data with associated colormap.
    
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 5),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data.index,np.log10(data.columns+6), data.T, cmap=cmap, rasterized=True, vmin=-0.02, vmax=0.02)
        fig.colorbar(psm, ax=ax)
        ax.title.set_text(name)
    plt.xlim(0.9,3.0)
    yvals = np.asarray([-1, 0, 1, 5, 10, 50, 100, 500, 1000])
    ylogs = np.around(np.log10(yvals + 6), 2)
    plt.ylim(ylogs[0], ylogs[-1])
    plt.yticks(ylogs, labels=yvals)
    if "ITIC" in name:
      peaks = [1.78, 1.93]
    elif "PBDBT" in name:
      peaks = [1.97, 2.12, 2.92]
    elif 'blend' in name:
      peaks = [1.78, 1.93,1.97, 2.12, 2.92]
    else:
      peaks=[]
      
    for eg in peaks:
      plt.axvline(eg, ls='-.', lw=1.2, c='green')

    plt.xlabel('Energy(eV)')
    plt.ylabel('Time (ps)')
for name in merged_data:
  merged_data[name] = (merged_data[name].T).interpolate().T

  plot_examples([viridis], merged_data[name], name)
  plt.show()
"""
#%%

#Kinetic models
from scipy.optimize import curve_fit
from lmfit import Model, Parameter, Parameters, report_fit
def single_exp(x, a1, t1):
  return a1 * np.exp(-(x)/t1)

def double_exp(x, a1, a2, t1, t2):
  return a1*np.exp(-1*(x)/t1) + (a2)*np.exp(-1*(x)/t2)

def tri_exp(x, a1, t1, a2, t2, a3, t3):
  return a1*np.exp(-1*(x)/t1) + (a2)*np.exp(-1*(x)/t2) + (a3)*np.exp(-1*(x)/t3)

def power_law(x,a1,t1):
  return t1/(x**a1)

def power_and_exp(x, a1, t1, a2, t2):
  return a1 * np.exp(-(x)/t1) + a2/(x**t2)

def k_fit(x, y, func, name=name):
  #find time of peak
  tmax = x[pd.DataFrame(y).idxmax()]
  x_fit = x[x>=tmax]
  y_fit = y[x>=tmax]
  model = Model(func, independent_vars=['x'])
  params = Parameters()
  params.add('a1', value=0.5, min=-1, max=1)
  params.add('t1', value=5, min=0, max=700000)


  result = model.fit(y_fit, params, x=x_fit)

  

  a1 = np.around(result.values['a1'], 2)

  t1 = np.around(result.values['t1'], 1)



  df = pd.DataFrame(data={'a1':a1, 't1':t1}, index=[0])
  print(df)

  plt.plot(x_fit, result.best_fit, '--', color=colormap(i/10), label='Best Fit')
  plt.legend(loc='best')
  return result


WLs_of_interest = [850, 969, 1130]
WL_UV = sorted(i for i in WLs_of_interest if i <= 800)
WL_NIR = sorted(i for i in WLs_of_interest if i > 800)




colormap = cm.get_cmap('plasma')
styles = [':', '--', '-.', (0, (3, 5, 1, 5)), (0, (1, 10)), (0, (1, 1)), (0, (3, 10, 1, 10, 1, 10)), (0, (5, 1)), (0, (3, 10, 1, 10)), (0, (5, 5)),(0, (3, 1, 1, 1)),  (0, (1, 1))]

#Kinetics of excitons 






def k_chart(data, name, guess, func):

  dex = find_nearest_dex(data.index, guess)
  region = ''
  if guess > 800:
    region = 'NIR'
  elif guess <= 800:
    region = 'UV-Vis'
  #dd = data.iloc[dex-10:dex+10].mean()
  #data2 = dd[dd.index>1]
  #data3 = dd[(dd.index>1) & (dd.index<10)]
  #data4 = dd[(dd.index>400) & (dd.index<4500)]
  #update = pd.DataFrame(data4, columns = ['vals'])
  #slope = linregress(np.asarray(data4.index), np.asarray(np.log10(abs(data4))))[0]
  #intercept = linregress(np.asarray(data4.index), np.asarray(np.log10(abs(data4))))[1]
  #stderr = linregress(np.asarray(data4.index), np.asarray(np.log10(abs(data4))))[4]
  #update['estimate'] = 10**(slope*update.index+intercept)
  #+" -- Slope: "+str(np.around(slope, 6))+ " Std Err: "+str(np.around(stderr,8))

  
  
  i = WLs.index(guess)
  label = str(guess)+' nm'

  peak_to_norm =  data.iloc[dex-10:dex+10].mean().max()
  trough_to_norm = data.iloc[dex-10:dex+10].mean().min()
  norm = max(abs(peak_to_norm), abs(trough_to_norm))
  x = np.asarray(data.columns)

  y = abs(np.asarray(data.iloc[dex-10:dex+10].mean())/norm)
  plt.scatter(x,y, label=label)
  #plt.plot(np.asarray(update.index), np.asarray(update['estimate']), linewidth=0.8, color='black')

  k_fit(x, y, power_law, name)
  
  plt.ylabel('ΔA (normalized)')
  plt.yscale('log')
  plt.xscale('symlog')
  plt.ylim(0.01,1.2)
  plt.xlim(0,6000)
  plt.xlabel('Time (ps)')








def k_chart_bk(data, guess, label):
  f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')

  # plot the same data on both axes
  #ax.plot(x, y)
  #ax2.plot(x, y)

  ax.set_xlim(0,7.5)
  ax2.set_xlim(40,42.5)

  # hide the spines between ax and ax2
  ax.spines['right'].set_visible(False)
  ax2.spines['left'].set_visible(False)
  ax.yaxis.tick_left()
  ax.tick_params(labelright='off')
  ax2.yaxis.tick_right()

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

  d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
  kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
  ax.plot((1-d,1+d), (-d,+d), **kwargs)
  ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

  kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
  ax2.plot((-d,+d), (1-d,1+d), **kwargs)
  ax2.plot((-d,+d), (-d,+d), **kwargs)

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'


for name in sorted(TA_data):
  if 'UV-Vis' in name:
    WLs = WL_UV
  elif 'NIR' in name:
    WLs = WL_NIR
    k_chart(TA_data[name], name, 969, single_exp)
    title  = name + ': ' +str('')+' uJ/pulse/cm^2'
    plt.title(title)
    plt.legend()
    plt.show()
#%%
#give rate (1/time constant) of exciton lifetime under weak illumination
k = 1/60


#Gamma function
def gamma(kinetic, k):
    t = kinetic.columns
    Y = np.exp(-k*t) / kinetic
    dY = []
    dt = []
    for i in (len(Y) - 1):
        dY[i] = Y[i+1] - Y[i]
        dt[i] = t[i+1] - t[i]
    der = np.asarray(dY)/np.asarray(dt)
    t = t.drop(t[-1])
    gamma = der*np.exp(k*t)*2
    return gamma
    


#%%
"""
#Various transient bandwidth data

for name in spec_groups:
  heights = []
  centers = []
  sigmas = []
  times = []
  start_time = spec_groups[name][0].split()[1]
  for spec in spec_groups[name]:
    beam_fit(laser_data[spec])
    centers.append(laser_data[spec].center)
    heights.append(laser_data[spec].height)
    sigmas.append(laser_data[spec].sigma)
    times.append(float((pd.to_timedelta(str(TA_data[spec].rec_time)) - pd.to_timedelta('0 days ' + start_time))/timedelta(minutes = 1)))
  plt.subplot(2,2,1)
  plt.plot(times, heights)
  plt.xlabel('Time(min)')
  plt.ylabel('Height(counts)')
  plt.subplot(2,2,2)
  plt.plot(times, centers)
  plt.xlabel('Time(min)')
  plt.ylabel('Center(WL(nm))')
  plt.subplot(2,2,3)
  plt.plot(times, sigmas)
  plt.xlabel('Time(min)')
  plt.ylabel('Sigmas(WL(nm))')
  plt.suptitle(name + " Laser Beam Data")
  plt.show()
  print('Wavelength Average: ' + str(statistics.mean(centers)))

"""

#%%

#Running Pyglotaran 
from glotaran.utils.ipython import display_file
from glotaran.analysis.optimize import optimize
from glotaran.io import load_model
from glotaran.io import load_parameters
from glotaran.io import save_dataset
from glotaran.io.prepare_dataset import prepare_time_trace_dataset
from glotaran.project.scheme import Scheme
from glotaran.examples.sequential import dataset
import xarray as xr

dataset

#change up data to xarray
df = TA_data['01cb nir.ufs NIR']
#stacking to create multiindex (better for to_xarray function, match sample data convention)
def pandas_to_xarray(df):
    df = (
        df.T
        .stack()
        .rename_axis(index={'Wavelength(nm)': 'spectral', None: 'time'})
        .rename('data')
        .reset_index()
    )
    df = df.set_index(['time', 'spectral'])
    xarr = df.to_xarray()
    return xarr
dataset = pandas_to_xarray(df)

#example kinetic plot
plot_data = dataset.data.sel(spectral=[1000, 1050, 1200], method="nearest")
plot_data.plot.line(xscale="symlog", x="time", aspect=2, size=5);
#example spectral plot
plot_data = dataset.data.sel(time=[1, 10, 20], method="nearest")
plot_data.plot.line(x="spectral", aspect=2, size=5);

#Start with Singular value decomposition
dataset = prepare_time_trace_dataset(dataset)
dataset
#Guess at number of elements from values, if they're small and don't vary much, probably not worth including
plot_data = dataset.data_singular_values.sel(singular_value_index=range(0, 10))
plot_data.plot(yscale="log", marker="o", linewidth=0, aspect=2, size=5);
#Shows the input model(? Can't we make a generalized version in python?)
display_file("model.yaml", syntax="yaml")
model = load_model("model.yaml")
#Check model (?)
model.validate()

#Make Parameters
display_file("parameters.yaml", syntax="yaml")
parameters = load_parameters("parameters.yaml")
#Validate again
model.validate(parameters=parameters)
#print the model to inspect
model
parameters
scheme = Scheme(model, parameters, {"dataset1": dataset})
result = optimize(scheme)
result
result.optimized_parameters
result_dataset = result.data["dataset1"]
result_dataset
residual_left = result_dataset.residual_left_singular_vectors.sel(left_singular_value_index=0)
residual_right = result_dataset.residual_right_singular_vectors.sel(right_singular_value_index=0)
residual_left.plot.line(x="time", aspect=2, size=5)
residual_right.plot.line(x="spectral", aspect=2, size=5);

#%%
