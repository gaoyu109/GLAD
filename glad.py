def select_data(glad_orig, start_time, end_time):
    '''
    Select data segments over specific time period.
    Input: 
    glad_orig : pandas dataframe consisting of the GLAD drifter
    D, time, U and V velocity, and their errors.
    start_time : start time of the period (e.g. 2012-07-22)
    end_time : end time of the period (e.g. 2012-08-04)
    '''
    
    import pandas as pd
    from datetime import datetime
    
    glad_date = glad_orig.set_index('Date')
    start = datetime.strptime(start_time, '%Y-%m-%d')
    end = datetime.strptime(end_time, '%Y-%m-%d')
    date_range = pd.date_range(start = start, end = end).strftime("%Y-%m-%d")
    glad_select = glad_date.loc[date_range.tolist(),:]
    glad_select.groupby('Date').groups.keys()
    glad_selected = glad_select.set_index('ID')
    
    return glad_selected

def autocorrelation(u,v):
    '''
    This function calculates the autocorrelation function
    of a GLAD drifter. 
    Input (u,v) are the two velocity components of a single 
    drifter.
    '''
    import numpy as np
    
    u_mean = u.mean()
    v_mean = v.mean()
    N = len(u)
    C = np.zeros(N)
    X = np.zeros(N)
    Y = np.zeros(N)
    Z = np.zeros(N)
    coruv = np.zeros(N)
    
    for i in range(N):
        for j in range(i,N):
            k = abs(i-j) #time lag
            X[k] = X[k] + (u[i] - u_mean) * (v[j] - v_mean)
            Y[k] = Y[k] + (u[i] - u_mean) * (u[i] - u_mean)
            Z[k] = Z[k] + (v[j] - v_mean) * (v[j] - v_mean)
            
    coruv = X/np.sqrt(Y*Z) 
    
    return coruv

def avg_autocorr(glad_week, nweeks):
    '''
    This function calculates the autocorrelation function 
    of multiple GLAD drifters. 
    Input:
    glad_week: a pandas dataframe consisting of the GLAD drifter
    D, time, U and V velocity, and their errors.
    x: how many weeks of data.
    '''
    import numpy as np
    import pandas as pd
    import numpy.ma as ma
    
    coruu = ma.empty([89,nweeks*672])
    corvv = ma.empty([89,nweeks*672])
    
    drifter_grouped = glad_week.groupby(['ID'])
    drifter_keys = drifter_grouped.groups.keys()
    
    for index,drifter in enumerate(drifter_keys):
        u_2 = glad_week.loc[drifter,:]['U'].values
        v_2 = glad_week.loc[drifter,:]['V'].values
        coruu[index,:len(u_2)] = autocorrelation(u_2,u_2)
        coruu[index,len(u_2):] = 1e-20
        corvv[index,:len(u_2)] = autocorrelation(v_2,v_2)
        corvv[index,len(u_2):] = 1e-20
        
    coruu_masked = ma.masked_equal(coruu,1e-20)
    corvv_masked = ma.masked_equal(corvv,1e-20)

    coruu_avg = np.empty(nweeks*672)
    corvv_avg = np.empty(nweeks*672)

    for i in range(nweeks*672):
        coruu_avg[i] = coruu_masked[:,i].mean()
        corvv_avg[i] = corvv_masked[:,i].mean()
        
    return coruu_avg, corvv_avg 

def avg_autocross(glad_week, nweeks):
    '''
    This function calculates the crosscorrelation function 
    of multiple GLAD drifters. 
    Input is a pandas dataframe consisting of the GLAD drifter
    D, time, U and V velocity, and their errors.
    '''
    import numpy as np
    import pandas as pd
    import numpy.ma as ma
    
    coruv = ma.empty([89,nweeks*672])
    corvu = ma.empty([89,nweeks*672])
    
    drifter_grouped = glad_week.groupby(['ID'])
    drifter_keys = drifter_grouped.groups.keys()
    
    for index,drifter in enumerate(drifter_keys):
        u_2 = glad_week.loc[drifter,:]['U'].values
        v_2 = glad_week.loc[drifter,:]['V'].values
        coruv[index,:len(u_2)] = autocorrelation(u_2,v_2)
        coruv[index,len(u_2):] = 1e-20
        corvu[index,:len(u_2)] = autocorrelation(v_2,u_2)
        corvu[index,len(u_2):] = 1e-20
        
    coruv_masked = ma.masked_equal(coruv,1e-20)
    corvu_masked = ma.masked_equal(corvu,1e-20)

    coruv_avg = np.empty(nweeks*672)
    corvu_avg = np.empty(nweeks*672)
    
    for i in range(nweeks*672):
        coruv_avg[i] = coruv_masked[:,i].mean()
        corvu_avg[i] = corvu_masked[:,i].mean()
    
    return coruv_avg, corvu_avg

def spaghetti_plot(glad_orig):
    '''
    Make spaghetti plot of drifter trajectories
    Input:
    pandas data frame consisting of the GLAD drifter
    D, time, U and V velocity, and their errors.
    '''
    import pandas as pd
    import math
    import numpy as np
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    
    
    drifter_grouped = glad_orig.groupby(['ID'])
    drifter_keys = drifter_grouped.groups.keys()
    glad = glad_orig.set_index('ID')
    plt.figure(figsize=(13,15))
    max_lat, min_lat = 30.5, 23
    max_lon, min_lon= -85, -91.5
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon,max_lon,min_lat,max_lat],ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black')
    for index,drifter in enumerate(drifter_keys):
        lat, lon = glad.loc[drifter,:]['Latitude'].values, glad.loc[drifter,:]['Longitude'].values
        ax.plot(lon,lat)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,color='grey')
    plt.savefig('spaghetti_plot.png')
    
def plot_corr(coruu,corvv,coruv,corvu,nweeks,fig_name):
    '''
    Plot autocorrelation and autocrosscorrelation function
    Input:
    Cuu, Cvv, Cuv, Cvu
    nweeks: how many weeks of data.
    figname: name of the figure.
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12,10))
    plt.subplot(2,1,1)
    days = np.arange(0,len(coruu)) * 0.0104166667
    plt.plot(days,coruu,'r-',days,corvv,'b-')
    plt.legend(['Cuu','Cvv'],loc='best')
    plt.xlabel('Temporal lag(days)')
    plt.ylabel('Cor')
    plt.xlim([0,7*nweeks - 1])
    plt.ylim([-1.0,1.0])

    plt.subplot(2,1,2)
    days = np.arange(0,len(coruv)) * 0.0104166667
    plt.plot(days,coruv,'r-',days,corvu,'b-')
    plt.legend(['Cuv','Cvu'],loc='best') 
    plt.xlabel('Temporal lag(days)')
    plt.ylabel('Cross')
    plt.xlim([0,7*nweeks - 1])
    plt.ylim([-1.0,1.0])
    plt.savefig(fig_name)

def lonlat_to_xy(lon,lat):
    '''
    Converts arrays of longitude and latitude locations
    to Cartesian x and y arrays in kilometers relative 
    to the deployment position of a drifter.
    lon: Longitude numpy array
    lat: Latitude numpy array
    '''
    
    import numpy as np
    
    R_Earth = 6371 #km
    DEG2RAD = np.pi/180.0
    DEG2KM  = DEG2RAD*R_Earth
    
    # Deployment position
    lon0, lat0 = lon[0], lat[0]
    
    x = (lon - lon0) * DEG2KM * np.cos(lat*DEG2RAD)
    y = (lat - lat0) * DEG2KM
    return x, y

def xy_variance(glad_week,nweeks):
    '''
    Calculate absolute dispersion, which is the variances at 
    time t of the particle position displacement data relative 
    to the deployment position, (x[i,0],y[i,0]) for each drifter i, 
    calculated by averaging over n ddrifters in a cluster.
    Input:
    glad_week : dataframe of drifter data
    nweeks: how many weeks
    fig_name: the name of figure
    '''
    
    import numpy as np
    import pandas as pd
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    
    #convert longitude and latitude to (x,y) relative to deployment position.
    drifter_grouped = glad_week.groupby(['ID'])
    drifter_keys = drifter_grouped.groups.keys()
    
    dx = np.zeros([len(drifter_keys),nweeks*672])
    dy = np.zeros([len(drifter_keys),nweeks*672])
    
    for index,drifter in enumerate(drifter_keys):
        lon = glad_week.loc[drifter,:]['Longitude'].values
        lat = glad_week.loc[drifter,:]['Latitude'].values
        N = len(lon)
        dx[index,:N], dy[index,:N] = lonlat_to_xy(lon, lat)
        dx[index,N:] = 0
        dy[index,N:] = 0

    sigma_x, sigma_y = np.zeros(nweeks*672), np.zeros(nweeks*672)
    
    for t in range(1,nweeks*672):
        # variance of position in x- and y-direction
        ndrifters = np.count_nonzero(dx[:,t])
        if(ndrifters > 1):
            sigma_x[t] = np.sum(dx[:,t]**2/(ndrifters-1))
            sigma_y[t] = np.sum(dy[:,t]**2/(ndrifters-1))
    
    return sigma_x, sigma_y

def uv_variance(glad_week,nweeks):
    '''
    Calculate absolute dispersion, which is the variances at 
    time t of the particle position displacement data relative 
    to the deployment position, (x[i,0],y[i,0]) for each drifter i, 
    calculated by averaging over n ddrifters in a cluster.
    Input:
    glad_week : dataframe of drifter data
    nweeks: how many weeks
    fig_name: the name of figure
    '''
    
    import numpy as np
    import pandas as pd
    import numpy.ma as ma
    import matplotlib.pyplot as plt
    
    #convert longitude and latitude to (x,y) relative to deployment position.
    drifter_grouped = glad_week.groupby(['ID'])
    drifter_keys = drifter_grouped.groups.keys()
    
    du = np.zeros([len(drifter_keys),nweeks*672])
    dv = np.zeros([len(drifter_keys),nweeks*672])
    
    for index,drifter in enumerate(drifter_keys):
        u = glad_week.loc[drifter,:]['U'].values
        v = glad_week.loc[drifter,:]['V'].values
        N = len(u)
        du[index,:N], dv[index,:N] = u, v
        du[index,N:] = 0
        dv[index,N:] = 0

    sigma_u, sigma_v = np.zeros(nweeks*672), np.zeros(nweeks*672)
    
    for t in range(1,nweeks*672):
        # variance of position in x- and y-direction
        ndrifters = np.count_nonzero(du[:,t])
        if(ndrifters > 1):
            sigma_u[t] = np.var((du[:,t]-du[:,0])**2/(ndrifters-1))
            sigma_v[t] = np.sum((dv[:,t]-dv[:,0])**2/(ndrifters-1))
            #sigma_u[t] = np.sum((du[:,t]-du[:,0])**2/(ndrifters-1))
            #sigma_v[t] = np.sum((dv[:,t]-dv[:,0])**2/(ndrifters-1))
            
    return sigma_u, sigma_v


def abs_disper(glad_week,nweeks):
    
    import numpy as np
    
    sigma_x,sigma_y = xy_variance(glad_week,nweeks)
    
    abs_dispersion = np.sqrt(sigma_x + sigma_y)
    Kx = np.diff(sigma_x) * 0.5 / (15 * 60)
    Ky = np.diff(sigma_y) * 0.5 / (15 * 60)
    diffusivity = np.sqrt(Kx**2 + Ky**2)
    return abs_dispersion, diffusivity

def integral_time_scale(Cuu,Cvv):
    
    import numpy as np
    
    Iu = np.nansum(Cuu*15*60)
    Iv = np.nansum(Cvv*15*60)
    
    return Iu, Iv

def eddy_diffusivity(glad_week,nweeks,Iu,Iv,sigma_u,sigma_v):
    
    import numpy as np
    import pandas as pd
    
    Ku = sigma_u * Iu
    Kv = sigma_v * Iv
    
    K = np.sqrt(Ku**2+Kv**2)
    
    return K
    
def plot_diffusivity(abs_dispersion3, diffusivity3, K_13, nweeks, fig_name):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    days = np.arange(0, len(abs_dispersion3[:3840])) *0.0104166667
    plt.figure(figsize=(12,15))
    plt.subplot(3,1,1)
    plt.plot(days, abs_dispersion3[:3840],'r-')
    plt.xlim([0, nweeks*7-1])
    plt.ylim([0, abs_dispersion3[:3840].max()+10])
    plt.xlabel('Days from launch')
    plt.ylabel('square root of abs. dispersion(km)')

    days_1 = np.arange(0,len(diffusivity3[:3840])) *0.0104166667
    plt.subplot(3,1,2)
    plt.plot(days_1, diffusivity3[:3840]/1000,'b-')
    plt.xlim([0,nweeks*7-1])
    plt.xlabel('Days from launch')
    plt.ylabel('diffusivity(*1000km/s)')
    
    Kdays = np.arange(0,len(K_13[:3840])) *0.0104166667
    plt.subplot(3,1,3)
    plt.plot(Kdays, K_13[:3840]/1000)
    plt.xlim([0, nweeks*7-1])
    plt.xlabel('Days from launch')
    plt.ylabel('diffusivity(*1000km/s)')

    plt.savefig(fig_name)
    

