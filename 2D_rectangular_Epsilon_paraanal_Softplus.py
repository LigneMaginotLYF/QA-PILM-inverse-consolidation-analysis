import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import diags
from matplotlib.widgets import Slider
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from datetime import datetime

# 0. Import external configurations. Except that this time we are considering a radial problem.
filedir = 'F:/博/PINN 2D consoli practice/250827 - Hindered by illness/Relocated to Python/lib/'
filelist = ['rawSlistA-0.3sig-2by2.csv', 'Lmat-2by2.csv'] # Several files for generating and controlling randomness.
rawS, Lmat = [], []
varlist = [rawS, Lmat]
for i in range(len(filelist)):
    with open(filedir + filelist[i], 'rt') as raw_data:
        varlist[i] = np.loadtxt(raw_data, delimiter=',')
rawS, Lmat = varlist[0], varlist[1]

# 1. Basic configurations

# 1.1 Geometry; only the first four lines are adjustable.
Lh=np.float64(10) # Horizontal span of the domain, in meters
Lv=np.float64(5)
Lt=2 # Maximum time span computed, in years
spatial_reso=np.array([0.2,0.2]) # resolution in spatial coordinates, meters
numh=int(Lh/spatial_reso[0])
numv=int(Lv/spatial_reso[1])
cpth=np.arange(0,Lh+0.001,spatial_reso[0]) # list of collocation points in x-direction.
cptv=np.arange(0,Lv+0.001,spatial_reso[1]) # list of collocation points in z-direction.
cpts2=list(itertools.product(cpth,cptv)) # list of points in 2D plane. Ordering= (x,z); first group = first column in matrix
print(len(cpts2))


def vec2mat3(vector): # Function to convert flattened list to 3D tensor, in 2D x-z plane.
    return np.array(vector).reshape(numh+1,numv+1,2).transpose(1,0,2) # matrix form. Validated so far.


def vec2mat2(vector): # Function to convert flattened list to matrix, in 2D x-z plane.
    return np.array(vector).reshape(numh+1,numv+1).transpose(1,0) # matrix form. Validated so far.


# 1.2 Material parameters, ground truth;
Ch=1 # Obsolete and irrelevant
Rcv=1 # ratio of vertical perm to horizontal perm, s.t. Cv=Rcv.Ch
sigma=0.3 # Controlling deviation of the random field, not necessarily used.
coeft=np.array([-1,-0.5,1,0.5,2,-0.2]) # Weights for the basis as ground truth.
u0=1 # Initial pwp of the overall field. Assumed to be scalar.
alpha=0.05 # Controls time step length.


def baset(vector): # Basis function to generate the ground truth of the trend.
    a=np.array(vector)
    b=np.array([Lh,Lv])
    x,z=a/b
    return np.array([1,x,z,(x)**2,(z)**2,(z)*x]) # For a single point, return values of all bases.


def basetdx(vector): # Basis function's partial derivative field wrt x axis.
    a=np.array(vector)
    b=np.array([Lh,Lv])
    x,z=a/b # Normalized to [0,1]
    return np.array([0,1/Lh,0,2*(x)/Lh,0,z/Lh]) # Partial phi / partial x.


def basetdz(vector): # Basis function's partial derivative field wrt z axis.
    a=np.array(vector)
    b=np.array([Lh,Lv])
    x,z=a/b # Normalized to [0,1]
    return np.array([0,0,1/Lv,0,2*(z)/Lv,x/Lv]) # Partial phi / partial z.


def map_vectors(vectors,func): # Function to process multiple inputs, substituting Map function in Mathematica.
    return np.array([func(v) for v in vectors])


def create_tridiag_mat(n,main,upper,lower):
    """
    Inputs dimension n, value on three diagonals, outputs the SPARSE tri-diagonal matrix (differential operator).
    """
    diagonals=[main*np.ones(n),upper*np.ones(n-1),lower*np.ones(n-1)]
    offsets=[0,1,-1]
    matrix=diags(diagonals,offsets,shape=(n,n),format='csr') # to multiply sparse matrix A, use A.dot(B) or B.T.dot(A.T).T
    return matrix


basest=map_vectors(cpts2,baset) # values of bases in flattened 2D plane. This is a matrix.
trend=basest@coeft # Linear combination. In matrix form, but require transposing. Validated.


def advanced_resize_matrix(matrix, target, interpolation='cubic',
                           preserve_aspect_ratio=False, mode='reflect'):
    """
    Resize the fluctuation matrix, such that the changed resolution is satisfied.
    matrix: input matrix
    target: target size
    interpolation: choose from these four methods ('nearest', 'linear', 'cubic', 'lanczos')
    preserve_aspect_ratio: true/false
    mode: Boundary treatment ('constant', 'reflect', 'wrap', 'nearest')

    returns the resized mat. Note that original input/outputs are actually lists.
    """
    from scipy import ndimage

    input_shape = matrix.shape

    interpolation_map = { # Map interpolation methods to orders
        'nearest': 0,
        'linear': 1,
        'cubic': 3,  # Generally we prefer this.
        'lanczos': 5  # In scipy, order=5 is used to approximate lanczos
    }

    if interpolation not in interpolation_map:
        raise ValueError(f"Unsupported interpolation method: {interpolation}")

    order = interpolation_map[interpolation]  # e.g., for cubic method, the order is 3.

    # Define the shape of the output
    if np.isscalar(target):
        # which means aspect ratio is locked. However it might not always be the case, as boundary has one additional row/col
        scale_factor = target
        target_shape = (int(input_shape[0] * scale_factor),
                        int(input_shape[1] * scale_factor))
    elif len(target) == 2:
        # A.R. not locked, interpolate separately. If locked and inputs a doublet, adjust.
        if preserve_aspect_ratio:
            aspect_ratio = input_shape[1] / input_shape[0]  # Original aspect ratio
            if target[0] is not None and target[1] is not None:  # Contributing to a contradiction
                scale_x = target[1] / input_shape[1]  # Ratio by which x is stretched
                scale_y = target[0] / input_shape[0]
                scale_factor = min(scale_x, scale_y)  # Must truncate, according to the smaller scale. Such that A.R. is locked.
                target_shape = (int(input_shape[0] * scale_factor),
                                int(input_shape[1] * scale_factor))  # Must be integer.
            elif target[0] is not None:  # No contradictions, no truncation needed.
                scale_factor = target[0] / input_shape[0]
                target_shape = (target[0], int(input_shape[1] * scale_factor))
            else:  # Use y as scaling factor.
                scale_factor = target[1] / input_shape[1]
                target_shape = (int(input_shape[0] * scale_factor), target[1])
        else:  # Most common usage. No fixed aspect ratio.
            target_shape = target
    else:
        raise ValueError("target must be scalar (locked A.R.) or doublet (unlocked A.R.)")

    target_shape = (max(1, target_shape[0]), max(1, target_shape[1]))

    # Compute zooming factor for interpolation.
    zoom_factors = (target_shape[0] / input_shape[0],
                    target_shape[1] / input_shape[1])

    # Interpolate. Check if correct.
    resized = ndimage.zoom(matrix, zoom_factors, order=order, mode=mode)

    return np.array(resized)


rawSlist = np.array(rawS)  # load the random vector from previous generations. Can also regenerate using sigma.
fluc_o = sigma * np.array(Lmat).T @ rawSlist  # Fluctuation field.
flucmat=fluc_o.reshape(50+1,25+1).transpose(1,0)  # The resolution of the original fluc is FIXED to 2*2.
const=-1
res_fluc=advanced_resize_matrix(flucmat,[numv+1,numh+1])  # Resized fluc matrix
fluc=np.reshape(res_fluc.transpose(1,0),-1)  # Resized fluc list
Olist=np.exp(fluc)+trend-const # Together, this forms the ground truth for the permeability field. Validated.
dx=spatial_reso[0]
dz=spatial_reso[1]
chm=vec2mat2(Olist) # Matrix form of horizontal permeability
chv=chm*Rcv # That of vertical permeability
basestdx=map_vectors(cpts2,basetdx) # Corresponding field of basis' 1st partial derivative (over x).
basestdz=map_vectors(cpts2,basetdz)
trenddx=basestdx@coeft
trenddz=basestdz@coeft # These are used to compute D1(D2) = Partial Ch(Cv) / partial x(z) for the ground truth.
o1=create_tridiag_mat(numh+1,0,1,-1)/(dx*2) # 1st order differential operator.
o2=create_tridiag_mat(numv+1,0,1,-1)/(dz*2)
d1=(vec2mat2(trenddx))+res_fluc*(o1.T.dot(res_fluc.T)).T # Matrix form of Partial hor. perm. over x
d2=(vec2mat2(trenddz))+res_fluc*o2.dot(res_fluc) # !!!Must delete multiplication of chm/chv in d1 and d2.
ah=alpha*(dx**2)/np.max(chm) # alpha is hyperparameter to control the convergence of FD scheme, <=0.125.
az=alpha*(dz**2)/np.max(chv)
bh=alpha*(dx*2)/np.max(d1+0.001)
bz=alpha*(dz*2)/np.max(d2+0.001)
dt=np.min([ah,az,bh,bz]) # Desired time interval for this problem.
print('Initialization complete! The time interval for this given perm field is set as ',dt,' year')
print('Cmax= ',np.max(chm),'; Cmin= ',np.min(chm))
# Following are some plotting functions. Not necessarily used.


def custom_colormap(): # Define a blue-red contour map
    from matplotlib.colors import LinearSegmentedColormap
    colors_list=['red','blue']
    custom_cmap=LinearSegmentedColormap.from_list('blue_red',colors_list)
    return custom_cmap


def singleplot2D(matrix,color,title): # Function to plot single field in 2D contour plane
    plt.figure(figsize=(10, 5))
    sns.heatmap(matrix,
                cmap=color,
                annot=False,  # Show values
                cbar=True,  # Show contour legend
                square=False)  # Maintain square shape
    plt.title(title)
    plt.tight_layout()
    plt.show()


def singleplot3D(matrix,color,title): # Function to plot single field in 3D surface
    fig=plt.figure(figsize=(10, 5))
    ax=fig.add_subplot(111,projection='3d')
    xs=cptv
    ys=cpth
    X,Y=np.meshgrid(ys,xs)
    surf=ax.plot_surface(X,Y,matrix,
                         cmap=color,
                         alpha=0.8,
                         edgecolor='gray',
                         linewidth=0.5
                         )
    plt.colorbar(surf,ax=ax,shrink=0.5,aspect=20,label=None)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel(title)
    #ax.set_zscale('log')
    #plt.title(title+' contour plot')
    plt.tight_layout()
    plt.show()


def triplot2D(data1,data2,titles,mode,cmap1=custom_colormap,cmap2='bwr',vmin12=None,vmax12=None,vmin3=None,vmax3=None,psavepath=None,figsize=(15,5),dpi=100): # Function to plot comparison of 2 fields in 2D contour plane
    if mode==1:
        data3=np.log(data1/data2) # Error=Real-Pred, by default(0). If 1, then log error is chosen.
    else:
        data3=data1-data2
    default_scientific = {
        'parameters': {
            'data1': {'name': 'Temperature', 'symbol': 'T', 'unit': 'K'},
            'data2': {'name': 'Pressure', 'symbol': 'P', 'unit': 'Pa'},
            'data3': {'name': 'Density', 'symbol': 'ρ', 'unit': 'kg/m³'}
        },
        'axes': {
            'x': {'name': 'Horizontal Distance', 'symbol': 'x', 'unit': 'm'},
            'y': {'name': 'Vertical Height', 'symbol': 'z', 'unit': 'm'}
        },
        'fontsize': {
            'title': 16,
            'label': 14,
            'tick': 12
        }
    }
    config=default_scientific
    xticks=[0,int(numh/4),int(numh/2),int(numh*3/4),numh]
    xtlabel=np.array([0,1/4,1/2,3/4,1])*Lh
    zticks=[0,int(numv/4),int(numv/2),int(numv*3/4),numv]
    ztlabel=[0,Lv/4,Lv/2,Lv*3/4,Lv]
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'
    })
    plt.rcParams['figure.dpi'] = dpi
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if vmin12 is None:
        vmin12 = min(np.min(data1), np.min(data2))
    if vmax12 is None:
        vmax12 = max(np.max(data1), np.max(data2))
    norm12 = colors.Normalize(vmin=vmin12, vmax=vmax12)
    im1 = axes[0].imshow(data1,
                         cmap=cmap1,
                         norm=norm12,
                         origin='upper',
                         aspect='auto',
                         interpolation='bicubic')  # Higher-reso
    axes[0].set_title(titles[0], fontsize=14, pad=10)
    axes[0].set_xlabel('r(m)', fontsize=12)
    axes[0].set_ylabel('z(m)', fontsize=12)
    # axes[0].tick_params(labelsize=config['fontsize']['tick'])
    im2 = axes[1].imshow(data2,
                         cmap=cmap1,
                         norm=norm12,
                         origin='upper',
                         aspect='auto',
                         interpolation='bicubic')  # Higher-reso
    axes[1].set_title(titles[1], fontsize=14, pad=10)
    axes[1].set_xlabel('r(m)', fontsize=12)
    axes[1].set_ylabel('z(m)', fontsize=12)
    if vmin3 is None:
        vmin3 = -np.abs(np.max(data3))
    if vmax3 is None:
        vmax3 = np.abs(np.max(data3))
    norm3 = colors.Normalize(vmin=vmin3, vmax=vmax3)
    im3 = axes[2].imshow(data3,
                         cmap=cmap2,
                         norm=norm3,
                         origin='upper',
                         aspect='auto',
                         interpolation='bicubic')  # Higher-reso
    axes[2].set_title(titles[2], fontsize=14, pad=10)
    axes[2].set_xlabel('X(m)', fontsize=12)
    axes[2].set_ylabel('Z(m)', fontsize=12)
    # Modify tick values
    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtlabel)
        ax.set_yticks(zticks)
        ax.set_yticklabels(ztlabel)
    # Define the colorbars
    # First colorbar, shared by ims 1,2
    divider1 = make_axes_locatable(axes[1])
    cax12 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar12 = plt.colorbar(im2, cax=cax12)
    cbar12.set_label('', rotation=270, labelpad=15, fontsize=12)
    # Second colorbar, used by im 3
    divider3 = make_axes_locatable(axes[2])
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(im3, cax=cax3)
    cbar3.set_label('', rotation=270, labelpad=15, fontsize=12)

    plt.tight_layout()
    if psavepath is None:
        plt.show()
    else:
        plt.savefig(psavepath,dpi=300,bbox_inches='tight')
        plt.close()

    return fig, axes, (cbar12, cbar3)


#triplot2D(np.log(vec2mat2(Olist)),vec2mat2(Olist),np.array(['Real C_h(m^2/year)','Predicted C_h(m^2/year)','Error(m^2/year)']))


def dualplot3D(mat1,mat2,col1,col2,title1,title2): # Function to plot two field's comparison in 3D surface
    fig=plt.figure(figsize=(10, 5))
    xs=cptv
    ys=cpth
    X,Y=np.meshgrid(ys,xs)
    ax=fig.add_subplot(111,projection='3d')
    surf1=ax.plot_surface(X,Y,mat1,
                         cmap=colors.ListedColormap([col1]),
                         alpha=1,
                         edgecolor='none',
                         label=title1
                         )
    surf2 = ax.plot_surface(X, Y, mat2,
                            cmap=colors.ListedColormap([col2]),
                            alpha=0.5,
                            edgecolor='none',
                            label=title2
                            )
    # fig.colorbar(surf1,ax=ax,shrink=0.5,aspect=20,label=title1)
    # fig.colorbar(surf2,ax=ax,shrink=0.5,aspect=20,label=title2)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('Permeability field')
    #ax.set_zscale('log')
    ax.set_title('Permeability contour plot: Solution vs Truth')
    ax.legend()
    plt.tight_layout()
    plt.show()


def dualplotmesh(mat1,mat2,title): # Function to plot single field in 3D surface
    fig=plt.figure(figsize=(14, 6))
    xs=cptv
    ys=cpth
    X,Y=np.meshgrid(ys,xs)
    ax=fig.add_subplot(122,projection='3d')
    wire1=ax.plot_wireframe(X,Y,mat1,color='blue',linewidth=1.2,alpha=0.7,label='real')
    wire2=ax.plot_wireframe(X,Y,mat2,color='red',linewidth=1.2,alpha=0.7,label='pred')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('Log Permeability field')
    #ax.set_zscale('log')
    ax.legend()
    plt.tight_layout()
    plt.show()


#singleplot3D(vec2mat2(Olist),'plasma','Coefficient of consolidation (m^2/year)') # Validated.
# dualplot3D(np.log(vec2mat2(Olist)),vec2mat2(Olist),'plasma','viridis','real','pred') # Validated. Subject to customize.
# dualplotmesh(np.log(vec2mat2(Olist)),vec2mat2(Olist),'Permeability field')


# 2. Forward FD solver definition

# 2.1 Compute & assemble differential operator matrices.
# The PDE is discretized as :U^(n+1)=U^(n)+Ch*(U^n@A1)+Cv*(A2@U^n)+[D1*(U^N@B1)+D2*(B2@U^n)]  <- Caution on the [] term
# A1 is operator for 2-order partial derivative on x, A2 - z; B1 is 1-order PD on x, B2 - z.


a1=create_tridiag_mat(numh+1,-2,1,1)*dt/(dx**2) # Laplacian matrix for horizontal direction. a1=raw
a2=create_tridiag_mat(numv+1,-2,1,1)*dt/(dz**2)
b1=create_tridiag_mat(numh+1,0,1,-1)*dt/(dx*2) # note that when computing D1, D2, the dt term should be eliminated.
b2=create_tridiag_mat(numv+1,0,1,-1)*dt/(dz*2) # Caution on the signs of B1, B2
ilim=int(Lt//dt) # Maximum time steps evaluated, after which further computation is truncated.


def forward_solver(chm,Rcv,numt,u0,top,bot,left,right):
    """
    Takes number of time step and IC/BCs as inputs, outputs the 3D tensor of epwp field & list of deg_consolidation.
    top~right are BC descriptions. 0=Dirichlet, 1=Neumann.
    Remark on the FD itself: The fluctuation term now ALWAYS causes some points to have negative epwp, near final state.
    """
    chv=chm*Rcv
    utens=np.zeros((numt+1,int(numv+1),int(numh+1)),dtype=np.float64) # Initialize U-tensor
    utens[0,:,:]=u0 # Initial condition
    udeg=np.ones(numt+1,dtype=np.float64) # Degree of consolidation, average
    for i in range(numt): # Stepwise evaluation
        cu=utens[i,:,:]
        utens[i+1,:,:]=cu+chm*((a1.T.dot(cu.T)).T)+chv*(a2.dot(cu))+d1*((b1.T.dot(cu.T)).T)+d2*(b2.dot(cu)) # Governing equation
        utens[i+1,:,0]=left*utens[i+1,:,1] # BC on left. So arranged so that Top/Bot always overwrites Left/Right.
        utens[i+1,:,-1]=right*utens[i+1,:,-2] # BC on right.
        utens[i+1,0,:]=top*utens[i+1,1,:] # BC on top, can be directly imposed according to 0=Dirichlet/1=Neumann.
        utens[i+1,-1,:]=bot*utens[i+1,-2,:] # BC on bottom.
        udeg[i+1]=np.mean(utens[i+1])/u0 # Compute average deg of saturation
    # End of Loop i
    return utens,udeg


def u_field_plot(u, udeg, title="Stepwise EPWP plot", cmap="jet", aspect_ratio=Lh/Lv, cloud_ratio=0.7):
    """
    Plot snapshots of the u field, according to a slider bar. Generated by Deepseek.
    """
    n_time, n_rows, n_cols = u.shape

    # Modify plot geometries
    fig = plt.figure(figsize=(15,6))

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix'
    })

    # Compute subplot positions
    cloud_width = cloud_ratio * 0.9  # For A
    time_width = 0.9 - cloud_width - 0.05  # For B
    colorbar_width = 0.01  # Colorbar

    # Subplot A on left is larger
    ax1 = plt.axes([0.01, 0.3, cloud_width, 0.6])  # [left, bottom, width, height]
    # Subplot B, smaller
    ax2 = plt.axes([0.05+cloud_width+0.05, 0.3, time_width, 0.6])
    cbar_ax = plt.axes([0.05+cloud_width-0.09, 0.3, colorbar_width, 0.6])
    # Slider
    ax_slider = plt.axes([0.25, 0.15, 0.5, 0.03])

    # Confine display interval to [0, 1]
    norm = colors.Normalize(vmin=0, vmax=1)

    # Subplot A: Contour
    current_time = 0
    im = ax1.imshow(u[current_time, :, :],
                    cmap=cmap, # Can customize if needed
                    norm=norm,
                    origin='upper',
                    aspect=aspect_ratio, # Adjusted according to the real aspect ratio
                    interpolation='bilinear',
                    extent=[0,Lh,Lv,0])

    ax1.set_title(f'{title}\nTime step: {current_time}/{n_time - 1}', fontsize=16, pad=20)
    ax1.set_xlabel('r(m)')
    ax1.set_ylabel('z(m)')

    # Color bar
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('EPWP', rotation=270, labelpad=20, fontsize=12)

    # Subplot B: Average EPWP dissipation
    time_values = np.arange(n_time)
    center_value = udeg

    line, = ax2.plot(time_values, center_value, 'b-', linewidth=2)
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Average EPWP level')
    ax2.set_title('EPWP dissipation')
    ax2.grid(True, alpha=0.3)

    # Current time marker
    current_point, = ax2.plot([current_time], [center_value[current_time]],
                              'ro', markersize=8)

    # Slider bar
    time_slider = Slider(ax_slider, 'Time step', 0, n_time - 1,
                         valinit=current_time, valfmt='%d')

    def update(val):
        time_step = int(val)

        # Update contour
        im.set_data(u[time_step, :, :])
        ax1.set_title(f'{title}\nTime step: {time_step}/{n_time - 1}', fontsize=16)

        # Update time series marker
        current_point.set_data([time_step], [center_value[time_step]])

        fig.canvas.draw_idle()

    time_slider.on_changed(update)

    # Initialize stats box
    update.stats_box = None
    update(current_time)

    plt.show()
    return fig

# 2.2 Define generator for analytical solution
# Obsolete.

# 3. Inverse PIML solver definition

# 3.1 Define estimator basis functions

orderx=1 # order of Fourier expansion of Sin(x), to the intensity of Sin(x*2**(order-1))
orderz=3 # that of z, always assumed to be higher than x.


def basee(vector): # Basis function used as estimators/neurons.
    a=np.array(vector)
    b=np.array([Lh,Lv])
    x,z=a/b # Normalized to [0,1]
    lb=np.array([1,x,z,x**2,z**2,x*z])  # Try adding 1/(1+r)
    lx=np.sin(2**np.arange(orderx)*x)
    lz=np.sin(2**np.arange(orderz)*z)
    blist=np.concatenate([lb,lx,lz],axis=None)
    return blist # Redundant, subject to change.


def baseedx(vector): # Basis function's partial derivative field wrt x-axis.
    a=np.array(vector)
    b=np.array([Lh,Lv])
    x,z=a/b # Normalized to [0,1]
    lb=np.array([0,1,0,x*2,0,z])/(Lh)
    lx=np.cos(2**np.arange(orderx)*x)*(2 ** np.arange(orderx))/(Lh)
    lz=np.cos(2**np.arange(orderz)*z)*0
    blist=np.concatenate([lb,lx,lz],axis=None)
    return blist # Redundant, subject to change.


def baseedz(vector): # Basis function's partial derivative field wrt z axis.
    a=np.array(vector)
    b=np.array([Lh,Lv])
    x,z=a/b # Normalized to [0,1]
    lb = np.array([0, 0, 1, 0, z*2, x])/Lv
    lx = np.cos(2 ** np.arange(orderx) * x) * 0
    lz = np.cos(2 ** np.arange(orderz) * z) * (2 ** np.arange(orderz))/Lv
    blist = np.concatenate([lb, lx, lz], axis=None)
    return blist  # Redundant, subject to change.


def softplus(x):   # Softplus function.
    y=np.log(1+np.exp(x+0))
    return y


def dsoftplus(x):   # Derivative of softplus, =sigmoid
    y=1/(1+np.exp(-x-0))
    return y


def isoftplus(y):   # Inverse of softplus
    x=np.log1p(np.expm1(y))-0   # Prevent catastrophic cancellation.
    return x


def scaledtanh(x,a=10**(-4),b=10**5):   # Would serve as the potential activation function.
    y=a+(b-a)/(1+np.exp(-x))
    return y


def dscaledtanh(x,a=10**(-4),b=10**5):   # Derivative.
    y=(b-a)*(1-np.tanh(x)**2)/2
    return y


def iscaledtanh(y,a=10**(-4),b=10**5):   # Inverse.
    x=np.atanh((2*y-a-b)/(b-a))
    return x


def epsilon_insensitive_loss_numpy(pred, real, epsilon=0.1):
    """
    ϵ-insensitive squared error loss

    """
    # Transform to np array
    pred = np.asarray(pred)
    real = np.asarray(real)

    # compute error
    err = real - pred

    # small number threshold
    real_abs = np.abs(real)
    real_safe = np.where(real_abs < 0.1, 0.1, real_abs)

    # thres t = (epsilon * real)^2
    t = (epsilon * real_safe) ** 2

    # squared error
    err_sq = err ** 2

    # loss = max(0, err^2 - t)
    loss = np.maximum(0, err_sq - t)

    # compute gradient
    # if err^2 > t, grad = -2*err
    # else = 0
    mask = (err_sq > t).astype(float)
    grad = -2.0 * err * mask

    return loss, grad


basese=map_vectors(cpts2,basee) # Similar to the forward solving procedure.
basesedx=map_vectors(cpts2,baseedx) # Corresponding field of basis' 1st partial derivative (over x).
basesedz=map_vectors(cpts2,baseedz)

# 3.2 Define measurement locations.

#ukx=np.array([10,25,40]) # Distribution of epwp measurements on x-axis. Assume regular gridding first, subject to change.
#ukz=np.array([5,10,15,20,25,30,35,40,45,50]) # Distribution of epwp measurements on z-axis. This can be random.
#ukx=np.array([1,3,12])
#ukz=np.array([2,5,16])
#ukmat=[[5,3],[10,3],[21,3],[3,25],[6,25],[13,25],[2,18],[14,18],[22,18]]

#ukt=(np.array([1,2,4])*rat).astype(int)


# 3.3 Define the inverse solver, packed in a single function


def loss_evaluator(numt,u0,Rcv,coefe,top,bot,left,right,ukmat,chkmat,uk,chk,lam,lamu):
    utens = np.zeros((int(numt[-1]+1), int(numv + 1), int(numh + 1)), dtype=np.float64)  # Initialize U-tensor
    stens = np.zeros((int(numt[-1]+1), int(numv + 1), int(numh + 1), len(basese[0])),
                     dtype=np.float64)  # Initialize S-tensor, sensitivity.
    utens[0, :, :] = u0  # Initial condition
    stens[0, :, :, :] = 0  # Initial condition
    chem = vec2mat2(softplus(basese @ coefe))  # Matrix form of horizontal permeability, estimation
    chev = chem * Rcv  # That of vertical permeability
    dcm=vec2mat2(dsoftplus(basese @ coefe))
    dcv=dcm*Rcv
    trendedx = vec2mat2(basesedx @ coefe)
    trendedz = vec2mat2(basesedz @ coefe)  # These are used to compute D1(D2) = Partial Ch(Cv) / partial x(z), estimation.
    d1e = dcm * trendedx  # Matrix form of Partial hor. perm. over x
    d2e = dcv * trendedz
    for i in range(numt[-1]):  # Stepwise forward evaluation
        # (1) evaluation of state field u
        cu = utens[i, :, :]
        utens[i + 1, :, :] = cu + chem * ((a1.T.dot(cu.T)).T) + chev * (a2.dot(cu)) + d1e * (
            (b1.T.dot(cu.T)).T) + d2e * (b2.dot(cu)) # Governing equation
        utens[i + 1, :, 0] = left * utens[i + 1, :,
                                    1]  # BC on left. So arranged so that Top/Bot always overwrites Left/Right.
        utens[i + 1, :, -1] = right * utens[i + 1, :, -2]  # BC on right.
        utens[i + 1, 0, :] = top * utens[i + 1, 1,
                                   :]  # BC on top, can be directly imposed according to 0=Dirichlet/1=Neumann.
        utens[i + 1, -1, :] = bot * utens[i + 1, -2, :]  # BC on bottom.
        # (2) evaluation of sensitivity fields s; not needed

    # End of Loop i
    # Loss and gradient evaluation
    loss_t = 0
    for i in ukmat:  # epwp loss
        uiloss, _ = epsilon_insensitive_loss_numpy(utens[numt, i[0], i[1]], uk[:, i[0], i[1]])
        loss_t += np.mean(uiloss) * lamu  # !Changed to ReLU
    for i in chkmat:  # permeability loss
        ciloss, _ = epsilon_insensitive_loss_numpy(chem[i[0], i[1]], chk[i[0], i[1]])
        loss_t += lam * np.mean(ciloss) * (1 + Rcv)

    return loss_t,utens,chem


def inverse_solver(numt,u0,Rcv,top,bot,left,right,ukmat,chkmat,uk,chk,lam,lr,ltol,gtol,itol,wlst,lamu):
    """
    Takes number of time step, measurements, and IC/BCs as inputs, outputs the weight vector.
    top~right are BC descriptions. 0=Dirichlet, 1=Neumann.
    lr=learning rate; lam=weight for loss terms; ltol=tolerance for loss; gtol->gradient; itol->num. of iterations.
    """
    coefer=(np.random.randn(len(basese[0]))) # Random initialization. randn gives [0,1), so modification needed.
    coefe=coefer*0.2+0 # Amend by * sigma and then + mean
    print('random init: ',coefe)
    initloss,_,_ = loss_evaluator(ukt, u0, Rcv, coefe, bcs[0], bcs[1], bcs[2], bcs[3], ukmat, chkmat, uk, chm, lam,lamu)
    print('init loss: ',initloss)
    losses=[]
    utens=np.zeros((int(numt[-1]+1),int(numv+1),int(numh+1)),dtype=np.float64) # Initialize U-tensor
    stens=np.zeros((int(numt[-1]+1),int(numv+1),int(numh+1),len(basese[0])),dtype=np.float64) # Initialize S-tensor, sensitivity.
    utens[0,:,:]=u0 # Initial condition
    stens[0,:,:,:]=0 # Initial condition
    beta1=0.9
    beta2=0.999 # Hyperparameters for Adam optimization
    mvec=np.zeros(len(basese[0]))
    vvec=mvec
    for j in range(itol): # Outer loop controlling the training process.
        m=vec2mat2(basese@coefe)
        chem = softplus(m)  # Matrix form of horizontal permeability, estimation
        chev = chem * Rcv  # That of vertical permeability
        dcm = dsoftplus(m)  # 1st derivative of activation function on M
        dcv = dcm * Rcv
        ddcm=dcm*(1-dcm)   # 2nd derivative of activation function on M
        ddcv=ddcm*Rcv
        trendedx = vec2mat2(basesedx@coefe)
        trendedz = vec2mat2(basesedz@coefe)  # These are used to compute D1(D2) = Partial Ch(Cv) / partial x(z), estimation.
        d1e = dcm * trendedx # Matrix form of Partial hor. perm. over x
        d2e = dcv * trendedz
        chdks=np.zeros((len(basese[0]),int(numv+1),int(numh+1)),dtype=np.float64)  # Store all matrices related to weights, to save cost.
        cvdks=np.zeros((len(basese[0]),int(numv+1),int(numh+1)),dtype=np.float64)
        d1dks=np.zeros((len(basese[0]),int(numv+1),int(numh+1)),dtype=np.float64)
        d2dks=np.zeros((len(basese[0]),int(numv+1),int(numh+1)),dtype=np.float64)
        for k in range(len(basese[0])):
            chdks[k]=dcm*vec2mat2(basese[:,k])  # Partial Ch(matrix) over k-th weight
            cvdks[k] = chdks[k] * Rcv  # Partial Cv
            d1dks[k] = ddcm * vec2mat2(basese[:, k]) * trendedx + vec2mat2(
                basesedx[:, k]) * dcm  # Partial D1 over k-th weight
            d2dks[k] = ddcv * vec2mat2(basese[:, k]) * trendedz + vec2mat2(
                basesedz[:, k]) * dcv  # Partial D2 over k-th weight
        for i in range(numt[-1]):  # Stepwise forward evaluation
            # (1) evaluation of state field u
            cu = utens[i, :, :]
            utens[i + 1, :, :] = cu + chem * ((a1.T.dot(cu.T)).T) + chev * (a2.dot(cu)) + d1e * (
                (b1.T.dot(cu.T)).T) + d2e * (b2.dot(cu)) # Governing equation
            utens[i + 1, :, 0] = left * utens[i + 1, :,
                                        1]  # BC on left. So arranged so that Top/Bot always overwrites Left/Right.
            utens[i + 1, :, -1] = right * utens[i + 1, :, -2]  # BC on right.
            utens[i + 1, 0, :] = top * utens[i + 1, 1,
                                       :]  # BC on top, can be directly imposed according to 0=Dirichlet/1=Neumann.
            utens[i + 1, -1, :] = bot * utens[i + 1, -2, :]  # BC on bottom.
            # (2) evaluation of sensitivity fields s
            for k in range(len(basese[0])):
                csk=stens[i,:,:,k]
                chdk=chdks[k]  # Partial Ch
                cvdk=cvdks[k]  # Partial Cv
                d1dk=d1dks[k]  # Partial D1 over k-th weight
                d2dk=d2dks[k]  # Partial D2 over k-th weight
                stens[i+1,:,:,k]=csk+chdk*((a1.T.dot(cu.T)).T)+chem*((a1.T.dot(csk.T)).T)+\
                                 cvdk*(a2.dot(cu))+chev*(a2.dot(csk))+ \
                                 d1dk*((b1.T.dot(cu.T)).T)+d1e*((b1.T.dot(csk.T)).T)+\
                                 d2dk*(b2.dot(cu))+d2e*(b2.dot(csk))
                stens[i + 1, :, 0,k] = left * stens[i + 1, :,1,k]
                stens[i + 1, :, -1,k] = right * stens[i + 1, :, -2,k]
                stens[i + 1, 0, :,k] = top * stens[i + 1, 1, :,k]
                stens[i + 1, -1, :,k] = bot * stens[i + 1, -2, :,k]  # BC on bottom.
            # End of Loop k
        # End of Loop i
        # Loss and gradient evaluation
        loss_t=0
        gradients=np.zeros(len(basese[0]))
        for i in ukmat: # epwp loss
            uiloss,uigrad=epsilon_insensitive_loss_numpy(utens[numt,i[0],i[1]],uk[:,i[0],i[1]])
            loss_t+=np.mean(uiloss)*lamu # !Changed to ReLU
            for k in range(len(basese[0])): # for gradient, l=relu(e^2-t)dl/dk=dl
                gradients[k]+=np.mean(uigrad*(stens[numt,i[0],i[1],k]))*lamu
        for i in chkmat: # permeability loss
            ciloss,cigrad=epsilon_insensitive_loss_numpy(chem[i[0],i[1]],chk[i[0],i[1]])
            loss_t+=lam*np.mean(ciloss)*(1+Rcv)
            for k in range(len(basese[0])):
                gradients[k]+=lam*np.mean(cigrad*chem*(vec2mat2(basese[:,k])[i[0],i[1]]))*(1+Rcv) #*chem*(vec2mat2(basese[:,k])[i[0],i[1]])
        losses.append(loss_t)
        if j%10==0:
            print('Epoch ',j,': Loss=',loss_t,'; gradients=',gradients)
        # Backward propagation using Adam
        mvec=mvec*beta1+(1-beta1)*gradients
        vvec=vvec*beta2+(1-beta2)*(gradients**2)
        mhat=mvec/(1-beta1**(j+1))
        vhat=vvec/(1-beta2**(j+1))
        update=mhat/(np.sqrt(vhat)+0.00000000001)
        coefe-=lr*update # Can adjust to dynamic update if needed.
        if loss_t<=ltol or np.sum(update**2)<=gtol:
            print('Early-stopped at epoch ',j)
            break
    # End of loop j
    print('final loss=',loss_t)
    np.save('QAPILM_Eps_loss.npy',losses)

    return coefe,utens,chem,loss_t


def cos_sim(tenA,tenB):
    """
    Takes two tensors of the same field, flatten them if multi-dim, outputs their cosine similarity.
    """
    vecA=tenA.flatten()
    vecB=tenB.flatten()
    csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
    return csim


# 3.4 A function to validate the correctness of the analytical gradients. There may be minor errors bc machine precision.

def gradient_validator(numt,u0,Rcv,top,bot,left,right,ukmat,chkmat,uk,chk,lam,lr,ltol,gtol,itol,lamu):
    """
    Takes number of time step, measurements, and IC/BCs as inputs, outputs the weight vector.
    top~right are BC descriptions. 0=Dirichlet, 1=Neumann.
    lr=learning rate; lam=weight for loss terms; ltol=tolerance for loss; gtol->gradient; itol->num. of iterations.
    """
    coefe=(np.random.randn(len(basese[0]))-0.5)/5 # Random initialization. randn gives [0,1), so modification needed.
    utens=np.zeros((len(numt),int(numv+1),int(numh+1)),dtype=np.float64) # Initialize U-tensor
    stens=np.zeros((len(numt),int(numv+1),int(numh+1),len(basese[0])),dtype=np.float64) # Initialize S-tensor, sensitivity.
    utens[0,:,:]=u0 # Initial condition
    stens[0,:,:,:]=0 # Initial condition
    beta1=0.9
    beta2=0.999 # Hyperparameters for Adam optimization
    mvec=np.zeros(len(basese[0]))
    vvec=mvec.copy()
    llist=[]
    for j in range(itol): # Outer loop controlling the training process.
        chem = vec2mat2(np.exp(basese@coefe))  # Matrix form of horizontal permeability, estimation
        chev = chem * Rcv  # That of vertical permeability
        trendedx = vec2mat2(basesedx@coefe)
        trendedz = vec2mat2(basesedz@coefe)  # These are used to compute D1(D2) = Partial Ch(Cv) / partial x(z), estimation.
        d1e = chem * trendedx # Matrix form of Partial hor. perm. over x
        d2e = chev * trendedz
        for i in range(len(numt)-1):  # Stepwise forward evaluation
            # (1) evaluation of state field u
            cu = utens[i, :, :]
            utens[i + 1, :, :] = cu + chem * ((a1.T.dot(cu.T)).T) + chev * (a2.dot(cu)) + d1e * (
                (b1.T.dot(cu.T)).T) + d2e * (b2.dot(cu)) # Governing equation
            utens[i + 1, :, 0] = left * utens[i + 1, :,
                                        1]  # BC on left. So arranged so that Top/Bot always overwrites Left/Right.
            utens[i + 1, :, -1] = right * utens[i + 1, :, -2]  # BC on right.
            utens[i + 1, 0, :] = top * utens[i + 1, 1,
                                       :]  # BC on top, can be directly imposed according to 0=Dirichlet/1=Neumann.
            utens[i + 1, -1, :] = bot * utens[i + 1, -2, :]  # BC on bottom.
            # (2) evaluation of sensitivity fields s
            for k in range(len(basese[0])):
                csk=stens[i,:,:,k]
                stens[i+1,:,:,k]=csk+(chem*vec2mat2(basese[:,k]))*((a1.T.dot(cu.T)).T)+chem*((a1.T.dot(csk.T)).T)+\
                                 (chev*vec2mat2(basese[:,k]))*(a2.dot(cu))+chev*(a2.dot(csk))+\
                                 (chem*vec2mat2(basese[:,k])*trendedx)*((b1.T.dot(cu.T)).T)+\
                                 (chem*vec2mat2(basesedx[:,k]))*((b1.T.dot(cu.T)).T)+d1e*((b1.T.dot(csk.T)).T)+\
                                 (chev*vec2mat2(basese[:,k])*trendedz)*(b2.dot(cu))+\
                                 (chev*vec2mat2(basesedz[:,k]))*(b2.dot(cu))+d2e*(b2.dot(csk))
                stens[i + 1, :, 0,k] = left * stens[i + 1, :,1,k]
                stens[i + 1, :, -1,k] = right * stens[i + 1, :, -2,k]
                stens[i + 1, 0, :,k] = top * stens[i + 1, 1, :,k]
                stens[i + 1, -1, :,k] = bot * stens[i + 1, -2, :,k]  # BC on bottom.
            # End of Loop k
        # End of Loop i
        # Loss and gradient evaluation
        loss_t=0
        gradients=np.zeros(len(basese[0]))
        gradient_tan=gradients.copy() # Use small interval to approximate the gradients
        for i in ukmat: # epwp loss
            loss_t+=np.mean((utens-uk)[:,i[0],i[1]]**2)*lamu
            for k in range(len(basese[0])):
                gradients[k]+=np.mean(2*(utens-uk)[:,i[0],i[1]]*(stens[:,i[0],i[1],k]))*lamu
        for i in chkmat: # permeability loss
            loss_t+=lam*np.mean((chem-chk)[i[0],i[1]]**2)*(1+Rcv)
            for k in range(len(basese[0])):
                gradients[k]+=lam*np.mean(2*(chem-chk)[i[0],i[1]]*chem*(vec2mat2(basese[:,k])[i[0],i[1]]))*(1+Rcv)
        # Now validate using the forward solver
        eps=0.0000001
        for k in range(len(basese[0])):
            loss1=0 # to be compared with loss_t
            coefe1=coefe.copy()
            coefe1[k]+=eps
            chem1 = vec2mat2(np.exp(basese @ coefe1))
            loss1 = loss_evaluator(numt, u0, Rcv, coefe1, top, bot, left, right, ukmat, chkmat, uk, chk, lam,lamu)
            gradient_tan[k]+=(loss1-loss_t)/eps
        if j%1==0:
            print('Epoch ',j,': Loss=',loss_t,'; gradients=',gradients,'; approx.gradients=',gradient_tan,'; diff=',gradients-gradient_tan)
        llist=np.concatenate([llist,gradients-gradient_tan],axis=None)
        # Backward propagation using Adam
        mvec=mvec*beta1+(1-beta1)*gradients
        vvec=vvec*beta2+(1-beta2)*(gradients**2)
        mhat=mvec/(1-beta1**(j+1))
        vhat=vvec/(1-beta2**(j+1))
        update=mhat/(np.sqrt(vhat)+0.00000000001)
        coefe-=lr*update/np.linalg.norm(update) # So fixed because the gradients are too small, might be an error.
        if loss_t<=ltol or np.sum(update**2)<=gtol:
            print('Early-stopped at epoch ',j)
            break
    # End of loop j

    return llist

# 3.5 Define boundary conditions and other hyperparams, to run the entire program.
bcs=[0,1,0,1] # Define BC's. The top boundary must be Dirichlet due to dC/dz being negative.
lam=1 # loss weight (ratio)
lamu=1
lr=0.1 # learning rate
ltol=0.00001 # loss tolerance
gtol=0.000000001 # update tolerance
itol=500 # number of iterations/epochs

# 4. Define functional functions, to pack the procedure together.

# 4.1 Forward solving function to be called


def func_forw_sol(chm,rcv,ilim,u0,bcs):
    print('Now performing forward analysis. Output will be an interactive contour plot of the u field.')
    u1,u2=forward_solver(chm,rcv,ilim,u0,bcs[0],bcs[1],bcs[2],bcs[3]) # Validated. Normally you want to have Dirichlets on top and bottom boundaries
    fig=u_field_plot(u1,u2) # Validated. Can add a save function if needed.
    return fig


# 4.2 Inverse solving function to be called

def func_inv_sol(chm,rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,lamu):
    print('Now performing inverse analysis. Output will be the comparison of the ch fields.')
    u1,u2=forward_solver(chm,rcv,ilim,u0,bcs[0],bcs[1],bcs[2],bcs[3]) # Generate forward instance
    print('Forward (ground truth) computation complete')
    singleplot2D(oplt, 'plasma', 'Measurement layout')  # The layout of measurements.
    print('The number of basis functions used is: ',len(basese[0]))
    wlst, res, rank, s = np.linalg.lstsq(basese, isoftplus(trend + np.exp(fluc)-const), rcond=None)  # Ideal possible solution via LSM
    print('The LS solution of weights is: ',wlst)
    triplot2D(chm, vec2mat2(softplus(basese@wlst)), np.array(['C_h_real (m²/year)', 'C_h_lsm (m$^2$/year)', 'Error=ln(C_h_real/C_h_lsm)']),1,'viridis','bwr',None,None,-0.5,0.5)
    uk = u1[ukt, :, :] # take only the first ukt steps
    lstloss = loss_evaluator(ukt, u0, rcv, wlst, bcs[0], bcs[1], bcs[2], bcs[3], ukmat, chkmat, uk, chm, lam,lamu)
    lstcsim = cos_sim(chm, vec2mat2(softplus(basese @ wlst)))
    print('lst_loss=', lstloss, '; LST_csim=', lstcsim)
    ce,ue,che,loss=inverse_solver(ukt,u0,rcv,bcs[0],bcs[1],bcs[2],bcs[3],ukmat,chkmat,uk,chm,lam,lr,ltol,gtol,itol,wlst,lamu)
    print('The PIML solution of weights is: ',ce)
    #dualplot3D(chm,che,'red','blue','real','pred')
    csim=cos_sim(chm,che)
    print('Solution cosine similarity=',csim)
    triplot2D(chm, che, np.array(['C_h_real (m²/year)', 'C_h_pred (m$^2$/year)', 'Error']),0,'viridis','bwr',None,None,-1,1)
    triplot2D(u1[ukt[-1]], ue[ukt[-1]], np.array(['Real u/u_0 at t=t_N', 'Predicted u/u_0 at t=t_N', 'Error']),0,'jet','bwr',0,1,-0.2,0.2)
    triplot2D(u1[int(0.1/dt)-1], ue[int(0.1/dt)-1], np.array(['Real u/u_0 at t=0.1 year', 'Predicted u/u_0 at t=0.1 year', 'Error']),0,'jet','bwr',0,1,-0.2,0.2)

    return ce


# 4.3 Grad validating function to be called


def func_grad_val(chm,rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,lamu):
    print('Now performing gradient validation. Output will be the comparison of ana/est gradients.')
    u1,u2=forward_solver(chm,rcv,ilim,u0,bcs[0],bcs[1],bcs[2],bcs[3]) # Generate forward instance
    print('Forward (ground truth) computation complete')
    print('The number of basis functions used is: ',len(basese[0]))
    uk = u1[ukt, :, :] # take only the first ukt steps
    llist=gradient_validator(ukt,u0,rcv,bcs[0],bcs[1],bcs[2],bcs[3],ukmat,chkmat,uk,chm,lam,lr,ltol,gtol,int(itol/50),lamu) # Validated. Correct.
    ln2=np.sum(llist**2)
    token=False # to return
    if ln2==0:
        print('Validated! Total (squared sum of) difference is ',ln2)
        token=True
    else:
        print('Validation unsuccessful. Total num. of non-zero grad value is ',np.count_nonzero(llist))
    return token


# 4.4 Inverse param analysis to be called (batch solving, random init)

def pararesultsaver(chm,rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,lamu,filename=None,solution=None):
    # (1) Create directory
    base_dir="./results"
    if filename is None:
        filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir=os.path.join(base_dir,filename)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    print('Now performing inverse analysis. Output will be stored in',experiment_dir)

    # (2) perform solution
    u1, u2 = forward_solver(chm, rcv, ilim, u0, bcs[0], bcs[1], bcs[2], bcs[3])  # Generate forward instance
    print('Forward (ground truth) computation complete')
    print('The number of basis functions used is: ', len(basese[0]))
    uk = u1[ukt, :, :]  # take only the first ukt steps
    if solution is None:
        wlst, res, rank, s = np.linalg.lstsq(basese, isoftplus(trend + np.exp(fluc) - const), rcond=None)
        ce, ue, che, loss = inverse_solver(ukt, u0, rcv, bcs[0], bcs[1], bcs[2], bcs[3], ukmat, chkmat, uk, chm, lam, lr, ltol,
                                 gtol, itol, wlst, lamu)
        print('The PIML solution of weights is: ', ce)
    else:
        ce = solution
        loss,ue,che=loss_evaluator(ukt,u0,Rcv,ce,bcs[0], bcs[1], bcs[2], bcs[3],ukmat,chkmat,uk,chm,lam,lamu)
    csim = cos_sim(chm, che)
    print('Solution complete! Cosine similarity=', csim)

    # (3) save data results
    filepath = os.path.join(experiment_dir, "config_n_data.txt")
    tensor_dict = {
        'bcs': bcs,
        'spatial_reso': spatial_reso,
        'sigma': sigma,
        'lams': [lam,lamu],
        'lr/ltol/gtol/itol': [lr,ltol,gtol,itol],
        'Rcv': rcv,
        'ukmat': ukmat,
        'chkmat': chkmat,
        'dt': dt,
        'ukts': [ukt[[0,1,-1]]*dt],
        'C_range': [np.min(chm),np.max(chm)],
        'nbases': [len(basese[0])],
        'coef_pred': ce,
        'u_pred': ue,
        'C_pred': che,
        'u_err': ue-u1[np.arange(ukt[-1]+1)],
        'C_err': che-chm,
        'fin_loss': loss,
        'csim': csim,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=== Store data in tensors ===\n")
        f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        for var_name, tensor in tensor_dict.items():
            tensor=np.array(tensor)
            f.write(f"item: {var_name}\n")
            f.write(f"shape: {tensor.shape}\n")
            f.write(f"datatype: {tensor.dtype}\n")
            f.write(f"data:\n")

            # Save wrt different tensor shape
            if tensor.ndim == 0:
                f.write(f"{tensor}\n")
            elif tensor.ndim == 1:
                for i, val in enumerate(tensor):
                    f.write(f"  [{i}] = {val:.8f}\n")
            elif tensor.ndim == 2:
                if len(tensor)<=15:
                    for i in range(tensor.shape[0]):
                        f.write("  [")
                        for j in range(tensor.shape[1]):
                            f.write(f"{tensor[i, j]:.6f}")
                            if j < tensor.shape[1] - 1:
                                f.write(", ")
                        f.write("]\n")
                else:
                    for i in range(2):
                        f.write("  [")
                        for j in range(tensor.shape[1]):
                            f.write(f"{tensor[i, j]:.3f}")
                            if j < tensor.shape[1] - 1:
                                f.write(", ")
                        f.write("]\n")
            elif tensor.ndim == 3:
                f.write("  [time, z, x] format\n")
                for t in range(min(5, tensor.shape[0])):  # Only the first 5 snapshots
                    f.write(f"  time snapshot {t}:\n")
                    for i in range(min(5, tensor.shape[1])):  # top-5 rows
                        f.write("    [")
                        for j in range(min(5, tensor.shape[2])):  # top-5 columns
                            f.write(f"{tensor[t, i, j]:.3f}")
                            if j < min(5, tensor.shape[2]) - 1:
                                f.write(", ")
                        f.write("...]\n" if tensor.shape[2] > 5 else "]\n")
                    if tensor.shape[1] > 5:
                        f.write("    ...\n")
                    f.write("\n")

            if tensor.size:
                f.write(f"stats:\n")
                f.write(f"  min: {np.min(tensor):.6f}\n")
                f.write(f"  max: {np.max(tensor):.6f}\n")
                f.write(f"  mean: {np.mean(tensor):.6f}\n")
                f.write(f"  mae: {np.mean(np.abs(tensor)):.6f}\n")
                f.write(f"  std: {np.std(tensor):.6f}\n")
                f.write("-" * 40 + "\n\n")

    print(f"tensor data saved to: {filepath}")

    # (4) save plots
    triplot2D(chm, che, np.array(['C_h_real (m²/year)', 'C_h_pred (m$^2$/year)', 'Error']), 0, 'viridis', 'bwr', None,
              None, -1, 1, os.path.join(experiment_dir,"C_comparison.png"))
    triplot2D(u1[ukt[-1]], ue[ukt[-1]], np.array(['Real u/u_0 at t=t_N', 'Predicted u/u_0 at t=t_N', 'Error']), 0,
              'jet', 'bwr', 0, 1, -0.2, 0.2, os.path.join(experiment_dir,"U_comp_tN.png"))
    triplot2D(u1[int(0.1 / dt) - 1], ue[int(0.1 / dt) - 1],
              np.array(['Real u/u_0 at t=0.1 year', 'Predicted u/u_0 at t=0.1 year', 'Error']), 0, 'jet', 'bwr', 0, 1,
              -0.2, 0.2, os.path.join(experiment_dir,"U_comp_0.1y.png"))

    print(f"All plots saved!")
    return ce

# 4.5 Output measurements for validation.
def create_dataset(chm,rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,lamu,filename_prefix='consolidation'):
    print('Now performing fwd analysis. Will save the results to file.')
    u1, u2 = forward_solver(chm, rcv, ilim, u0, bcs[0], bcs[1], bcs[2], bcs[3])  # Generate forward instance
    uk=u1[ukt,:,:]
    print('Forward (ground truth) computation complete')
    c_arr=np.zeros((len(chkmat),3),dtype=np.float64)
    print(c_arr.shape)
    u_arr=np.zeros((len(ukmat)*len(ukt),4),dtype=np.float64)
    print(u_arr.shape)
    #  mcpts2=np.array(cpts2).reshape(numh+1,numv+1,2).transpose(1,0,2)
    for i in range(len(chkmat)):
        point=chkmat[i]
        x0=point[1]/(numh-1)*Lh
        y0=point[0]/(numv-1)*Lv
        c_arr[i,0]=x0
        c_arr[i,1]=y0
        c_arr[i,2]=chm[point[0],point[1]]
    print(c_arr)
    np.save(f'{filename_prefix}_C.npy',c_arr)  #  Validated.
    for i in range(len(ukmat)):
        point=ukmat[i]
        x0=point[1]/(numh-1)*Lh
        y0=point[0]/(numv-1)*Lv
        for j in range(len(ukt)):
            u_arr[i*len(ukt)+j,0]=x0
            u_arr[i*len(ukt)+j,1]=y0
            u_arr[i*len(ukt)+j,2]=ukt[j]*dt
            u_arr[i*len(ukt)+j,3]=uk[j,point[0],point[1]]
    np.save(f'{filename_prefix}_u.npy',u_arr)  #  Validated.

    return 0


# To be added.


# 4.X Final governing function. One ring to rule them all, one ring to find them. One ring to bring them all, and in darkness bind them.


def func_ruling_ring(chm,rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,mode,lamu,fname,solu):

    out=0
    if mode==1:
        out=func_forw_sol(chm,rcv,ilim,u0,bcs)
    elif mode==2:
        out=func_inv_sol(chm,rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,lamu)
    elif mode==3:
        out=func_grad_val(chm,rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,lamu)
    elif mode==4:
        out=pararesultsaver(chm,rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,lamu,fname,solu)
    elif mode == 5:
        out = create_dataset(chm, rcv, ilim, u0, bcs, ukmat, chkmat, ukt, lam, lr, ltol, gtol, itol, lamu)
    else:
        print('Invalid input for mode. Please select from 1 to 4.')

    return out


mode=4 # Analysis mode. 1 for forward, 2 for inverse, 3 for grad val, 4 for batch sensitivity analysis
solu=None
tasklist=10
ukmatl=[[5,3],[10,3],[21,3],[3,25],[6,25],[13,25],[2,18],[14,18],[22,18]] # Deleted are put here.
chkmatl=np.array([[4,6],[6,43],[23,49]])  # These are the original configurations. Deleted:
ukt=np.arange(0,int(1/dt),1) # List of measurement temporal location. The tunable number is in years; outputs a list
ulist=[ukmatl,ukmatl[0:5],ukmatl[0:2]]
clist=[chkmatl,chkmatl[0:1],[chkmatl[0]]]
fnlist=['PAUQ1U3C.csv','PAUQ6U3C.csv','PAUQ3U3C.csv','PAUQ9U2C.csv','PAUQ9U1C.csv']
coefs=np.zeros([tasklist,len(basese[0])])
ukmat=ulist[0]
chkmat=clist[0]
print(np.shape(coefs))
for j in [0]: # j is the length of the tasklist. For non-parametric use, set to [0].

    if j<=2:
        chkmat=clist[0]
        #ukmat=ulist[j]
        ukmat=[[10,3]] # If 1 measurement of u, use the one in the left column.
        #lamu=0 # If 0 measurements of c, then set its loss weight to 0.
    elif j>2:
        chkmat=clist[j-2]
        ukmat=ulist[0]
    print(chkmat)
    print(ukmat)

    for i in range(tasklist):

        print(datetime.now())

        fname = fnlist[j]+f'_{i+1}_in_{j}'  # Top Left; Outer; 01->SCALING FACTOR
        print(fname)
        # ux=25 # 10 for left, 25 for center, 40 for right
        coefs[i]=func_ruling_ring(chm,Rcv,ilim,u0,bcs,ukmat,chkmat,ukt,lam,lr,ltol,gtol,itol,mode,lamu,fname,solu)
        print(i,' finished')
        print(datetime.now())

    np.savetxt(fnlist[j],coefs,delimiter=',')
    coefs*=0

    print(f'{j} complete')