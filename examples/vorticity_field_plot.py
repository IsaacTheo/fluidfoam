# Contained in this code is: - Script for computing frames for an animation
#                            - Some useful functions I created for use with this library
# Note that running this code will not produce an animation. It will produce the frames for it which I then recommend compiling into an animation using ffmpeg.
# Here is an example of the ffmpeg command I use to compile the frames into an animation:
# ffmpeg -r <desired fps> -i <frame names>.png -c:v libx264 -profile:v high -crf 10 (increase this number if you want the animation file to take up less memory) -pix_fmt yuv420p -y <animation name>.mp4

from fluidfoam import readmesh, readvector, readscalar
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle

def setup_case(case,xres,yres,xmin,xmax,ymin,ymax):
    '''
    Reads simulation directory to allow you to start pulling data (such as the vorticity fields) out of it.
    case: str

        Simulation directory

    xres,yres: int

        Horizontal and vertical resolutions of the image you will produce. For the 90x30 meshes, use xres=720 and yres=240. For the 30X30 meshes, use xres=yres=240.

    xmin,xmax,ymin,ymax:

        Minimum and maximum x and y spatial dimensions of the mesh. For example, the 90x30 mesh will have (xmin,xmax,ymin,ymax)=(-10,80,-15,15), and the short mesh will have (xmin,xmax,ymin,ymax)=(-10,20,-15,15).
    '''
    print('Reading case')

    x, y, z = readmesh(case)

    xi = np.linspace(xmin, xmax, xres)
    yi = np.linspace(ymin, ymax, yres)

    # Structured grid creation
    xinterp, yinterp = np.meshgrid(xi, yi)

    print('Readmesh complete')

    return x,y,z,xi,yi,xinterp,yinterp
  
def preserve_ints(a):
    '''
    Removes decimal from any points in array which are integers and returns new array with that correction applied.
    '''
    new_a = []
    for i in range(len(a)):
        if (a[i] - int(a[i])) == 0:
            new_a.append(int(a[i]))
        else:
            new_a.append(a[i])

    return new_a

def norm_the_vort(vortz_i,n,thresh,minvort,maxvort):
    '''
    Normalizes and scales the vorticity field to make the vortices clearly visible when it is plotted. If this isn't used (at least in my case) you will see nothing in the vorticity field.
    '''
    # Values of n between 0.1 and 0.01 seem to work best for achieving the desired scaling
    norm_vort = np.zeros(vortz_i.shape)
    for i in range(vortz_i.shape[0]):
        for j in range(vortz_i.shape[1]):
            v = vortz_i[i,j]
            if v < -thresh: # Blue vortices
                v = (v / abs(minvort))
                # if v > -0.368:
                    # norm_vort[i,j] = v
                # else:
                norm_vort[i,j] = -np.power(abs(v),n/np.sqrt(abs(v)))
            elif v > thresh: # Red vortices
                v = (v / abs(maxvort))
                # if v < 0.368:
                #     norm_vort[i,j] = v
                # else:
                norm_vort[i,j] = np.power(v,n/np.sqrt(v))
            else:
                norm_vort[i,j] = 0
    
    return norm_vort
  
case_path = '<path>' # Case path
case = setup_case(case_path,240,240,-10,20,-15,15)

times = preserve_ints(np.arange(900.25,1000.25,0.25)) # My time steps were from 900.25 to 1000 seconds in step of 0.25. Make sure you add 1*dt to the last value of time. np.arange was the best option here but unfortunately it is not inclusive for the last element
# time = [1000] # Uncomment this line and use it instead to test and make sure your code is working and the frames look how you want them to before running for all time steps.

# What the following for loop will do is find the maximum and minimum vorticity values for every timestep. From that list of extrema, you can you can pick out the highest max and lowest min over all time steps to use it for the scaling of the vorticity field. If you don't use different extrema for scaling your vorticity field for each time step, you will see that the vortices "pulse" which looks odd and is definitely non-physical behavior.
extrema = [] # List of extrema for vorticity fields
for i in times:
    vort = readvector(case_path, str(times[i]), 'vorticity') # Reading the vorticity field from file
    extrema.append(np.nanmin(vort[2,:])) # Using nanmin and nanmax here is super important. There will be probably be lots of nans in a 2D simulation in the vorticity field and you WILL get your extrema recorded as nans if your not careful. Thankfully, numpy has us covered.
    extrema.append(np.nanmax(vort[2,:])) # The indexing I'm using here ([2,:]) is for selecting the z component of the vorticity vectors which is the only thing I need to know in my case.

min_vort = np.nanmin(extrema)
max_vort = np.nanmax(extrema)

# You will almost definitely have to play around with the following two values (n and thresh) to fit your case. They impact the visibility of vortices a lot.
n = 0.03 # Values between 0.05 and 0.01 work best. The lower the value, the more intense and uniform the colour of the vortices will be
thresh = 0.02
for i in range(len(times)):
  # Normalize vorticity field
    vort = readvector(case_path, str(times[i]), 'vorticity') # Reading the vorticity field from file
    vortz_i = griddata((case[0],case[1]), vort[2,:], (case[5],case[6]), method='linear') # Assigning the vorticity field the spatial coordinate you gave it when you called the setup_case function above
    norm_vort = norm_the_vort(vortz_i,n,thresh,min_vort,max_vort) # Normalize and rescale vorticity field to a visible data set. I chose the range [-1,1] but you will have to go into the function norm_the_vort function and mess with it if you want a different range
    
    # Initialize plot
    fig = plt.figure(figsize=(10, 6), constrained_layout=True, dpi=250)
    plt.rcParams['font.family'] = 'monospace'
    gs = fig.add_gridspec(1, 1) # I am using gridspec here because I was putting several animations side-by-side to compare them in my own code. If you only want to do one plot then just take all of the gridspec stuff and just use the classic plt.plot functions. This is probably the easiest part of the code to do by oneself. Also the most flexible and subjective. # The (1,1) is in the matrix index notation i.e. (# of rows, # of columns).
    fig.suptitle('<plot title>', size=15, weight='bold')

    # Subplots
    rect = plt.Rectangle((-0.05, -0.5), 0.1, 1, fc='k', edgecolor='k', fill=True, zorder=10) # This is the thin flat plate in my case. Note that if you're going to be making two contours side by side for comparison, you will have to redefine the rectangle above the code for the next subplot. You can do this by just copying and pasting this exact line of code below. You don't even need to change the name 'rect'.
    ax0 = fig.add_subplot(gs[0,:]) # Long
    cont = ax0.contourf(case[5],case[6],norm_vort, 125, cmap='seismic')
    ax0.add_patch(rect)
    ax0.set_ylabel('<your y label>'); ax0.set_xlabel('<your x label>')
    ax0.text(-7.5,6.3, 'Time: ' + str(times[i])) # Time text
    ax0.set_xlim([-8,50]); ax0.set_ylim([-8,8]) # The window I chose. This is very much optional.
    
    cbar_ax = fig.add_axes([1.02, 0.08, 0.02, 0.85])
    plt.colorbar(cont, label=r'$\Omega$', cax=cbar_ax, format='%.1f')

    plt.savefig(f'<frame name>{i:04}.png') # The '{i:04}' is important because it gives your frames different names. Otherwise it would be overwriting them after every iteration through the for loop
    plt.close() # Prevents figure from showing in case you want to save memory while computing all frames. Comment this line out while testing to see if the frames looks the way you want them to
    # plt.show()
