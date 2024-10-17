import numpy as np
from scipy.optimize import curve_fit
import icf


# This function will generate a perfect Gaussian 
# You can replace this function with any other you want to fit with...
# added y0
def gaussian(x, *params):

	A = params[0]
	x0 = params[1]
	c = params[2]
	y0 = params[3]
		
	return y0 + A*np.exp(-(x-x0)**2/(2*c*c))


def supergaussian(x, *params):

	A = params[0]
	x0 = params[1]
	c = params[2]
	y0 = params[3]
	
	return y0 + A*np.exp(-((x-x0)/(np.sqrt(2)*c))**n)

def supergaussian_linear(x, *params):

	A = params[0]
	x0 = params[1]
	c = params[2]
	y0 = params[3]
	z0 = params[4]
	
	return y0 + z0*x**3 + A*np.exp(-((x-x0)/(np.sqrt(2)*c))**n)

#
# This section will make a numpy array containing a gaussian 
#

# # This makes a numpy array with 100 equally spaced points between 0 and 4
# xdata = np.linspace(0,4,100)

# # This makes a gaussian using these x points
# ydata = gaussian(xdata, 3, 2, 0.2)

# Lets add some noise
# for i in range(len(ydata)):
# 	ydata[i] +=  0.4*(np.random.random_sample()-0.5)

# Create x/y data from a lineout file
xdata1 , ydata1 = icf.load_2col ("lineout1.csv")
xdata1 = xdata1 * 60.0/3.5
maskmin1, maskmax1 = 480/3.5, 2700/3.5
maskmin1, maskmax1 = 100/3.5, 3000/3.5
maskmin3, maskmax3 = 350/3.5, 2500/3.5
maskmin4, maskmax4 = 650/3.5, 2700/3.5

xdata2 , ydata2 = icf.load_2col ("lineout2.csv")
xdata2 = xdata2 * 60.0/3.5

xdata3 , ydata3 = icf.load_2col ("lineout3.csv")
xdata3 = xdata3 * 60.0/3.5

xdata4 , ydata4 = icf.load_2col ("lineout4.csv")
xdata4 = xdata4 * 60.0/3.5

# mask the data to pass into the fitting function
mask1 = (xdata1 >=maskmin1) & (xdata1 <= maskmax1)
xdata1_mask = xdata1[mask1]
ydata1_mask = ydata1[mask1]

mask2 = (xdata2 >=maskmin1) & (xdata2 <= maskmax1)
xdata2_mask = xdata2[mask2]
ydata2_mask = ydata2[mask2]

mask3 = (xdata3 >=maskmin3) & (xdata3 <= maskmax3)
xdata3_mask = xdata3[mask3]
ydata3_mask = ydata3[mask3]

mask4 = (xdata4 >=maskmin4) & (xdata4 <= maskmax4)
xdata4_mask = xdata4[mask4]
ydata4_mask = ydata4[mask4]

#
# This section will do a fit
#

# This does the fit, and returns the fit parameters and the covariances
# guess now has 4 parameters to include y0

# cycle through each value of n=1,...,8 to find the optimum supergaussian fit
Rvals = []
nvals = [2, 4, 6, 8, 10]
for i in nvals:
	guess = np.array([3000,1600,500,28000, 3.5])/3.5
	n = i
	res, cov = curve_fit(supergaussian_linear, xdata2_mask, ydata2_mask, p0=guess)
	yfit = supergaussian_linear(xdata2_mask, *res)
	Rvals.append(icf.r_squared(ydata2_mask, yfit))

n = nvals[np.argmax(Rvals)]

print(f'The best value of n is: {n}')

guess = np.array([3000,1600,500,28000, 3.5])/3.5
print("Our initial guess is", guess)
popt, pcov = curve_fit(supergaussian_linear, xdata2_mask, ydata2_mask, p0=guess)

for i in range(len(popt)):
	print ("Parameter",i,":",popt[i],"+/-",np.sqrt(pcov[i][i]))
	
print("Fit parameters : ", popt)
print("Fit standard deviations : ", np.sqrt(np.diag(pcov)))

# This generates a new list with a Gaussian using the identified fit parameters
# This data is therefore the best fit curve 

yfit = supergaussian(xdata2_mask, *popt, n)

print("R^2 = ", icf.r_squared(ydata2_mask, yfit))

# This will plot the output, both the original data and the best fit, as well as a residual
# Note this is a special plotting routine written for the icf labs, hence the 'icf' prefix
# The source code can be found in icf.py if you want to copy/alter it
 
icf.fit_plot(xdata2, ydata2, xdata2_mask, ydata2_mask, yfit)

