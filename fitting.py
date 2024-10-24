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
	
	return y0 + z0*x + A*np.exp(-((x-x0)/(np.sqrt(2)*c))**n)

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
xdata , ydata = icf.load_2col ("line4.2.csv")
xdata = xdata * 60.0/3.5
# IMPORTANT: Mask must be changed manually
maskmin, maskmax = 270, 1250

# mask the data to pass into the fitting function
mask = (xdata >=maskmin) & (xdata <= maskmax)
xdata_mask = xdata[mask]
ydata_mask = ydata[mask]

#
# This section will do a fit
#

# This does the fit, and returns the fit parameters and the covariances
# guess now has 4 parameters to include y0

# cycle through each value of n=1,...,8 to find the optimum supergaussian fit
Rvals = []
nvals = [2, 4, 6, 8, 10]
for i in nvals:
	guess = np.array([2000,700,150,28000, 1])
	n = i
	res, cov = curve_fit(supergaussian_linear, xdata, ydata, p0=guess)
	yfit = supergaussian_linear(xdata, *res)
	Rvals.append(icf.r_squared(ydata, yfit))

n = nvals[np.argmax(Rvals)]

print(f'The values of R are {Rvals}')
print(f'The best value of n is: {n}')

guess = np.array([2000,700,150,28000, 1])
print("Our initial guess is", guess)
popt, pcov = curve_fit(supergaussian_linear, xdata, ydata, p0=guess)

for i in range(len(popt)):
	print ("Parameter",i,":",popt[i],"+/-",np.sqrt(pcov[i][i]))
	
print("Fit parameters : ", popt)
print("Fit standard deviations : ", np.sqrt(np.diag(pcov)))

# This generates a new list with a Gaussian using the identified fit parameters
# This data is therefore the best fit curve 

yfit = supergaussian_linear(xdata_mask, *popt, n)


print("R^2 = ", icf.r_squared(ydata_mask, yfit))

# This will plot the output, both the original data and the best fit, as well as a residual
# Note this is a special plotting routine written for the icf labs, hence the 'icf' prefix
# The source code can be found in icf.py if you want to copy/alter it
 
icf.fit_plot(xdata, ydata, xdata_mask, ydata_mask, yfit)

