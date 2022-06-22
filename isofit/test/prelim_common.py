import numpy as np
# Test for load_wavelen(...)

def load_wavelen(wavelength_file: str):

    """""
    Load a wavelength file, and convert to nanometers if needed.
    Args:
        wavelength_file: file to read wavelengths from
    Returns:
        (np.array, np.array): wavelengths, full-width-half-max
    """""

    q = np.loadtxt(wavelength_file)
    print(q)
    if q.shape[1] > 2:
        q = q[:, 1:3]
    if q[0, 0] < 100:
        q = q * 1000.0
    wl, fwhm = q.T
    return wl, fwhm



print("BEGIN")

myList = ["Ames", "Marshall", "Goddard", "Kennedy"]

for i in myList:
    print(i)




# Case 1: More than 2 columns



#wl = np.random.rand(425,4)
#file = open("wl_sample.txt", "w+")
#np.savetxt("wl_sample.txt", wl)

#wl_modified, fwhm_modified = load_wavelen("C:/Users/vpatro/Desktop/wl_sample.txt")
#file.close()
#assert(wl_modified.ndim == 1)
#assert(fwhm_modified.ndim == 1)
#assert(wl_modified[0] > 100)
















#wl_modified, fwhm_modified = load_wavelen("C:/Users/vpatro/Desktop/Test/wl_multicol.txt")
#print(wl_modified.shape)

    



print("FINISHED")
