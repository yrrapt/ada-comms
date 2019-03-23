import numpy as np
import matplotlib.pyplot as plt

# system variables
fs = 100e3
f = 1e3
phi = np.pi/4
N = 4*fs/f
n_var = 0.01

# create some empty vectors to fill
x = np.zeros(N, dtype=complex)
n_a = np.zeros(N, dtype=complex)
e = np.zeros(N)
w = np.zeros(N)
y = np.zeros(N, dtype=complex)
y_ = np.zeros(N, dtype=complex)
w_ = np.zeros(N)

# loop through performing esitmation
for n in xrange(int(N)):

	# create reference signal
	x[n] = np.exp(1j*(2*n*np.pi*f/fs + phi))

	# create noise to get received signal
	n_a[n] = float(np.random.normal(0, np.sqrt(n_var), 1)) + 1j*float(np.random.normal(0, np.sqrt(n_var), 1))
	y[n] = x[n] + n_a[n]

	# create the estimated signal
	y_[n] = np.exp(1j*sum(w_))

	# create the error signal
	e[n] = y[n] * y_[n]

	# create new frequency estimate
	w_[n] = e[n]

# plot the results
plt.plot(np.real(x))
plt.plot(np.imag(y_))
plt.title("Maximum Likelihood Phase Estimation")
plt.xlabel("samples")
plt.ylabel("amplitude")
plt.show()