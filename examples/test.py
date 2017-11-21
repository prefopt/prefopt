from __future__ import print_function
import prefopt
import numpy as np

# flake8: noqa

'''
exp = prefopt.PreferenceExperiment([[0.0,10.0]])
exp.X = []
for i in xrange(0,10):
	exp.X.append([i])
exp.X = np.array(exp.X,dtype=np.float32)

exp.pref_idxs = []
for i in xrange(0,10):
	for k in xrange(i+1,10):
		exp.pref_idxs.append([k,i])
'''

'''
n = 20.0
exp = prefopt.PreferenceExperiment([[0.0,n],[0.0,n]])
exp.X = []
for i in xrange(0,int(n)):
	exp.X.append([i,np.random.uniform(low=0.0,high=n)])
exp.X = np.array(exp.X,dtype=np.float32)

exp.pref_idxs = []
for i in xrange(0,int(n)):
	for k in xrange(i+1,int(n)):
		exp.pref_idxs.append([k,i])
'''

'''
exp = prefopt.PreferenceExperiment([[0.0,10.0],[0.0,10.0]])
exp.X = []
for i in xrange(0,3):
	exp.X.append([i,np.random.uniform(low=0.0,high=10.0)])
exp.X = np.array(exp.X,dtype=np.float32)


exp.pref_idxs = []
for i in xrange(0,3):
	for k in xrange(i+1,3):
		exp.pref_idxs.append([k,i])
'''

'''
exp.model()
exp.inference()
'''


n = 20.0
exp = prefopt.PreferenceExperiment([[0.0,n],[0.0,n]])
# optionally seed the latent function with selected examples X and known latent values y
#exp.init (X ,y)
x1 = [1.0,6.0]
x2 = [5.0,1.0]
x3 = [14.0,2.0]
exp.prefer(x1,x2,-1)
exp.prefer(x1,x3,-1)
exp.prefer(x2,x3,-1)

def pref_func_eq_test(x1,x2):
	if abs(x1[0]-x2[0]) < 0.2:
		return 0
	else:
		return np.sign(np.array(x1)[0]-np.array(x2)[0])

exp.model()
exp.inference()
#pref_func = lambda x1,x2 : np.sign(np.array(x1)[0]-np.array(x2)[0])
pref_func = pref_func_eq_test
xb = x3
for i in xrange (0, 7) :
	xn = exp.find_next(xb, acquisition_function="expected_improvement",
                    verbose=True)
	# get preference input from user xn vs xb
	exp.prefer(xb, xn, pref_func(xb,xn))
	print(xn," vs ",xb)
	if pref_func(xb,xn) == -1:
		xb = xn
	exp.model()
	exp.inference()

print("Q_f     mean : ",exp.qf.mean().eval())
print("X            : ",exp.X_np)
print("Q_gamma mean : ",exp.qgamma.mean().eval())
