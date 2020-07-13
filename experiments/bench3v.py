from pygsl.testing import sf
from mpmath import *
import numpy as np
mp.dps = 40

#f1
gf1 = lambda x,y,z: sf.ellint_P(x,y,z,0)
rf1 = lambda x,y,z: ellippi(-z,x,y*y)

#f2
gf2 = lambda x,y,z: sf.ellint_RD(x,y,z,0)
rf2 = lambda x,y,z: elliprd(x,y,z)

#f3
gf3 = lambda x,y,z: sf.ellint_RF(x,y,z,0)
rf3 = lambda x,y,z: elliprf(x,y,z)

#f4
gf4 = lambda x,y,z: sf.beta_inc(x,y,z)
rf4 = lambda x,y,z: re(betainc(x,y,0,z))/beta(x,y)

#f5
gf5 = lambda x,y,z: sf.hyperg_1F1(x,y,z)
rf5 = lambda x,y,z: hyp1f1(x,y,z)

#f6
gf6 = lambda x,y,z: sf.hyperg_U(x,y,z)
rf6 = lambda x,y,z: power(z,-x)*hyp2f0(x,1.0+x-y,power(-z,-1.0))

#f7
gf7 = lambda x,y,z: sf.hyperg_2F0(x,y,z)
rf7 = lambda x,y,z: hyp2f0(x,y,z)

input_domain = [[[[0, 100], [0, 100], [0, 100]]], [[[0, 100], [0, 100], [0, 100]]], [[[0, 100], [0, 100], [0, 100]]], [[[0, 100], [0, 100], [0, 1]]], [[[0, 100], [0, 100], [0, 100]]], [[[0, 10], [0, 10], [0, 10]]], [[[0, 10], [0, 10], [-10, 0]]]]
rfl = [rf1, rf2, rf3, rf4, rf5, rf6, rf7]
gfl = [gf1, gf2, gf3, gf4, gf5, gf6, gf7]
nrfl_fname = [u'gsl_sf_ellint_P', u'gsl_sf_ellint_RD ', u'gsl_sf_ellint_RF', u'gsl_sf_beta_inc', u'gsl_sf_hyperg_1F1', u'gsl_sf_hyperg_U ', u'gsl_sf_hyperg_2F0']
ngfl_fname = [u'gsl_sf_ellint_P', u'gsl_sf_ellint_RD ', u'gsl_sf_ellint_RF', u'gsl_sf_beta_inc', u'gsl_sf_hyperg_1F1', u'gsl_sf_hyperg_U ', u'gsl_sf_hyperg_2F0']
