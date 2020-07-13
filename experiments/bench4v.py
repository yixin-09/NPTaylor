from pygsl.testing import sf
from mpmath import *
import numpy as np
mp.dps = 40

#f1
gf1 = lambda x,y,z,p: sf.ellint_RJ(x,y,z,p,0)
rf1 = lambda x,y,z,p: elliprj(x,y,z,p)

#f2
gf2 = lambda x,y,z,p: sf.hyperg_2F1(x,y,z,p)
rf2 = lambda x,y,z,p: hyp2f1(x,y,z,p)

#f3
gf3 = lambda x,y,z,p: sf.hyperg_2F1_conj(x,y,z,p)
rf3 = lambda x,y,z,p: re(hyp2f1(x+y*j,x-y*j,z,p))

#f4
gf4 = lambda x,y,z,p: sf.hyperg_2F1_renorm(x,y,z,p)
rf4 = lambda x,y,z,p: hyp2f1(x,y,z,p)/gamma(z)

#f5
gf5 = lambda x,y,z,p: sf.hyperg_2F1_conj_renorm(x,y,z,p)
rf5 = lambda x,y,z,p: re(hyp2f1(x+y*j,x-y*j,z,p))/gamma(z)


input_domain = [[[[0, 10.0], [0, 10.0], [0, 10.0], [0, 10.0]]], [[[0, 10], [0, 10], [-10, 0], [-1, 1]]], [[[0, 10], [0, 10], [-10, 0], [-1, 1]]], [[[0, 10], [0, 10], [-10, 0], [-1, 1]]], [[[0, 10], [0, 10], [-10, 0], [-1, 1]]]]
rfl = [rf1, rf2, rf3, rf4, rf5]
gfl = [gf1, gf2, gf3, gf4, gf5]
nrfl_fname = [u'gsl_sf_ellint_RJ', u'gsl_sf_hyperg_2F1 ', u'gsl_sf_hyperg_2F1_conj', u'gsl_sf_hyperg_2F1_renorm', u'gsl_sf_hyperg_2F1_conj_renorm']
ngfl_fname = [u'gsl_sf_ellint_RJ', u'gsl_sf_hyperg_2F1 ', u'gsl_sf_hyperg_2F1_conj', u'gsl_sf_hyperg_2F1_renorm', u'gsl_sf_hyperg_2F1_conj_renorm']
#[[[0, 10000000000.0], [0, 10000000000.0], [0, 10000000000.0], [0, 10000000000.0]]]