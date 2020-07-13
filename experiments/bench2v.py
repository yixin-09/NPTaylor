from pygsl.testing import sf
from mpmath import *
import numpy as np
mp.dps = 40
from scipy import special
#f1
gf1 = lambda x,y: sf.bessel_Jnu(x,y)
rf1 = lambda x,y: besselj(x,y)
# print gf1(1.0, 1.0)
#f2
gf2 = lambda x,y: sf.bessel_Ynu(x,y)
rf2 = lambda x,y: bessely(x,y)

#f3
gf3 = lambda x,y: sf.bessel_Inu(x,y)
rf3 = lambda x,y: besseli(x,y)
# print gf3(0.0, 287.3502843671995)
# print rf3(0.0, 287.3502843671995)
#f4
gf4 = lambda x,y: sf.bessel_Inu_scaled(x,y)
rf4 = lambda x,y: fmul(besseli(x,y),exp(-y),exact=True)


#f5
gf5 = lambda x,y: sf.bessel_Knu(x,y)
rf5 = lambda x,y: besselk(x,y)
# print gf5(0.0, 1.7976931348623155e+308)
# print rf5(0.0, -1.7976931348623155e+208)
#f6
gf6 = lambda x,y: sf.bessel_lnKnu(x,y)
rf6 = lambda x,y: log(besselk(x,y))

#f7
gf7 = lambda x,y: sf.bessel_Knu_scaled(x,y)
rf7 = lambda x,y: fmul(besselk(x,y),exp(y),exact=True)

#f8
gf8 = lambda x,y: sf.hydrogenicR_1(x,y)
rf8 = lambda x,y: fmul(2.0*x,fmul(sqrt(x),exp(-fmul(x,y,exact=True)),exact=True),exact=True)

#f9
gf9 = lambda x,y: sf.ellint_Pcomp(x,y,0)
rf9 = lambda x,y: fsub(elliprf(0.0,fsub(1.0,power(x,2.0),exact=True),1.0),fmul((y),elliprj(0.0,fsub(1.0,power(x,2.0),exact=True),1.0,fadd(1.0,y,exact=True)),exact=True)/3.0,exact=True)

#f10
gf10 = lambda x,y: sf.ellint_F(x,y,0)
rf10 = lambda x,y: ellipf(x,y*y)


#f11
gf11 = lambda x,y: sf.ellint_E(x,y,0)
rf11 = lambda x,y: ellipe(x,y*y)

#f12
gf12 = lambda x,y: sf.ellint_D(x,y,0)
rf12 = lambda x,y: fmul(power(np.sin(x),3.0),elliprd(fsub(1.0,pow(np.sin(x),2.0),exact=True),fsub(1,fmul(pow(y,2.0),power(np.sin(x),2.0),exact=True),exact=True),1),exact=True)/3.0

#f13
gf13 = lambda x,y: sf.ellint_RC(x,y,0)
rf13 = lambda x,y: elliprc(x,y)

#f14
gf14 = lambda x,y: sf.exp_mult(x,y)
rf14 = lambda x,y: fmul(y,exp(x),exact=True)

#f15
gf15 = lambda x,y: sf.poch(x,y)
rf15 = lambda x,y: gamma(fadd(x,y,exact=True))/gamma(x)

#f16
gf16 = lambda x,y: sf.lnpoch(x,y)
rf16 = lambda x,y: fsub(loggamma(fadd(x,y,exact=True)),loggamma(x),exact=True)

#f17
gf17 = lambda x,y: sf.pochrel(x,y)
rf17 = lambda x,y: fsub(fdiv(gamma(fadd(x,y,exact=True)),gamma(x)),1,exact=True)/y


#f18
gf18 = lambda x,y: sf.gamma_inc(x,y)
rf18 = lambda x,y: gammainc(x,y)
# print gf18(-338.122,0.122)
# print rf18(-338.122,0.122)
#f19
gf19 = lambda x,y: sf.gamma_inc_Q(x,y)
rf19 = lambda x,y: gammainc(x,y,regularized=True)

#f20
gf20 = lambda x,y: sf.gamma_inc_P(x,y)
rf20 = lambda x,y: gammainc(x,0,y,regularized=True)

#f21
gf21 = lambda x,y: sf.beta(x,y)
rf21 = lambda x,y: beta(x,y)

#f22
gf22 = lambda x,y: sf.lnbeta(x,y)
rf22 = lambda x,y: ln(beta(x,y))
# print gf22(1.4812745831311185, 1.7976931348623157e+8)
# print rf22(1.4812745831311185, 1.7976931348623157e+8)
# print special.betaln(1.0454509778092025, 0.9577104453780154)
# print rf22(1.0454509778092025, 0.9577104453780154)
# print beta(1.0454509778092025, 0.9577104453780154)
# print "%.18e" % special.beta(1.0454509778092025, 0.9577104453780154)
# print gf22(1.0454509778092025, 0.9577104453780154)
# print "%.18e" % sf.beta(1.0454509778092025, 0.9577104453780154)
# print "%.18e" % beta(1.0454509778092025, 0.9577104453780154)
# import numpy
# a = numpy.float128(1)
# b = numpy.float128(1e16)
# import time
# st = time.time()
# for i in range(10000):
#     a+b-b
#     # 1+1e16-1e16
# print time.time()-st
# print a+b-b
# print 1+1e16-1e16
# n = numpy.float128(6755399441055744.0)
# x = numpy.float128(1.6)
# print (x + n) - n
#f23
gf23 = lambda x,y: sf.gegenpoly_1(x,y)
rf23 = lambda x,y: gegenbauer(1,x,y)

#f24
gf24 = lambda x,y: sf.gegenpoly_2(x,y)
rf24 = lambda x,y: gegenbauer(2,x,y, accurate_small=False)

#f25
gf25 = lambda x,y: sf.gegenpoly_3(x,y)
rf25 = lambda x,y: gegenbauer(3,x,y, accurate_small=False) if x!=-2.0 else fmul(12,y,exact=True)
# print gf25(-1.7976931348623155e+8, -1.7976931348623155e+8)
# print rf25(-1.7976931348623155e+8, -1.7976931348623155e+8)
# print gf25(-1.999999999762337444, -7.944474534863850567e+04)
# print rf25(-1.999999999762337444, -7.944474534863850567e+04)
# print rf25(-1.9999999997623334, -79444.12713162278)
#f26
gf26 = lambda x,y: sf.hyperg_0F1(x,y)
rf26 = lambda x,y: hyp0f1(x,y)

#f27
gf27 = lambda x,y: sf.laguerre_1(x,y)
rf27 = lambda x,y: fadd(1.0,fsub(x,y,exact=True),exact=True)

#f28
gf28 = lambda x,y: sf.laguerre_2(x,y)
rf28 = lambda x,y: laguerre(2, x, y)
# print gf28(-1.7976931348623155e+308, -1.7976931348623155e+308)
# print rf28(-1.7976931348623157e+308, -1.7963071803437983e+308)
#f29
gf29 = lambda x,y: sf.laguerre_3(x,y)
rf29 = lambda x,y: laguerre(3, x, y)

#f30
gf30 = lambda x,y: sf.conicalP_half(x,y)
rf30 = lambda x,y: re(legenp(-0.5+j*x,0.5,y,type=3))

#f31
gf31 = lambda x,y: sf.conicalP_mhalf(x,y)
rf31 = lambda x,y: re(legenp(-0.5+j*x,-0.5,y,type=3))
# print gf31(0.0, 1.7976931348623155e+308)
# print rf31(0.0, 1.7976931348623155e+308)
#f32
gf32 = lambda x,y: sf.conicalP_0(x,y)
rf32 = lambda x,y: re(legenp(-0.5+j*x,0,y,type=3))
# print gf32(19.997926865290395, 1.4369317698853783)
# print rf32(19.997926865290395, 1.4369317698853783)
#f33
def gf33(x,y):
    a = list(sf.conicalP_1_e(x, y))
    if len(a) == 1:
        return a[0]
    else:
        return a[1]
rf33 = lambda x,y: re(legenp(-0.5+j*x,1,y,type=3))
# print gf33(490.76270861475155, 1.0)
# print rf33(-3,13)
#f34
gf34 = lambda x,y: sf.legendre_H3d_0(x,y)
rf34 = lambda x,y: sin(fmul(x,y,exact=True))/(fmul(x,sinh(y),exact=True))

#f35
gf35 = lambda x,y: sf.legendre_H3d_1(x,y)
rf35 = lambda x,y: fmul(sin(fmul(x,y)),(fsub(coth(y),x*cot(fmul(x,y)),exact=True)),exact=True)/fmul(sqrt(fadd(power(x,2.0),1.0,exact=True)),(fmul(x,sinh(y))),exact=True)

#f36
gf36 = lambda x,y: sf.hypot(x,y)
rf36 = lambda x,y: sqrt(fadd(power(x,2.0),power(y,2.0),exact=True))

#f37
gf37 = lambda x,y: sf.hzeta(x,y)
rf37 = lambda x,y: autoprec(zeta)(x,y)

#f38
gf38 = lambda x,y: sf.multiply(x,y)
rf38 = lambda x,y: fmul(x,y,exact=True)

#f39
gf39 = lambda x,y: sf.fermi_dirac_inc_0(x,y)
rf39 = lambda x,y: float(-1.0*polylog(1.0,-exp(y-x)))-(y-x) if y-x > -50.0 else float(exp(y-x))-(y-x)

input_domain = [[[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[-1, 1], [0, 10000000000.0]]], [[[-100, 100], [-1, 1]]], [[[-100, 100], [-1, 1]]], [[[-1.57079632679,1.57079632679],[-9.999999999999998890e-01,9.999999999999998890e-01]]], [[[0, 10000000000.0], [0, 10000000000.0]]], [[[-708, 709], [-10000000000.0, 10000000000.0]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[0, 100], [0, 100]]], [[[-10000000000.0, 10000000000.0], [-10000000000.0, 10000000000.0]]], [[[-10000000000.0, 10000000000.0], [-10000000000.0, 10000000000.0]]], [[[-100000.0, 100000.0], [-100000.0, 100000.0]]], [[[-100, 100], [-100, 100]]], [[[-100000.0, 100000.0], [-100000.0, 100000.0]]], [[[-100000.0, 100000.0], [-100000.0, 100000.0]]], [[[-100000.0, 100000.0], [-100000.0, 100000.0]]], [[[-100.0, 100.0], [0, 100000.0]]], [[[0, 100.0], [0, 100000.0]]], [[[0, 100.0], [0, 100000.0]]], [[[0, 100.0], [0, 100000.0]]], [[[0, 1000.0], [0, 709]]], [[[0, 10000000000.0], [0, 709]]], [[[-13407807929.942596, 13407807929.942596], [-13407807929.942596, 13407807929.942596]]], [[[1, 100], [0, 100]]], [[[-100000.0, 100000.0], [-100000.0, 100000.0]]], [[[-300, 300], [0, 309]]]]
rfl = [rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9, rf10, rf11, rf12, rf13, rf14, rf15, rf16, rf17, rf18, rf19, rf20, rf21, rf22, rf23, rf24, rf25, rf26, rf27, rf28, rf29, rf30, rf31, rf32, rf33, rf34, rf35, rf36, rf37, rf38, rf39]
gfl = [gf1, gf2, gf3, gf4, gf5, gf6, gf7, gf8, gf9, gf10, gf11, gf12, gf13, gf14, gf15, gf16, gf17, gf18, gf19, gf20, gf21, gf22, gf23, gf24, gf25, gf26, gf27, gf28, gf29, gf30, gf31, gf32, gf33, gf34, gf35, gf36, gf37, gf38, gf39]
nrfl_fname = [u'gsl_sf_bessel_Jnu', u'gsl_sf_bessel_Ynu', u'gsl_sf_bessel_Inu ', u'gsl_sf_bessel_Inu_scaled', u'gsl_sf_bessel_Knu', u'gsl_sf_bessel_lnKnu', u'gsl_sf_bessel_Knu_scaled', u'gsl_sf_hydrogenicR_1', u'gsl_sf_ellint_Pcomp', u'gsl_sf_ellint_F', u'gsl_sf_ellint_E', u'gsl_sf_ellint_D', u'gsl_sf_ellint_RC', u'gsl_sf_exp_mult', u'gsl_sf_poch', u'gsl_sf_lnpoch', u'gsl_sf_pochrel', u'gsl_sf_gamma_inc', u'gsl_sf_gamma_inc_Q', u'gsl_sf_gamma_inc_P', u'gsl_sf_beta', u'gsl_sf_lnbeta', u'gsl_sf_gegenpoly_1', u'gsl_sf_gegenpoly_2', u'gsl_sf_gegenpoly_3', u'gsl_sf_hyperg_0F1', u'gsl_sf_laguerre_1', u'gsl_sf_laguerre_2', u'gsl_sf_laguerre_3', u'gsl_sf_conicalP_half', u'gsl_sf_conicalP_mhalf', u'gsl_sf_conicalP_0', u'gsl_sf_conicalP_1', u'gsl_sf_legendre_H3d_0', u'gsl_sf_legendre_H3d_1', u'gsl_sf_hypot', u'gsl_sf_hzeta', u'gsl_sf_multiply', u'gsl_sf_fermi_dirac_inc_0']
ngfl_fname = [u'gsl_sf_bessel_Jnu', u'gsl_sf_bessel_Ynu', u'gsl_sf_bessel_Inu ', u'gsl_sf_bessel_Inu_scaled', u'gsl_sf_bessel_Knu', u'gsl_sf_bessel_lnKnu', u'gsl_sf_bessel_Knu_scaled', u'gsl_sf_hydrogenicR_1', u'gsl_sf_ellint_Pcomp', u'gsl_sf_ellint_F', u'gsl_sf_ellint_E', u'gsl_sf_ellint_D', u'gsl_sf_ellint_RC', u'gsl_sf_exp_mult', u'gsl_sf_poch', u'gsl_sf_lnpoch', u'gsl_sf_pochrel', u'gsl_sf_gamma_inc', u'gsl_sf_gamma_inc_Q', u'gsl_sf_gamma_inc_P', u'gsl_sf_beta', u'gsl_sf_lnbeta', u'gsl_sf_gegenpoly_1', u'gsl_sf_gegenpoly_2', u'gsl_sf_gegenpoly_3', u'gsl_sf_hyperg_0F1', u'gsl_sf_laguerre_1', u'gsl_sf_laguerre_2', u'gsl_sf_laguerre_3', u'gsl_sf_conicalP_half', u'gsl_sf_conicalP_mhalf', u'gsl_sf_conicalP_0', u'gsl_sf_conicalP_1', u'gsl_sf_legendre_H3d_0', u'gsl_sf_legendre_H3d_1', u'gsl_sf_hypot', u'gsl_sf_hzeta', u'gsl_sf_multiply', u'gsl_sf_fermi_dirac_inc_0']
