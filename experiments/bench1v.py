from pygsl.testing import sf
from mpmath import *
import numpy as np
import src.basic_func as bf
mp.dps = 1000

#f1
gf1 = lambda x: sf.airy_Ai(x,0)
rf1 = lambda x: airyai(x)

#f2
gf2 = lambda x: sf.airy_Bi(x,0)
rf2 = lambda x: airybi(x)

#f3
gf3 = lambda x: sf.airy_Ai_scaled(x,0)
rf3 = lambda x: fmul(exp(fmul(2.0/3.0,power(x,3.0/2.0),exact=True)),airyai(x),exact=True) if x>0 else airyai(x)

#f4
gf4 = lambda x: sf.airy_Bi_scaled(x,0)
rf4 = lambda x: fmul(exp(fmul(-2.0/3.0,power(x,3.0/2.0),exact=True)),airybi(x),exact=True) if x>0 else airybi(x)

#f5
gf5 = lambda x: sf.airy_Ai_deriv(x,0)
rf5 = lambda x: airyai(x,1)

#f6
gf6 = lambda x: sf.airy_Bi_deriv(x,0)
rf6 = lambda x: airybi(x,1)

#f7
gf7 = lambda x: sf.airy_Ai_deriv_scaled(x,0)
rf7 = lambda x: fmul(exp(fmul(2.0/3.0,power(x,3.0/2.0),exact=True)),airyai(x,1),exact=True) if x>0 else airyai(x,1)

#f8
gf8 = lambda x: sf.airy_Bi_deriv_scaled(x,0)
rf8 = lambda x: fmul(exp(fmul(-2.0/3.0,power(x,3.0/2.0),exact=True)),airybi(x,1),exact=True) if x>0 else airybi(x,1)

#f9
gf9 = lambda x: sf.bessel_J0(x)
rf9 = lambda x: besselj(0,x)

#f10
gf10 = lambda x: sf.bessel_J1(x)
rf10 = lambda x: besselj(1,x)

#f11
gf11 = lambda x: sf.bessel_Y0(x)
rf11 = lambda x: bessely(0,x)

#f12
gf12 = lambda x: sf.bessel_Y1(x)
rf12 = lambda x: bessely(1,x)

#f13
gf13 = lambda x: sf.bessel_I0(x)
rf13 = lambda x: besseli(0,x)

#f14
gf14 = lambda x: sf.bessel_I1(x)
rf14 = lambda x: besseli(1,x)

#f15
gf15 = lambda x: sf.bessel_I0_scaled(x)
rf15 = lambda x: fmul(besseli(0,x),exp(-fabs(x)),exact=True)

#f16
gf16 = lambda x: sf.bessel_I1_scaled(x)
rf16 = lambda x: fmul(besseli(1,x),exp(-fabs(x)),exact=True)

#f17
gf17 = lambda x: sf.bessel_K0(x)
rf17 = lambda x: besselk(0,x)

#f18
gf18 = lambda x: sf.bessel_K1(x)
rf18 = lambda x: besselk(1,x)

#f19
gf19 = lambda x: sf.bessel_K0_scaled(x)
rf19 = lambda x: fmul(besselk(0,x),(exp(x)),exact=True) if (x>0) else besselk(0,x)

#f20
gf20 = lambda x: sf.bessel_K1_scaled(x)
rf20 = lambda x: fmul(besselk(1,x),(exp(x)),exact=True) if (x>0) else besselk(1,x)

#f21
gf21 = lambda x: sf.bessel_j0(x)
rf21 = lambda x: fmul(sqrt(pi/(2*x)),besselj(0.5,x),exact=True)

#f22
gf22 = lambda x: sf.bessel_j1(x)
rf22 = lambda x: -fmul(sqrt(pi/(2*x)),besselj(1.5,x),exact=True)

#f23
gf23 = lambda x: sf.bessel_j2(x)
rf23 = lambda x: -fmul(sqrt(pi/(2*x)),besselj(2.5,x),exact=True)

#f24
gf24 = lambda x: sf.bessel_y0(x)
# rf24 = lambda x: fmul(sqrt(pi/(2*x)),bessely(0.5,x),exact=True)
rf24 = lambda x: -cos(x)/x

#f25
gf25 = lambda x: sf.bessel_y1(x)
# rf25 = lambda x: fmul(sqrt(pi/(2*x)),bessely(1.5,x),exact=True)
rf25 = lambda x: -(cos(x)/x + sin(x)) / x

#f26
gf26 = lambda x: sf.bessel_y2(x)
rf26 = lambda x: fmul(sqrt(pi/(2*x)),bessely(2.5,x),exact=True)

#f27
gf27 = lambda x: sf.bessel_i0_scaled(x)
# rf27 = lambda x: re(fmul(exp(-fabs(x)),besseli(0.5,x)/sqrt(2.0*x/pi),exact=True))
rf27 = lambda x: re(fmul(exp(-fabs(x)),besseli(0.5,x)/sqrt(2.0*x/pi),exact=True))

#f28
gf28 = lambda x: sf.bessel_i1_scaled(x)
# rf28 = lambda x: re(fmul(exp(-fabs(x)),besseli(1.5,x)/sqrt(2.0*x/pi),exact=True))
rf28 = lambda x: re(fmul(exp(-fabs(x)),fdiv(besseli(1.5,x),sqrt(fmul(2.0,x)/pi)),exact=True))


#f29
gf29 = lambda x: sf.bessel_i2_scaled(x)
rf29 = lambda x: re(fmul(exp(-fabs(x)),besseli(2.5,x)/sqrt(2.0*x/pi),exact=True))

#f30
gf30 = lambda x: sf.bessel_k0_scaled(x)
rf30 = lambda x: fmul(fmul(exp(x),besselk(0.5,x),exact=True),sqrt(pi/(2*x)),exact=True)

#f31
gf31 = lambda x: sf.bessel_k1_scaled(x)
rf31 = lambda x: fmul(fmul(exp(x),besselk(1.5,x),exact=True),sqrt(pi/(2*x)),exact=True)

#f32
gf32 = lambda x: sf.bessel_k2_scaled(x)
rf32 = lambda x: fmul(fmul(exp(x),besselk(2.5,x),exact=True),sqrt(pi/(2*x)),exact=True)

#f33
gf33 = lambda x: sf.clausen(x)
rf33 = lambda x: clsin(2,x)

#f34
gf34 = lambda x: sf.dawson(x)
rf34 = lambda x: fmul(sqrt(pi),fmul(exp(-x*x),erfi(x),exact=True),exact=True)/2.0

#f35
gf35 = lambda x: sf.debye_1(x)
rf35 = lambda x: quad(lambda t: power(t,1)/(exp(t)-1),[0,x])/x

#f36
gf36 = lambda x: sf.debye_2(x)
rf36 = lambda x: 2.0 * quad(lambda t: power(t,2.0)/(exp(t)-1),[0,x])/power(x,2.0)

#f37
gf37 = lambda x: sf.debye_3(x)
rf37 = lambda x: 3.0 * quad(lambda t: power(t,3.0)/(exp(t)-1),[0,x])/power(x,3.0)

#f38
gf38 = lambda x: sf.debye_4(x)
rf38 = lambda x: 4.0 * quad(lambda t: power(t,4.0)/(exp(t)-1),[0,x])/power(x,4.0)

#f39
gf39 = lambda x: sf.debye_5(x)
rf39 = lambda x: 5.0 * quad(lambda t: power(t,5.0)/(exp(t)-1),[0,x])/power(x,5.0)

#f40
gf40 = lambda x: sf.debye_6(x)
rf40 = lambda x: 6.0 * quad(lambda t: power(t,6.0)/(exp(t)-1.0),[0,x])/power(x,6.0)

#f41
gf41 = lambda x: sf.dilog(x)
rf41 = lambda x: re(polylog(2,x))

#f42
gf42 = lambda x: sf.ellint_Kcomp(x,0)
rf42 = lambda x: ellipk(power(x,2.0))

#f43
gf43 = lambda x: sf.ellint_Ecomp(x,0)
rf43 = lambda x: ellipe(power(x,2.0))

#f44
gf44 = lambda x: sf.erf(x)
rf44 = lambda x: erf(x)

#f45
gf45 = lambda x: sf.erfc(x)
rf45 = lambda x: erfc(x)


#f46
gf46 = lambda x: sf.log_erfc(x)
rf46 = lambda x: log(erfc(x))

#f47
gf47 = lambda x: sf.erf_Z(x)
rf47 = lambda x: exp(-power(x,2.0)/2)/sqrt(2*pi)

#f48
gf48 = lambda x: sf.erf_Q(x)
rf48 = lambda x: erfc(x/sqrt(2.0))/2.0

#f49
gf49 = lambda x: sf.hazard(x)
rf49 = lambda x: fmul(sqrt(2.0/pi),exp(-power(x,2.0)/2.0),exact=True)/erfc(x/sqrt(2.0))


#f50
gf50 = lambda x: sf.exp(x)
rf50 = lambda x: exp(x)

#f51
gf51 = lambda x: sf.exp_e10_e(x)[1]
rf51 = lambda x: exp(x)

#f52
gf52 = lambda x: sf.expm1(x)
rf52 = lambda x: expm1(x)

#f53
gf53 = lambda x: sf.exprel(x)
rf53 = lambda x: (fsub(exp(x),1,exact=True))/x

#f54
gf54 = lambda x: sf.exprel_2(x)
rf54 = lambda x: 2.0*(fsub(exp(x),fadd(1.0,x,exact=True),exact=True))/(power(x,2.0))

#f55
gf55 = lambda x: sf.expint_E1(x)
rf55 = lambda x: expint(1,x)

#f56
gf56 = lambda x: sf.expint_E2(x)
rf56 = lambda x: expint(2,x)

#f57
gf57 = lambda x: sf.expint_Ei(x)
rf57 = lambda x: ei(x)

#f58
gf58 = lambda x: sf.Shi(x)
rf58 = lambda x: shi(x)
# print gf58(-1.7976931348623155e+108)
# print float(rf58(-1.7976931348623155e+108))
#f59
gf59 = lambda x: sf.Chi(x)
rf59 = lambda x: chi(x)

#f60
gf60 = lambda x: sf.expint_3(x)
rf60 = lambda x: quad(lambda t: exp(-power(t,3.0)),[0,x])

#f61
gf61 = lambda x: sf.Si(x)
rf61 = lambda x: si(x)

#f62
gf62 = lambda x: sf.Ci(x)
rf62 = lambda x: ci(x)

#f63
gf63 = lambda x: sf.atanint(x)
rf63 = lambda x: quad(lambda t: atan(t)/t,[0,x])

#f64
gf64 = lambda x: sf.fermi_dirac_m1(x)
rf64 = lambda x: exp(x)/(fadd(1,exp(x),exact=True))

#f65
gf65 = lambda x: sf.fermi_dirac_0(x)
rf65 = lambda x: -1.0*polylog(1.0,-exp(x)) if x > -50.0 else exp(x)

#f66
gf66 = lambda x: sf.fermi_dirac_1(x)
rf66 = lambda x: -1.0*polylog(2.0,-exp(x)) if x > -50.0 else exp(x)

#f67
gf67 = lambda x: sf.fermi_dirac_2(x)
rf67 = lambda x: -1.0*polylog(3.0,-exp(x)) if x > -50.0 else exp(x)

#f68
gf68 = lambda x: sf.fermi_dirac_mhalf(x)
rf68 = lambda x: quad(lambda t: 1/fmul(sqrt(t),(fadd(exp(fsub(t,x,exact=True)),1)),exact=True),[0,inf])/gamma(0.5) if x > -50.0 else exp(x)

#f69
gf69 = lambda x: sf.fermi_dirac_half(x)
rf69 = lambda x: quad(lambda t: sqrt(t)/(fadd(exp(fsub(t,x,exact=True)),1)),[0,inf])/gamma(1.5) if x > -50.0 else exp(x)

#f70
gf70 = lambda x: sf.fermi_dirac_3half(x)
rf70 = lambda x: quad(lambda t: fmul(t,sqrt(t),exact=True)/(fadd(exp(fsub(t,x,exact=True)),1)),[0,inf])/gamma(2.5) if x > -50.0 else exp(x)

#f71
gf71 = lambda x: sf.gamma(x)
rf71 = lambda x: gamma(x)

#f72
gf72 = lambda x: sf.lngamma(x)
rf72 = lambda x: loggamma(x)

#f73
gf73 = lambda x: sf.gammastar(x)
rf73 = lambda x: gamma(x)/(fmul(sqrt(2*pi),fmul(power(x,(fsub(x,0.5,exact=True))),exp(-x),exact=True),exact=True))

#f74
gf74 = lambda x: sf.gammainv(x)
rf74 = lambda x: rgamma(x)

#f75
gf75 = lambda x: sf.lambert_W0(x)
rf75 = lambda x: lambertw(x)

#f76
gf76 = lambda x: sf.lambert_Wm1(x)
rf76 = lambda x: re(lambertw(x,-1)) if (x<0.0) & (x>(-1.0/2.71828182845904523536028747135)) else sf.lambert_W0(x)

#f77
gf77 = lambda x: sf.legendre_P1(x)
rf77 = lambda x: legendre(1,x)

#f78
gf78 = lambda x: sf.legendre_P2(x)
rf78 = lambda x: legendre(2,x)
print gf78(7.741001517595158e+153)
print float(rf78(7.741001517595158e+153))

#f79
gf79 = lambda x: sf.legendre_P3(x)
rf79 = lambda x: legendre(3,x)

#f80
gf80 = lambda x: sf.legendre_Q0(x)
rf80 = lambda x: legenq(0,0,x,type=3).real

#f81
gf81 = lambda x: sf.legendre_Q1(x)
rf81 = lambda x: legenq(1,0,x,type=3).real

#f82
gf82 = lambda x: sf.log(x)
rf82 = lambda x: log(x)

#f83
gf83 = lambda x: sf.log_abs(x)
rf83 = lambda x: log(abs(x))

#f84
gf84 = lambda x: sf.log_1plusx(x)
rf84 = lambda x: log(fadd(1,x,exact=True))

#f85
gf85 = lambda x: sf.log_1plusx_mx(x)
rf85 = lambda x: fsub(log(fadd(1,x,exact=True)),x,exact=True)

#f86
gf86 = lambda x: sf.psi(x)
rf86 = lambda x: digamma(x)

#f87
gf87 = lambda x: sf.psi_1piy(x)
rf87 = lambda x: re(psi(0,1+x*j))

#f88
gf88 = lambda x: sf.psi_1(x)
rf88 = lambda x: psi(1,x)

#f89
gf89 = lambda x: sf.synchrotron_1(x)
rf89 = lambda x: fmul(x,quad(lambda t: besselk(5.0/3.0,t),[x,inf]),exact=True) if x< 60 else fmul(x,quad(lambda t: besselk(5.0/3.0,t),linspace(x,x+21,5)+[inf]),exact=True)

#f90
gf90 = lambda x: sf.synchrotron_2(x)
rf90 = lambda x: fmul(x,besselk(2.0/3.0,x),exact=True)

#f91
gf91 = lambda x: sf.transport_2(x)
# rf91 = lambda x: quad(lambda t: fmul(power(t,2.0),exp(t),exact=True)/power((fsub(exp(t),1.0,exact=True)),2.0),[0,x])
rf91 = lambda x: quad(lambda t: fmul(power(t,2.0),exp(t),exact=True)/power((fsub(exp(t),1.0)),2.0),[0,x])

#f92
gf92 = lambda x: sf.transport_3(x)
rf92 = lambda x: quad(lambda t: fmul(power(t,3.0),exp(t),exact=True)/power((fsub(exp(t),1.0)),2.0),[0,x])

#f93
gf93 = lambda x: sf.transport_4(x)
rf93 = lambda x: quad(lambda t: fmul(power(t,4.0),exp(t),exact=True)/power((fsub(exp(t),1.0)),2.0),[0,x])

#f94
gf94 = lambda x: sf.transport_5(x)
rf94 = lambda x: quad(lambda t: fmul(power(t,5.0),exp(t),exact=True)/power((fsub(exp(t),1.0)),2.0),[0,x])

#f95
gf95 = lambda x: sf.sin(x)
rf95 = lambda x: sin(x)

#f96
gf96 = lambda x: sf.cos(x)
rf96 = lambda x: cos(x)

#f97
gf97 = lambda x: sf.sinc(x)
# rf97 = lambda x: sinc(fmul(np.pi,x,exact=True))
rf97 = lambda x: sinc(np.pi*x)

#f98
gf98 = lambda x: sf.lnsinh(x)
rf98 = lambda x: log(sinh(x))
#f99
gf99 = lambda x: sf.lncosh(x)
rf99 = lambda x: log(cosh(x))

#f100
gf100 = lambda x: sf.angle_restrict_symm(x)
rf100 = lambda x: fmod(x,2.0*pi) if fmod(x,2.0*pi)< pi else fmod(x,2.0*pi)-2.0*pi

#f101
gf101 = lambda x: sf.angle_restrict_pos(x)
rf101 = lambda x: fmod(x,2*pi)

#f102
gf102 = lambda x: sf.zeta(x)
rf102 = lambda x: zeta(x)

#f103
gf103 = lambda x: sf.zetam1(x)
# rf103 = lambda x: fsub(zeta(x),1)
rf103 = lambda x: fsub(zeta(x),1,exact=True) if x < 120 else fsub(zeta(x,dps=400),1.0,exact=True)

#f104
gf104 = lambda x: sf.eta(x)
rf104 = lambda x: altzeta(x)
input_domain = [[[-1000.0, 1000.0]], [[-1000.0, 100.0]], [[-1000.0, 1000.0]], [[-1000.0, 1000.0]], [[-1000.0, 1000.0]], [[-1000.0, 100.0]], [[-1000.0, 1000.0]], [[-1000.0, 100.0]], [[-1e+100, 1e+100]], [[-1e+100, 1e+100]], [[-10000000000.0, 10000000000.0]], [[-10000000000.0, 10000000000.0]], [[-709, 709]], [[-709, 709]], [[-1.7976931348623157e+308, 1.7976931348623157e+308]], [[-1.7976931348623157e+308, 1.7976931348623157e+308]], [[0, 1e+100]], [[0, 1e+100]], [[0, 17000000000.0]], [[0, 17000000000.0]], [[-1.7976931348623157e+100, 1.7976931348623157e+100]], [[-1.3407807929942596e+154, 1.3407807929942596e+154]], [[-823549.6645, 823549.6645]], [[0, 823549.6645]], [[0, 823549.6645]], [[0, 823549.6645]], [[-8.98846567431158e+107, 8.98846567431158e+107]], [[-1.3407807929942596e+154, 1.3407807929942596e+154]], [[-5.64380309412229e+102, 5.64380309412229e+102]], [[0, 8.988465674311579e+307]], [[0, 8.988465674311579e+307]], [[0, 8.988465674311579e+307]], [[-823549.6645, 823549.6645]], [[-709, 709]], [[0, 709]], [[0, 709]], [[0, 709]], [[0, 709]], [[0, 709]], [[0, 709]], [[-1e+100, 1]], [[-1, 1]], [[-1, 1]], [[-10, 10]], [[-40, 40]], [[-1.073315388749996e+51, 1.073315388749996e+51]], [[-40, 40]], [[-10, 40]], [[-40, 40]], [[-709.782712893384, 709.782712893384]], [[-709.782712893384, 709.782712893384]], [[-709.782712893384, 709.782712893384]], [[-709.782712893384, 709.782712893384]], [[-709.782712893384, 709.782712893384]], [[-701.8334146820821, 701.8334146820821]], [[-701.8334146820821, 701.8334146820821]], [[-701.8334146820821, 701.8334146820821]], [[-701.8334146820821, 701.8334146820821]], [[-701.8334146820821, 701.8334146820821]], [[0, 3]], [[-1e+100, 1e+100]], [[0, 823549.6645]], [[-17976931348.623158, 17976931348.623158]], [[-709, 709]], [[-709, 709]], [[-709, 30]], [[-709, 30]], [[-708, 30]], [[-708, 30]], [[-708, 30]], [[-168, 171.6243769563027]], [[0, 1000]], [[0, 17976931348.623158]], [[-168, 171.6243769563027]], [[-0.4678794411714423, 1e+100]], [[-0.4678794411714423, 1e+100]], [[-1e+100, 1e+100]], [[-1e+100, 1e+100]], [[-1e+100, 1e+100]], [[-1, 1e+100]], [[-1, 1e+100]], [[0, 1e+100]], [[-1e+100, 1e+100]], [[-1, 1e+100]], [[-1, 1e+100]], [[-262144.0, 262144.0]], [[0, 1.3407807929942596e+154]], [[-5, 10000000000.0]], [[0, 500]], [[0, 500]], [[0, 1351079888211.1487]], [[0, 1351079888211.1487]], [[0, 1351079888211.1487]], [[0, 1351079888211.1487]], [[-823549.6645, 823549.6645]], [[-823549.6645, 823549.6645]], [[-262143.99997369398, 262143.99997369398]], [[0, 1e+100]], [[-1e+100, 1e+100]], [[-823549.6645, 823549.6645]], [[-823549.6645, 823549.6645]], [[-170, 1000]], [[-170, 1000]], [[-168, 100]]]
rfl = [rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9, rf10, rf11, rf12, rf13, rf14, rf15, rf16, rf17, rf18, rf19, rf20, rf21, rf22, rf23, rf24, rf25, rf26, rf27, rf28, rf29, rf30, rf31, rf32, rf33, rf34, rf35, rf36, rf37, rf38, rf39, rf40, rf41, rf42, rf43, rf44, rf45, rf46, rf47, rf48, rf49, rf50, rf51, rf52, rf53, rf54, rf55, rf56, rf57, rf58, rf59, rf60, rf61, rf62, rf63, rf64, rf65, rf66, rf67, rf68, rf69, rf70, rf71, rf72, rf73, rf74, rf75, rf76, rf77, rf78, rf79, rf80, rf81, rf82, rf83, rf84, rf85, rf86, rf87, rf88, rf89, rf90, rf91, rf92, rf93, rf94, rf95, rf96, rf97, rf98, rf99, rf100, rf101, rf102, rf103, rf104]
gfl = [gf1, gf2, gf3, gf4, gf5, gf6, gf7, gf8, gf9, gf10, gf11, gf12, gf13, gf14, gf15, gf16, gf17, gf18, gf19, gf20, gf21, gf22, gf23, gf24, gf25, gf26, gf27, gf28, gf29, gf30, gf31, gf32, gf33, gf34, gf35, gf36, gf37, gf38, gf39, gf40, gf41, gf42, gf43, gf44, gf45, gf46, gf47, gf48, gf49, gf50, gf51, gf52, gf53, gf54, gf55, gf56, gf57, gf58, gf59, gf60, gf61, gf62, gf63, gf64, gf65, gf66, gf67, gf68, gf69, gf70, gf71, gf72, gf73, gf74, gf75, gf76, gf77, gf78, gf79, gf80, gf81, gf82, gf83, gf84, gf85, gf86, gf87, gf88, gf89, gf90, gf91, gf92, gf93, gf94, gf95, gf96, gf97, gf98, gf99, gf100, gf101, gf102, gf103, gf104]
nrfl_fname = [u'gsl_sf_airy_Ai', u'gsl_sf_airy_Bi', u'gsl_sf_airy_Ai_scaled', u'gsl_sf_airy_Bi_scaled', u'gsl_sf_airy_Ai_deriv', u'gsl_sf_airy_Bi_deriv ', u'gsl_sf_airy_Ai_deriv_scaled', u'gsl_sf_airy_Bi_deriv_scaled', u'gsl_sf_bessel_J0', u'gsl_sf_bessel_J1', u'gsl_sf_bessel_Y0', u'gsl_sf_bessel_Y1 ', u'gsl_sf_bessel_I0 ', u'gsl_sf_bessel_I1', u'gsl_sf_bessel_I0_scaled', u'gsl_sf_bessel_I1_scaled', u'gsl_sf_bessel_K0', u'gsl_sf_bessel_K1', u'gsl_sf_bessel_K0_scaled', u'gsl_sf_bessel_K1_scaled', u'gsl_sf_bessel_j0', u'gsl_sf_bessel_j1', u'gsl_sf_bessel_j2', u'gsl_sf_bessel_y0', u'gsl_sf_bessel_y1', u'gsl_sf_bessel_y2', u'gsl_sf_bessel_i0_scaled', u'gsl_sf_bessel_i1_scaled ', u'gsl_sf_bessel_i2_scaled', u'gsl_sf_bessel_k0_scaled ', u'gsl_sf_bessel_k1_scaled', u'gsl_sf_bessel_k2_scaled ', u'gsl_sf_clausen', u'gsl_sf_dawson', u'gsl_sf_debye_1', u'gsl_sf_debye_2', u'gsl_sf_debye_3', u'gsl_sf_debye_4', u'gsl_sf_debye_5', u'gsl_sf_debye_6', u'gsl_sf_dilog', u'gsl_sf_ellint_Kcomp', u'gsl_sf_ellint_Ecomp', u'gsl_sf_erf ', u'gsl_sf_erfc', u'gsl_sf_log_erfc', u'gsl_sf_erf_Z', u'gsl_sf_erf_Q', u'gsl_sf_hazard ', u'gsl_sf_exp', u'gsl_sf_exp_e10_e ', u'gsl_sf_expm1', u'gsl_sf_exprel', u'gsl_sf_exprel_2', u'gsl_sf_expint_E1', u'gsl_sf_expint_E2', u'gsl_sf_expint_Ei', u'gsl_sf_Shi', u'gsl_sf_Chi', u'gsl_sf_expint_3', u'gsl_sf_Si', u'gsl_sf_Ci', u'gsl_sf_atanint ', u'gsl_sf_fermi_dirac_m1', u'gsl_sf_fermi_dirac_0 ', u'gsl_sf_fermi_dirac_1 ', u'gsl_sf_fermi_dirac_2', u'gsl_sf_fermi_dirac_mhalf', u'gsl_sf_fermi_dirac_half', u'gsl_sf_fermi_dirac_3half', u' gsl_sf_gamma', u'gsl_sf_lngamma', u'gsl_sf_gammastar', u'gsl_sf_gammainv ', u'gsl_sf_lambert_W0', u'gsl_sf_lambert_Wm1', u'gsl_sf_legendre_P1', u'gsl_sf_legendre_P2', u'gsl_sf_legendre_P3', u'gsl_sf_legendre_Q0', u'gsl_sf_legendre_Q1', u'gsl_sf_log ', u'gsl_sf_log_abs', u'gsl_sf_log_1plusx', u'gsl_sf_log_1plusx_mx', u'gsl_sf_psi', u'gsl_sf_psi_1piy', u'gsl_sf_psi_1 ', u'gsl_sf_synchrotron_1', u'gsl_sf_synchrotron_2', u'gsl_sf_transport_2', u'gsl_sf_transport_3', u'gsl_sf_transport_4', u'gsl_sf_transport_5', u'gsl_sf_sin', u'gsl_sf_cos', u'gsl_sf_sinc', u'gsl_sf_lnsinh ', u'gsl_sf_lncosh', u'gsl_sf_angle_restrict_symm', u'gsl_sf_angle_restrict_pos', u'gsl_sf_zeta', u'gsl_sf_zetam1', u'gsl_sf_eta']
ngfl_fname = [u'gsl_sf_airy_Ai', u'gsl_sf_airy_Bi', u'gsl_sf_airy_Ai_scaled', u'gsl_sf_airy_Bi_scaled', u'gsl_sf_airy_Ai_deriv', u'gsl_sf_airy_Bi_deriv ', u'gsl_sf_airy_Ai_deriv_scaled', u'gsl_sf_airy_Bi_deriv_scaled', u'gsl_sf_bessel_J0', u'gsl_sf_bessel_J1', u'gsl_sf_bessel_Y0', u'gsl_sf_bessel_Y1 ', u'gsl_sf_bessel_I0 ', u'gsl_sf_bessel_I1', u'gsl_sf_bessel_I0_scaled', u'gsl_sf_bessel_I1_scaled', u'gsl_sf_bessel_K0', u'gsl_sf_bessel_K1', u'gsl_sf_bessel_K0_scaled', u'gsl_sf_bessel_K1_scaled', u'gsl_sf_bessel_j0', u'gsl_sf_bessel_j1', u'gsl_sf_bessel_j2', u'gsl_sf_bessel_y0', u'gsl_sf_bessel_y1', u'gsl_sf_bessel_y2', u'gsl_sf_bessel_i0_scaled', u'gsl_sf_bessel_i1_scaled ', u'gsl_sf_bessel_i2_scaled', u'gsl_sf_bessel_k0_scaled ', u'gsl_sf_bessel_k1_scaled', u'gsl_sf_bessel_k2_scaled ', u'gsl_sf_clausen', u'gsl_sf_dawson', u'gsl_sf_debye_1', u'gsl_sf_debye_2', u'gsl_sf_debye_3', u'gsl_sf_debye_4', u'gsl_sf_debye_5', u'gsl_sf_debye_6', u'gsl_sf_dilog', u'gsl_sf_ellint_Kcomp', u'gsl_sf_ellint_Ecomp', u'gsl_sf_erf ', u'gsl_sf_erfc', u'gsl_sf_log_erfc', u'gsl_sf_erf_Z', u'gsl_sf_erf_Q', u'gsl_sf_hazard ', u'gsl_sf_exp', u'gsl_sf_exp_e10_e ', u'gsl_sf_expm1', u'gsl_sf_exprel', u'gsl_sf_exprel_2', u'gsl_sf_expint_E1', u'gsl_sf_expint_E2', u'gsl_sf_expint_Ei', u'gsl_sf_Shi', u'gsl_sf_Chi', u'gsl_sf_expint_3', u'gsl_sf_Si', u'gsl_sf_Ci', u'gsl_sf_atanint ', u'gsl_sf_fermi_dirac_m1', u'gsl_sf_fermi_dirac_0 ', u'gsl_sf_fermi_dirac_1 ', u'gsl_sf_fermi_dirac_2', u'gsl_sf_fermi_dirac_mhalf', u'gsl_sf_fermi_dirac_half', u'gsl_sf_fermi_dirac_3half', u' gsl_sf_gamma', u'gsl_sf_lngamma', u'gsl_sf_gammastar', u'gsl_sf_gammainv ', u'gsl_sf_lambert_W0', u'gsl_sf_lambert_Wm1', u'gsl_sf_legendre_P1', u'gsl_sf_legendre_P2', u'gsl_sf_legendre_P3', u'gsl_sf_legendre_Q0', u'gsl_sf_legendre_Q1', u'gsl_sf_log ', u'gsl_sf_log_abs', u'gsl_sf_log_1plusx', u'gsl_sf_log_1plusx_mx', u'gsl_sf_psi', u'gsl_sf_psi_1piy', u'gsl_sf_psi_1 ', u'gsl_sf_synchrotron_1', u'gsl_sf_synchrotron_2', u'gsl_sf_transport_2', u'gsl_sf_transport_3', u'gsl_sf_transport_4', u'gsl_sf_transport_5', u'gsl_sf_sin', u'gsl_sf_cos', u'gsl_sf_sinc', u'gsl_sf_lnsinh ', u'gsl_sf_lncosh', u'gsl_sf_angle_restrict_symm', u'gsl_sf_angle_restrict_pos', u'gsl_sf_zeta', u'gsl_sf_zetam1', u'gsl_sf_eta']
