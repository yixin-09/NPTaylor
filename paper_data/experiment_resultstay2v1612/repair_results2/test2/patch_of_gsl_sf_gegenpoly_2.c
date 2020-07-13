#include "eft_patch.h"
static double array_idx_sf_gegenpoly_2_0[2] = {
6.000700461887122365e+05,
6.012341994069815846e+05,
};
static double array_cof_float_sf_gegenpoly_2_0[1][16] = {
{
-3.471343646883573208e-71,
-6.182010251145701966e-69,
7.215671465473651123e+11,
-1.316678910463534355e+09,
2.484533313396849264e-11,
0.000000000000000000e+00,
2.402610491191387642e+06,
-4.384161029153060554e+03,
9.999983351455906666e-01,
2.000000000000000000e+00,
-3.649497948357893762e-03,
1.664854409383559429e-06,
1.957531539154740884e-76,
-1.545502562786425687e-67,
8.527431132561820277e-71,
4.000000000000000000e+00,
}
};
static double array_cof_err_sf_gegenpoly_2_0[1][16] = {
{
1.340509788923965790e-87,
-3.431705059645352422e-85,
1.205921890118369301e-05,
1.184442498339133979e-08,
5.649833497539554624e-28,
0.000000000000000000e+00,
0.000000000000000000e+00,
2.625483916238578562e-13,
3.256229036248191252e-17,
0.000000000000000000e+00,
0.000000000000000000e+00,
4.590761071763191962e-23,
-1.022727805270359642e-92,
1.098145619086512775e-83,
5.362039155695863159e-87,
}
};
static double array_pointx_sf_gegenpoly_2_0[1] = {
6.006521227978469105e+05,
};
static double array_pointy_sf_gegenpoly_2_0[1] = {
-9.123744870894734404e-04,
};
static double array_cofidx_sf_gegenpoly_2_0[1] = {
4.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_gegenpoly_2_0(double x,double y)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((x>=array_idx_sf_gegenpoly_2_0[idx])&&(x<=array_idx_sf_gegenpoly_2_0[idx+1])){
         double pointx = array_pointx_sf_gegenpoly_2_0[idx];
         double pointy = array_pointy_sf_gegenpoly_2_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_gegenpoly_2_0[idx];
         eft_tay2v(array_cof_float_sf_gegenpoly_2_0[idx],array_cof_err_sf_gegenpoly_2_0[idx],pointx,pointy,x,y,&res,length);
         return res;
     }
     else if(x<array_idx_sf_gegenpoly_2_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(x>array_idx_sf_gegenpoly_2_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_gegenpoly_2(double x,double y)
{
if((x<=601234.1994069816)&&(y<=-0.0009118323860032307)){
 return accuracy_improve_patch_of_gsl_sf_gegenpoly_2_0(x,y);
}
}
