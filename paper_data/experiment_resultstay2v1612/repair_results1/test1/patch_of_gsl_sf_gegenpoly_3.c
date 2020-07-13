#include "eft_patch.h"
static double array_idx_sf_gegenpoly_3_0[2] = {
-7.951688670776461367e+04,
-7.937136755548094516e+04,
};
static double array_cof_float_sf_gegenpoly_3_0[1][22] = {
{
0.000000000000000000e+00,
0.000000000000000000e+00,
6.337768826005307508e-10,
-1.510495537032003927e-04,
7.999999948997932542e+00,
1.275080140766108889e-03,
-5.838743601353404138e-47,
2.666666664765335870e+00,
-6.355530165998336161e+05,
5.049095465564353943e+10,
-1.337069940537758750e+15,
-3.999999999049334676e+00,
9.533295253528989851e+05,
-7.573643201246531677e+10,
2.005604911203858750e+15,
-3.177765085264911177e+05,
2.524547734282176971e+10,
-6.685349705072117500e+14,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
5.000000000000000000e+00,
}
};
static double array_cof_err_sf_gegenpoly_3_0[1][22] = {
{
0.000000000000000000e+00,
0.000000000000000000e+00,
4.868079332834772675e-27,
1.389471886272007431e-21,
-2.742668935503070945e-16,
-3.967165877886996303e-20,
-3.241153790552677793e-63,
1.482556777563654777e-16,
4.194548453050154402e-11,
1.721844882692333331e-06,
6.251269523231388348e-02,
0.000000000000000000e+00,
3.720799410327937896e-11,
5.196068415469511621e-06,
-8.699281295703029104e-02,
0.000000000000000000e+00,
7.870336693715792675e-07,
2.679298068373254030e-02,
0.000000000000000000e+00,
0.000000000000000000e+00,
0.000000000000000000e+00,
}
};
static double array_pointx_sf_gegenpoly_3_0[1] = {
-1.999999999762333669e+00,
};
static double array_pointy_sf_gegenpoly_3_0[1] = {
-7.944412713162277942e+04,
};
static double array_cofidx_sf_gegenpoly_3_0[1] = {
5.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_gegenpoly_3_0(double x,double y)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((y>=array_idx_sf_gegenpoly_3_0[idx])&&(y<=array_idx_sf_gegenpoly_3_0[idx+1])){
         double pointx = array_pointx_sf_gegenpoly_3_0[idx];
         double pointy = array_pointy_sf_gegenpoly_3_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_gegenpoly_3_0[idx];
         eft_tay2v(array_cof_float_sf_gegenpoly_3_0[idx],array_cof_err_sf_gegenpoly_3_0[idx],pointx,pointy,x,y,&res,length);
         return res;
     }
     else if(y<array_idx_sf_gegenpoly_3_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(y>array_idx_sf_gegenpoly_3_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_gegenpoly_3(double x,double y)
{
if((x<=-1.9988897767377083)&&(y<=-79371.36755548095)){
 return accuracy_improve_patch_of_gsl_sf_gegenpoly_3_0(x,y);
}
}
