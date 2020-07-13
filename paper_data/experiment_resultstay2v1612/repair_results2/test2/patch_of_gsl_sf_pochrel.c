#include "eft_patch.h"
static double array_idx_sf_pochrel_0[2] = {
1.701404597999757184e-01,
1.704180155561320076e-01,
};
static double array_cof_float_sf_pochrel_0[1][37] = {
{
2.462318869058618617e-02,
-3.726017529047320753e-02,
5.966394945955769136e-02,
-8.033679133855148313e-02,
1.576691050645999403e-01,
-1.182705304334415314e-01,
4.717023875855974602e-01,
5.867609421253845954e-17,
1.968749783348792970e-01,
-2.604840931311322438e-01,
3.583335114182505388e-01,
-3.997206953542339347e-01,
6.362868974685121914e-01,
-3.444097726651733637e-01,
9.685682209209579563e-01,
6.606465620512909087e-01,
-7.433210788622722776e-01,
8.217410928890740429e-01,
-7.453911704205820055e-01,
7.326577447602469206e-01,
-3.644148049210661489e-01,
1.197826675777960537e+00,
-1.092673445526594955e+00,
8.816688965784498233e-01,
-5.784443861881265470e-01,
1.915038017890187128e-01,
1.255556043989900505e+00,
-8.485573085687112549e-01,
4.316426528088244119e-01,
-1.120861690540034677e-01,
7.490538617697103385e-01,
-3.164112059552158240e-01,
6.875182325779807813e-02,
2.292625317200369106e-01,
-4.316140007842171611e-02,
2.743234177671707505e-02,
7.000000000000000000e+00,
}
};
static double array_cof_err_sf_pochrel_0[1][37] = {
{
7.139447186986675058e-19,
-8.956604770091598702e-19,
-1.910059820906014006e-18,
2.494404860788308449e-18,
5.915899746872489271e-18,
3.771396525127108004e-18,
1.878420278172644831e-17,
1.070042811323566421e-33,
7.734132498202369202e-18,
1.565117130205985329e-17,
2.356678082581252493e-17,
-2.046369243363293615e-17,
-1.636687006006828928e-17,
2.694146811781146557e-17,
4.980801309640663586e-17,
2.301946097989508370e-17,
-5.759803962223826630e-18,
-2.645330153890735565e-17,
-5.346893894960444848e-17,
5.273727136429583130e-17,
2.308831401337071467e-17,
-6.965906952386404503e-17,
2.587406772814793821e-17,
-1.539899556611527155e-17,
-5.485848545438839358e-17,
8.267505769756130822e-18,
-6.548852042125802236e-17,
2.369732875260863559e-17,
2.092671479947717095e-18,
4.123209876248719183e-18,
4.552048209934880355e-17,
-2.600639066699696925e-17,
-7.860494028541198177e-19,
-1.451996254211464865e-18,
1.928655906243190677e-18,
3.169369147434348950e-19,
}
};
static double array_pointx_sf_pochrel_0[1] = {
1.377598292745821684e+00,
};
static double array_pointy_sf_pochrel_0[1] = {
1.702792376780538630e-01,
};
static double array_cofidx_sf_pochrel_0[1] = {
7.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_pochrel_0(double x,double y)
{
 int len_glob = 1;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((y>=array_idx_sf_pochrel_0[idx])&&(y<=array_idx_sf_pochrel_0[idx+1])){
         double pointx = array_pointx_sf_pochrel_0[idx];
         double pointy = array_pointy_sf_pochrel_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_pochrel_0[idx];
         eft_tay2v(array_cof_float_sf_pochrel_0[idx],array_cof_err_sf_pochrel_0[idx],pointx,pointy,x,y,&res,length);
         return res;
     }
     else if(y<array_idx_sf_pochrel_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(y>array_idx_sf_pochrel_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_pochrel(double x,double y)
{
if((x<=1.3787085157704415)&&(y<=0.170418015556132)){
 return accuracy_improve_patch_of_gsl_sf_pochrel_0(x,y);
}
}
