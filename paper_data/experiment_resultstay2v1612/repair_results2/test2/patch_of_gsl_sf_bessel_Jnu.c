#include "eft_patch.h"
static double array_idx_sf_bessel_Jnu_0[10] = {
9.251579109047924021e+01,
9.253158092905168530e+01,
9.254737076762413039e+01,
9.256316060619657549e+01,
9.257895044476902058e+01,
9.259474028334147988e+01,
9.261053012191392497e+01,
9.262631996048637006e+01,
9.264210979905881516e+01,
9.265789963763126025e+01,
};
static double array_cof_float_sf_bessel_Jnu_0[9][37] = {
{
-1.285907046809922511e-06,
2.398244830219209544e-06,
1.176748228060497465e-04,
-3.918767805564193045e-05,
-5.148011637619587418e-03,
-3.680072125760581038e-04,
6.809876757141876402e-02,
3.252845152751968956e-16,
9.834876749978635501e-06,
-1.824996806298081518e-05,
-6.453629663246533438e-04,
2.453066242049959452e-04,
1.695250579953235665e-02,
2.904452340983749233e-04,
-7.471863843635365154e-02,
-3.219027399062721246e-05,
5.706667445385545486e-05,
1.414727274555330466e-03,
-5.310880965749498412e-04,
-1.860348850293237072e-02,
1.533508136506087625e-04,
5.844087485952475912e-05,
-9.409456989120848068e-05,
-1.549381071441092040e-03,
4.864830974475394692e-04,
6.802886864958410627e-03,
-6.354758971852808580e-05,
8.644375140111197099e-05,
8.476506911218092272e-04,
-1.616377453933215674e-04,
4.138052124411036080e-05,
-4.201136793529869183e-05,
-1.853073029581804403e-04,
-1.493827917575157586e-05,
8.447239729805135103e-06,
2.305776546609118606e-06,
7.000000000000000000e+00,
},
{
-1.285393044266868929e-06,
2.397647809166424424e-06,
1.176394047516622973e-04,
-3.918469974613329331e-05,
-5.146956713298173405e-03,
-3.679042170316704739e-04,
6.809132681302150492e-02,
-3.248801925235736591e-16,
9.830831918411685880e-06,
-1.824471650398965067e-05,
-6.451613523023591590e-04,
2.452643214210805897e-04,
1.694884087941409337e-02,
2.903579831672707797e-04,
-7.470963998155646935e-02,
-3.217667214292635052e-05,
5.704833304191896689e-05,
1.414269363452267747e-03,
-5.309641434215536934e-04,
-1.859925814474430728e-02,
1.533037639334093485e-04,
5.841554173973571493e-05,
-9.406151861290874090e-05,
-1.548862402129630618e-03,
4.863480029915472221e-04,
6.801264239937277085e-03,
-6.351937255768753986e-05,
8.641105667087846279e-05,
8.473577255180619406e-04,
-1.615871858104926532e-04,
4.136172906866640646e-05,
-4.199444110305337359e-05,
-1.852412928855578990e-04,
-1.493135213523721031e-05,
8.443643218656623120e-06,
2.304686622487891508e-06,
7.000000000000000000e+00,
},
{
-1.284879324893010674e-06,
2.397050971473000154e-06,
1.176040027735384731e-04,
-3.918171994888434599e-05,
-5.145902170125728550e-03,
-3.678012676669637458e-04,
6.808388809778784245e-02,
2.070068803471698993e-16,
9.826789380369791395e-06,
-1.823946700526441836e-05,
-6.449598332048713792e-04,
2.452220275507221384e-04,
1.694517735197416386e-02,
2.902707739341151699e-04,
-7.470064420869780564e-02,
-3.216307821813444698e-05,
5.702999994923504658e-05,
1.413811675404585402e-03,
-5.308402357375622627e-04,
-1.859502946956942404e-02,
1.532567371150335867e-04,
5.839022374995540750e-05,
-9.402848390711438626e-05,
-1.548343993581443668e-03,
4.862129685272246473e-04,
6.799642288106860256e-03,
-6.349117264393966398e-05,
8.637837962470817986e-05,
8.470649115876163933e-04,
-1.615366512385833168e-04,
4.134294862645044807e-05,
-4.197752399867406292e-05,
-1.851753179282964704e-04,
-1.492442950430388416e-05,
8.440048881476253009e-06,
2.303597404355749775e-06,
7.000000000000000000e+00,
},
{
-1.284365888489820802e-06,
2.396454317094386613e-06,
1.175686168619318112e-04,
-3.917873866625592517e-05,
-5.144848007904975762e-03,
-3.676983644529900467e-04,
6.807645142481838496e-02,
-1.068418715037603174e-16,
9.822749134196419877e-06,
-1.823421956585245013e-05,
-6.447584089724669445e-04,
2.451797425939857521e-04,
1.694151521645494152e-02,
2.901836063705959972e-04,
-7.469165111650495625e-02,
-3.214949221037082991e-05,
5.701167517101837789e-05,
1.413354210267151476e-03,
-5.307163735009494399e-04,
-1.859080247645138531e-02,
1.532097331807933294e-04,
5.836492087867018575e-05,
-9.399546576307693459e-05,
-1.547825845621609046e-03,
4.860779940180758193e-04,
6.798021009069593612e-03,
-6.346298996385837607e-05,
8.634572025017346836e-05,
8.467722492258687597e-04,
-1.614861416607313684e-04,
4.132417990813942439e-05,
-4.196061661490382199e-05,
-1.851093780615613651e-04,
-1.491751127938300019e-05,
8.436456716563450273e-06,
2.302508891632080274e-06,
7.000000000000000000e+00,
},
{
-1.283852734858954052e-06,
2.395857845985616940e-06,
1.175332470071036825e-04,
-3.917575590058061728e-05,
-5.143794226438768921e-03,
-3.675955073614917612e-04,
6.806901679321425425e-02,
-3.454408644724671082e-16,
9.818711178236597695e-06,
-1.822897418479892299e-05,
-6.445570795454731022e-04,
2.451374665508208525e-04,
1.693785447209936179e-02,
2.900964804498911841e-04,
-7.468266070370603582e-02,
-3.213591411376051269e-05,
5.699335870247985688e-05,
1.412896967894960651e-03,
-5.305925566895180285e-04,
-1.858657716443461172e-02,
1.531627521152075790e-04,
5.833963311437783013e-05,
-9.396246417004615542e-05,
-1.547307958075360953e-03,
4.859430794274981267e-04,
6.796400402428231548e-03,
-6.343482450403124145e-05,
8.631307853484899192e-05,
8.464797383283112747e-04,
-1.614356570600520668e-04,
4.130542290441989645e-05,
-4.194371894448872161e-05,
-1.850434732605411316e-04,
-1.491059745690971688e-05,
8.432866722218611117e-06,
2.301421083736891170e-06,
7.000000000000000000e+00,
},
{
-1.283339863802236057e-06,
2.395261558101700555e-06,
1.174978931993230337e-04,
-3.917277165418812208e-05,
-5.142740825530104157e-03,
-3.674926963642382704e-04,
6.806158420207714110e-02,
4.294136548898461678e-16,
9.814675510836821668e-06,
-1.822373086114946800e-05,
-6.443558448642651296e-04,
2.450951994211724707e-04,
1.693419511815091524e-02,
2.900093961452103659e-04,
-7.467367296903000551e-02,
-3.212234392243388224e-05,
5.697505053883371350e-05,
1.412439948143127955e-03,
-5.304687852810833007e-04,
-1.858235353256423514e-02,
1.531157939028039575e-04,
5.831436044558687505e-05,
-9.392947911728043772e-05,
-1.546790330768083206e-03,
4.858082247189176358e-04,
6.794780467785839842e-03,
-6.340667625105864613e-05,
8.628045446632029685e-05,
8.461873787905278962e-04,
-1.613851974196751720e-04,
4.128667760598753937e-05,
-4.192683098018154791e-05,
-1.849776035004466526e-04,
-1.490368803332273504e-05,
8.429278896743773295e-06,
2.300333980090775929e-06,
7.000000000000000000e+00,
},
{
-1.282827275121646817e-06,
2.394665453397972142e-06,
1.174625554288655471e-04,
-3.916978592942838678e-05,
-5.141687804982098160e-03,
-3.673899314324157090e-04,
6.805415365050929144e-02,
4.860710478708500557e-16,
9.810642130344920153e-06,
-1.821848959395244025e-05,
-6.441547048692609131e-04,
2.450529412050828906e-04,
1.693053715385358163e-02,
2.899223534284568228e-04,
-7.466468791120667303e-02,
-3.210878163052617731e-05,
5.695675067530368482e-05,
1.411983150866876402e-03,
-5.303450592536393133e-04,
-1.857813157988605010e-02,
1.530688585288516267e-04,
5.828910286081561950e-05,
-9.389651059405578870e-05,
-1.546272963525293634e-03,
4.856734298559093818e-04,
6.793161204745767033e-03,
-6.337854519155260418e-05,
8.624784803219106275e-05,
8.458951705081851988e-04,
-1.613347627227779862e-04,
4.126794400354628174e-05,
-4.190995271474493288e-05,
-1.849117687565088417e-04,
-1.489678300506397080e-05,
8.425693238443183623e-06,
2.299247580114858258e-06,
7.000000000000000000e+00,
},
{
-1.282314968619342306e-06,
2.394069531829467802e-06,
1.174272336860145757e-04,
-3.916679872863071811e-05,
-5.140635164598004667e-03,
-3.672872125377018432e-04,
6.804672513761345076e-02,
1.647321617684901175e-16,
9.806611035110237692e-06,
-1.821325038225481924e-05,
-6.439536595009275619e-04,
2.450106919025117798e-04,
1.692688057845188887e-02,
2.898353522725873543e-04,
-7.465570552896659551e-02,
-3.209522723217819317e-05,
5.693845910711180030e-05,
1.411526575921551521e-03,
-5.302213785850635715e-04,
-1.857391130544656585e-02,
1.530219459780668054e-04,
5.826386034859350264e-05,
-9.386355858964948954e-05,
-1.545755856172663367e-03,
4.855386948019823181e-04,
6.791542612911672175e-03,
-6.335043131213845182e-05,
8.621525922006972615e-05,
8.456031133770435405e-04,
-1.612843529525261564e-04,
4.124922208780945084e-05,
-4.189308414094558108e-05,
-1.848459690039814618e-04,
-1.488988236857901133e-05,
8.422109745622249351e-06,
2.298161883230868764e-06,
7.000000000000000000e+00,
},
{
-1.281802944097651717e-06,
2.393473793351176630e-06,
1.173919279610611161e-04,
-3.916381005411994368e-05,
-5.139582904181214454e-03,
-3.671845396518433400e-04,
6.803929866249294744e-02,
7.715166110440757855e-18,
9.802582223483606220e-06,
-1.820801322510387927e-05,
-6.437527086997806491e-04,
2.449684515134078013e-04,
1.692322539119093389e-02,
2.897483926506825755e-04,
-7.464672582104125986e-02,
-3.208168072153612578e-05,
5.692017582948301672e-05,
1.411070223162621358e-03,
-5.300977432532349900e-04,
-1.856969270829304103e-02,
1.529750562351236726e-04,
5.823863289746081923e-05,
-9.383062309334685807e-05,
-1.545239008536014027e-03,
4.854040195206652387e-04,
6.789924691887533510e-03,
-6.332233459945444082e-05,
8.618268801757515902e-05,
8.453112072929558699e-04,
-1.612339680920976627e-04,
4.123051184949953546e-05,
-4.187622525155669550e-05,
-1.847802042181407461e-04,
-1.488298612031700134e-05,
8.418528416587985400e-06,
2.297076888861124624e-06,
7.000000000000000000e+00,
}
};
static double array_cof_err_sf_bessel_Jnu_0[9][37] = {
{
-6.644821815430767891e-24,
1.470310129066605376e-23,
-1.571251010186921238e-21,
1.158769715526422544e-21,
-2.196260824472832757e-19,
1.857915459257994236e-20,
3.313579513874647983e-18,
-1.292848308943520856e-33,
1.023634480192946803e-22,
-1.309017789436441947e-21,
1.979048283062331372e-20,
-1.882503138938371070e-20,
1.012621931980793116e-18,
2.360721136696699281e-20,
4.229087001006664598e-18,
2.327058280718688910e-21,
-2.211293635117996086e-21,
-3.718638427608072814e-20,
1.569317921669965320e-20,
1.249799992298721614e-18,
-4.536231685236620936e-21,
1.464677705254761220e-21,
6.277986991518641887e-21,
-1.518854170646441172e-20,
6.399774903418187559e-21,
-9.024975219385036875e-20,
1.493592030497345685e-21,
3.535321999997657773e-21,
3.126862423263664750e-20,
-3.578580894333887433e-21,
-3.283241011819996701e-21,
-4.770682886792929589e-22,
-6.178738209187199338e-23,
-5.495632714387992860e-22,
8.110144480629063463e-22,
-2.272118061291666474e-23,
},
{
-1.054545306501095581e-22,
-6.918752124596180365e-23,
4.488022988606598312e-21,
-2.662818591350609059e-21,
4.226706826879634236e-20,
2.340886287087423551e-20,
3.296701049834421311e-18,
1.957881590887386511e-32,
3.871699851405756030e-23,
1.260675166370701364e-21,
-1.988878085362098939e-20,
-4.740914287246875299e-21,
7.347178084256710302e-19,
-1.627104541491386674e-20,
-6.833564676831634423e-18,
-9.008680043546527953e-22,
2.615542427344322849e-23,
5.008985883928632328e-20,
-1.410060591022260865e-21,
-1.653133019744582136e-18,
-1.327970998214191302e-20,
2.779073162253511638e-23,
1.942208206876962029e-21,
-5.453500561719164850e-20,
-1.132979703160918791e-20,
2.459515797304906907e-19,
-2.420489733440308100e-21,
-1.303818150800769651e-21,
3.343983623238545826e-21,
1.239379585051841683e-20,
-1.545075840209743192e-22,
2.644073215118301029e-21,
-1.822508978925189786e-21,
-4.143073525466728398e-22,
-6.190657090589065938e-22,
2.609957242855871900e-23,
},
{
2.699581228064049716e-24,
-3.446970140471511725e-24,
5.162510474086751073e-21,
-5.962784328320575045e-22,
-3.454344237968301516e-19,
-2.689893643015535633e-20,
-5.760605012665583090e-18,
-2.646779235678453453e-33,
-1.098111307860229523e-22,
1.816972778385252562e-22,
1.630427704300003263e-20,
-9.652941629157638148e-21,
-1.674323994611974528e-18,
2.014768226046135328e-20,
5.231058417923572318e-19,
-7.727739518430389431e-22,
-1.176084980373293282e-21,
-9.610873585008111698e-20,
4.345372428839940139e-20,
1.433617212391692167e-18,
-3.343111394460903828e-21,
-2.484134444520665128e-21,
-5.918115523660282364e-21,
-8.211045451158297108e-20,
2.103635335016583694e-20,
-4.021276141534267977e-19,
1.144346027836617027e-21,
5.940564184567501437e-21,
2.272242870489962035e-20,
9.253318759695215589e-22,
2.743539888478824126e-21,
1.454843436177086418e-21,
-9.758428559911680561e-21,
-8.282260993737607921e-22,
1.338067042363326562e-23,
3.338293034483242491e-23,
},
{
-4.854561003183968249e-23,
1.874849860120427429e-22,
-1.944620362804307030e-21,
-7.163755604487066958e-22,
3.774099563170448985e-19,
2.385434688876782232e-20,
-3.662822857311397032e-18,
-2.148475338273947647e-33,
-1.003204262444267970e-22,
1.174001139319361740e-21,
-2.588895521472913615e-20,
3.536262718009545279e-21,
-9.913956281073363905e-20,
1.692412306341383118e-21,
-1.792190778933892546e-18,
-1.131620877814376333e-22,
-2.733008573700356567e-21,
-3.514224718249197902e-20,
1.388509689570604960e-22,
-1.630679843719484493e-18,
-5.963666183804056694e-21,
2.259259098376658229e-21,
-5.778639983213991632e-21,
1.361785695103319271e-20,
-2.384397667090562913e-20,
-3.107215943244362082e-19,
-6.655634031611036152e-21,
2.760666954374829194e-21,
-2.244062434320618244e-20,
8.153288997296111003e-21,
-2.001598190370713498e-21,
-1.988733617402864673e-21,
1.196158126226539865e-20,
2.136484011603873993e-22,
8.465072346350353048e-22,
-8.354299072354030385e-23,
},
{
8.641447164119552816e-26,
1.805305985958723113e-22,
-5.215809496140802644e-21,
-2.991252730333856497e-21,
6.066621694665963108e-20,
3.315645542956288085e-21,
-3.899811547828887452e-18,
-4.293197191812311683e-33,
-6.878207182963651209e-22,
2.939930753827631400e-22,
-4.674842769761144631e-20,
-1.681941280763563809e-20,
1.365929406680083201e-18,
2.557509024656805375e-20,
1.653332653029977821e-18,
-1.236643804040600197e-22,
6.462007099801344130e-22,
1.729999745277097734e-20,
-1.790373302130739111e-20,
-1.265887925935451031e-20,
1.287641695647993025e-20,
2.093259702217778003e-22,
3.572428402400183606e-21,
6.240212260730390008e-20,
2.519019122263453913e-20,
-3.919884026525790414e-19,
-5.908960967767995094e-21,
2.278503084493606578e-21,
-2.067664431722649182e-20,
1.087465983827182250e-20,
9.949226872255136640e-22,
-7.679551389953199464e-22,
6.624457574279219195e-21,
1.136769534247130314e-22,
-3.676739972623944254e-22,
1.906624398953835932e-22,
},
{
-7.459571612290525749e-23,
-1.971591582167330639e-22,
1.684465537865444605e-21,
2.632999412581013965e-21,
3.698799945688102372e-19,
8.652255728841581945e-22,
-4.471200217162102520e-18,
-2.142609159961985849e-32,
1.534503849046213235e-23,
7.705350369505423476e-22,
1.460640243464547471e-20,
-5.608326103711602779e-21,
-6.366963710061362770e-19,
-5.724351432071261559e-21,
-3.035509995017135707e-18,
6.467303233630859665e-23,
1.839635247221427648e-21,
5.183539393507479093e-20,
1.947356321166229818e-23,
-3.724620602901281092e-19,
-9.607324458791291509e-21,
-2.879246607605424482e-21,
-5.864066860266493140e-22,
7.445806192796153178e-20,
-2.078338193763684132e-20,
1.243819219874392121e-19,
5.640766416333018855e-21,
6.427159636820429277e-21,
5.318631190158991766e-20,
-4.970465190213105598e-21,
1.587539953829636233e-21,
1.073788858154047215e-21,
2.793959468736062744e-21,
2.935016063860154352e-22,
-4.294855818836479588e-22,
1.951858695798749007e-22,
},
{
8.393182785252006688e-23,
-1.260153285128445268e-23,
6.503571226614317368e-21,
7.737640922840635643e-22,
-2.733148176114602141e-19,
8.805339028137284778e-21,
-4.168902131083005955e-18,
-5.103922789618612007e-33,
-3.683193182687779389e-22,
3.564024639290367180e-22,
-8.401398785987015522e-21,
-1.518496699208293637e-20,
-1.234884836707795660e-18,
-1.406989854730116883e-20,
1.203362317607858080e-18,
-2.263553509093784112e-21,
-7.100463768227838781e-23,
-9.979877261704232409e-20,
1.046530812156702247e-20,
4.807034371928937428e-19,
9.593630578595330502e-21,
-2.795339748531492117e-21,
-3.358416396541436943e-21,
-8.584450724471099005e-21,
1.198636554288777632e-20,
4.269944682673019268e-20,
2.855467541181522589e-21,
-6.225887793181843005e-21,
2.929015061211693474e-20,
1.086823976929726047e-20,
-2.895268016207694811e-21,
1.073258273947635895e-21,
9.246086593709847249e-21,
-2.376263358130415239e-22,
-2.411689624260777425e-22,
1.311670784438305931e-22,
},
{
1.039410435204225161e-22,
1.079018751904711185e-22,
3.247758905063610990e-21,
-2.240250577444950114e-21,
1.761178923086930552e-20,
-1.263401747668689986e-20,
-6.011715187357278913e-18,
-1.519972199587397563e-33,
-1.212101974607020894e-22,
5.224973641513299231e-22,
-3.410107895457138024e-21,
2.589579693554861214e-20,
8.591251542583383685e-19,
1.593245797156119695e-20,
-5.023304077037241347e-18,
-1.615180331345086755e-21,
2.151700691113730624e-21,
2.352165014022488704e-20,
-3.438264701695900906e-20,
-1.501349446441638109e-18,
5.626332819915430733e-21,
2.409767335764968874e-21,
-1.457803315322885062e-21,
4.637971010907438746e-21,
-1.031195333348126944e-20,
4.189669178870295279e-19,
-1.705948135732296046e-21,
-2.947376130483853669e-21,
2.053160090310019731e-20,
3.323484591630220059e-21,
1.603716831511357616e-22,
3.075915474123887318e-21,
-9.606737277954409581e-23,
-7.430350391591705482e-22,
-1.532057208474672439e-22,
-8.669166297776191289e-23,
},
{
3.686998781116731767e-23,
-6.885319409605804029e-23,
1.303253327013206284e-21,
2.935097940894797388e-22,
-4.250093312577982453e-19,
-2.519193411527556869e-20,
-5.037939945423788167e-20,
3.812552095291021053e-34,
-6.322902807495967404e-22,
1.004562945684848707e-21,
2.685530736645722164e-20,
8.733904537381397611e-21,
1.578021015937251861e-18,
4.152032229125354073e-21,
-1.999294325687970932e-18,
-9.235623925599787480e-22,
8.523576653991309362e-22,
9.214263330886289417e-20,
-2.834474442940173569e-20,
1.045937293891944465e-18,
-8.096263792955034804e-21,
2.897283953688987973e-21,
-3.619717625518797609e-21,
1.288528839991563118e-20,
1.781845669235986450e-20,
-1.699357989930358080e-19,
3.506090699324367418e-21,
-2.358922017677635630e-21,
-4.916138718829910204e-20,
7.602210265528998653e-21,
2.806556459791457963e-21,
-3.227392954133132775e-21,
-4.598698009649877964e-21,
-7.441284877557483948e-22,
-3.347670175855618014e-22,
1.343865157569808853e-22,
}
};
static double array_pointx_sf_bessel_Jnu_0[9] = {
6.837925169175099427e+01,
6.839364267336144110e+01,
6.840803381567108943e+01,
6.842242511863383925e+01,
6.843681658220367581e+01,
6.845120820633452752e+01,
6.846559999098037963e+01,
6.847999193609528845e+01,
6.849438404163322502e+01,
};
static double array_pointy_sf_bessel_Jnu_0[9] = {
9.252368600976546986e+01,
9.253947584833790074e+01,
9.255526568691036005e+01,
9.257105552548279093e+01,
9.258684536405525023e+01,
9.260263520262770953e+01,
9.261842504120014041e+01,
9.263421487977259972e+01,
9.265000471834503060e+01,
};
static double array_cofidx_sf_bessel_Jnu_0[9] = {
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
7.000000000000000000e+00,
};
double accuracy_improve_patch_of_gsl_sf_bessel_Jnu_0(double x,double y)
{
 int len_glob = 9;
 int idx = floor(len_glob/2);
 int dw_idx = 0;
 int up_idx = len_glob;
 while((idx>=0)&&(idx<len_glob)){
     if((y>=array_idx_sf_bessel_Jnu_0[idx])&&(y<=array_idx_sf_bessel_Jnu_0[idx+1])){
         double pointx = array_pointx_sf_bessel_Jnu_0[idx];
         double pointy = array_pointy_sf_bessel_Jnu_0[idx];
         double res = 0.0;
         int length = (int)array_cofidx_sf_bessel_Jnu_0[idx];
         eft_tay2v(array_cof_float_sf_bessel_Jnu_0[idx],array_cof_err_sf_bessel_Jnu_0[idx],pointx,pointy,x,y,&res,length);
         return res;
     }
     else if(y<array_idx_sf_bessel_Jnu_0[idx]){
         up_idx = idx;
         idx = dw_idx + floor((idx-dw_idx)/2.0);
     }
     else if(y>array_idx_sf_bessel_Jnu_0[idx+1]){
         dw_idx = idx;
         idx = idx + floor((up_idx-idx)/2.0);
     }
 }
}
double accuracy_improve_patch_of_gsl_sf_bessel_Jnu(double x,double y)
{
if((x<=68.5078708557797)&&(y<=92.65789963763126)){
 return accuracy_improve_patch_of_gsl_sf_bessel_Jnu_0(x,y);
}
}
