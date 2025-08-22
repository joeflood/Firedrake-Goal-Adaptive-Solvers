***********************************************
* written by GAMS/JAMS at 08/30/17 06:39:10
* for more information use JAMS option "Dict"
***********************************************

Variables  x0,x11,x12,y1,y2,pi1,pi2,thetaP,u4,u5;

Positive Variables  x0,x11,x12,y1,y2,pi1,pi2;

Positive Variables  u4,u5;

Equations  e4,e5,e6,e7,dL_dx0,dL_dx11,dL_dx12,dL_dy1,dL_dy2,dL_dthetaP;


e4.. (0.75*(pi1*(x0 + x11) - 5.75*sqr(x0) - 0.5*sqr(x11)) + 0.25*(pi2*(x0 + x12)
      - 5.75*sqr(x0) - 1.75*sqr(x12))) - thetaP =G= 0;

e5.. (0.25*(pi1*(x0 + x11) - 5.75*sqr(x0) - 0.5*sqr(x11)) + 0.75*(pi2*(x0 + x12)
      - 5.75*sqr(x0) - 1.75*sqr(x12))) - thetaP =G= 0;

e6..    x0 + x11 - y1 =G= 0;

e7..    x0 + x12 - y2 =G= 0;

dL_dx0..  + (-(0.75*(pi1 - 11.5*x0) + 0.25*(pi2 - 11.5*x0)))*u4 + (-(0.25*(pi1 - 11.5*x0)
          + 0.75*(pi2 - 11.5*x0)))*u5 =G= 0;

dL_dx11..  + (-0.75*(pi1 - x11))*u4 + (-0.25*(pi1 - x11))*u5 =G= 0;

dL_dx12..  + (-0.25*(pi2 - 3.5*x12))*u4 + (-0.75*(pi2 - 3.5*x12))*u5 =G= 0;

dL_dy1.. (-(4 - pi1 - 2*y1))/(1) =G= 0;

dL_dy2.. (-(9.6 - pi2 - 10*y2))/(1) =G= 0;

dL_dthetaP.. -1 + u4 + u5 =E= 0;

* set non-default levels
pi1.l = 1;
pi2.l = 1;
* pi1.fx = 1.22557823;
* pi2.fx = 2.06984127;

* pi1.fx = 1.24782821;
* pi2.fx = 2.15636896;

Model m / e4.u4,e5.u5,e6.pi1,e7.pi2,dL_dx0.x0,dL_dx11.x11,dL_dx12.x12,dL_dy1.y1
         ,dL_dy2.y2,dL_dthetaP.thetaP /;

m.limrow=0; m.limcol=0;

Solve m using MCP;

$exit

pi1.lo = 0.9*pi1.l;
pi1.up = 1.2*pi1.l;
pi2.lo = 0.9*pi2.l;
pi2.up = 1.2*pi2.l;
Solve m using MCP;

parameter w1, w2, w;
w1 = -2*power(y1.l,2)/2 + ( 4 - pi1.l )*y1.l;
w2 = -10*power(y2.l,2)/2 + ( 48/5 - pi2.l )*y2.l;
w = y1.l + y2.l;
option decimals = 8;
display pi1.l, pi2.l, w1, w2, w, thetaP.l;
