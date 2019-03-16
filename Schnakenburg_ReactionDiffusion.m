function Schnakenburg_ReactionDiffusion
%%% done in an alternating fashion -- PDE solved with Crank Nicholson
%%% Solving the Schnakenburg model on a unit disk, moving mesh
%%% v_t = eps^2 lap(v) - v + uv^2 + d_n(v) = 0
%%% u_t = 1/(eps^2)[Dlap(u) + a - eps^(-2)uv^2] + coupling_u, d_n(u) = 0
%%% coupling_v = nabla_x(v) dot x_t
%%% coupling_u = nabla_x(u) dot x_t  (couupling with the mesh Q)
close all;
a = -1; b = -a; %Square Computational Domain
X.n = 31; %number of grid points. 
%%PMA Parameters
MeshPar.gamma = 0.2; MeshPar.epsilon = 0.2; 
MeshPar.monsmooth = 4; MeshPar.mcav = 1; 
%%%PDE Parameters
PDEPar.eps = 0.03; PDEPar.tau = eps^2; PDEPar.apde = 8.8; 
x0 = 0.3; y0 = 0.3; %%%Center of spot
%%Square computational domain
[X.x1,X.x2] = ndgrid(linspace(a,b,X.n),linspace(a,b,X.n));
X.h = X.x1(2,1)-X.x1(1,1); 
M = make_D_matrices(X.n,X.h); %Derivative matrix
Ibdy = get_boundary_indices(a,b,X.x2,X.x1); %Boundary indices
qstart = reshape(0.5*X.x1.^2 + 0.5*X.x2.^2,X.n^2,1); %initial uniform mesh
%Represent the circle target domain
Targ.NY = 10*X.n; %number of discrete points
theta = linspace(0,2*pi,Targ.NY)'; 
r1 =@(s) 1.0 + 0*s; Targ.xt = r1(theta).*cos(theta); 
Targ.yt = r1(theta).*sin(theta); 
j = 1:Targ.NY;
%Admissible directions for signed distane
Targ.nj = [cos((2*pi*j)/Targ.NY)' sin((2*pi*j)/Targ.NY)'];
Targ.Hs = max([Targ.xt Targ.yt]*Targ.nj')'; %Legendre Fencel Transform
%%%Initial uniform density functions: 
density.g = @(x,y) 1 + 0*x; density.f = @(x,y) 1 + 0*x;
density.gx = @(x,y) 0*x; density.gy = @(x,y) 0*x;
%%Initial circle adapted grid. 
q0 = SignedDistance(qstart,X,Targ,M,Ibdy,density); 
xorig = M.D1XC*q0; yorig = M.D1YC*q0;
%%%Make the initial condition from the steady state profile. 
[v_ss,u_ss,rho] = findSchnakenSteadyStateFullSystem(PDEPar.apde);
num_points = size(u_ss,2);
radius = PDEPar.eps*rho(2:end);theta = linspace(0,2*pi,num_points);
theta = theta(1:end-1);[Xr,T] = meshgrid(radius,theta);
xx = Xr.*cos(T); yy = Xr.*sin(T);
uu = zeros(size(xx));vv = zeros(size(xx));
for j = 1:(num_points-1)
    uu(j,:) = u_ss(2:end);
    vv(j,:) = v_ss(2:end);
end
FFu = scatteredInterpolant(xx(:),yy(:),uu(:));
FFv = scatteredInterpolant(xx(:),yy(:),vv(:));
v0 = FFv(xorig-x0,yorig-y0); u0 = FFu(xorig-x0,yorig-y0);
sol = [v0;u0]; 
%%% Find a good initial mesh
dtmesh = 1e-5; %time step for finding initial mesh
MeshPar.gamma = 0.075; MeshPar.epsilon = 0.075;
q0 = loop_pma(q0,v0,M,Ibdy,MeshPar,Targ.Hs,Targ.nj,dtmesh);
MeshPar.gamma = 0.2; MeshPar.epsilon = 0.2;
%%%Solving the Schnakenburg problem. Update Mesh every 1 sec time point.
%%%Build Laplacian for Crank Nicolson Method 
Lap = build_lap(q0,M,Ibdy); Id = speye(size(Lap));
Z = zeros(size(Id)); IA = [Id, Z; Z, Id]; 
L = [(PDEPar.eps)^2*Lap - Id, Z; Z, (1/PDEPar.tau)*Lap];
dt = 5e-2; tfinal = 115; time = 0; count = 0; 
while time < tfinal
    %%Crank Nicholson Time stepping
    dt2 = dt/2; 
    [sol,~] = dampednewtons(@(x) (IA-dt2*L)*x - (sol + dt2*(L*sol) + dt2*rhs(sol,M,PDEPar) + dt2*rhs(x,M,PDEPar)),...
        @(x)(IA-dt2*L)-dt2*Jac(x,M,PDEPar),sol,5e-3,20);
    if mod(count,20) == 0
        %% Add an update of the mesh here.
        %%Old mesh
        xf1 = M.D1XC*q0; yf1 = M.D1YC*q0; 
        v0 = sol(1:X.n^2);
        u0 = sol(X.n^2+1:end);
        %%%Updated mesh
        q0 = loop_pma(q0,v0,M,Ibdy,MeshPar,Targ.Hs,Targ.nj,dtmesh);
        newx = M.D1XC*q0; newy = M.D1YC*q0; 
        Lap = build_lap(q0,M,Ibdy);
        L = [(PDEPar.eps^2)*Lap - Id, Z; Z, (1/PDEPar.tau)*Lap];
        %% Taylor expansion to interpolate onto the new mesh
        u0x = M.D1XC*u0; u0y = M.D1YC*u0; 
        v0x = M.D1XC*v0; v0y = M.D1YC*v0; 
        diffx = xf1 - newx; diffy = yf1 - newy; 
        v = v0 + v0x.*diffx + v0y.*diffy; 
        u = u0 + u0x.*diffx + u0y.*diffy; 
        sol = [v;u];
        xx1 = reshape(newx,X.n,X.n); yy1 = reshape(newy,X.n,X.n);
        figure(1)
        plot(xx1,yy1,'b',xx1',yy1','b');
        title(time+dt)
        figure(2)
        pcolor(xx1,yy1,reshape(v,X.n,X.n))
        title([time+dt max(v)])
        drawnow
    end
    count = count + 1; 
    time = time + dt; 
end

function out = rhs(sol,M,PDEPar)
%%%Nonlinear part of the Schnakenburg Problem
n = M.n; out = zeros(size(sol));
v = sol(1:n^2); u = sol(n^2+1:end);
out(1:n^2) = u.*v.^2; 
out(n^2+1:end) = (PDEPar.apde - u.*(v.^2)/(PDEPar.eps^2))/PDEPar.tau;

function J = Jac(sol,M,PDEPar)
%%Jacobian of nonlinear part of Schnakenburg Problem
n = M.n; n2 = n^2; 
v = sol(1:n^2); u = sol(n^2+1:end);
J = [spdiags(2*v.*u,0,n2,n2), spdiags(v.^2,0,n2,n2);...
    spdiags(-2*v.*u/(PDEPar.eps^2 * PDEPar.tau),0,n2,n2),...
    spdiags(-v.^2/(PDEPar.eps^2 * PDEPar.tau),0,n2,n2) ];

function L = build_lap(q,M,Ibdy)
n = M.n;qder = meshder(q,M,Ibdy); 
J = qder.Q2eta.*qder.Q2xi-qder.Q2xieta.^2;
%%%Smooth the jacobian at the corners
num_int = 2:3; num_int_m = num_int - 1;
n_int = 1:3; nm_int = n_int - 1;
%%Bottom Left
blidx = [num_int,n+n_int,2*n+n_int];
J(Ibdy.BottomLeft) = sum(J(blidx))/length(blidx);
%%%Bottom Right
bridx = [n-num_int_m,2*n-nm_int,3*n-nm_int];
J(Ibdy.BottomRight) = sum(J(bridx))/length(bridx);
%%%Top Left
tlidx = [(n^2)-n+num_int,(n*(n-1))-n+n_int,(n*(n-2))-n+n_int];
J(Ibdy.TopLeft) = sum(J(tlidx))/length(tlidx);
%%%Top Right
tridx = [(n^2)-num_int_m,(n*(n-1))-nm_int,(n*(n-2))-nm_int];
J(Ibdy.TopRight) = sum(J(tridx))/length(tridx);
invJ = 1./J;
Q2eta = invJ.*qder.Q2eta; Q2xieta = invJ.*qder.Q2xieta;
Q2xi = invJ.*qder.Q2xi;
Luxx = Lxixi(Q2eta,Q2eta,M,Ibdy)-Lxieta(Q2xieta,Q2eta,M,Ibdy) -...
    Letaxi(Q2eta,Q2xieta,M,Ibdy) + Letaeta(Q2xieta,Q2xieta,M,Ibdy);
Luyy = Lxixi(Q2xieta,Q2xieta,M,Ibdy)-Lxieta(Q2xi,Q2xieta,M,Ibdy) - ...
    Letaxi(Q2xieta,Q2xi,M,Ibdy) + Letaeta(Q2xi,Q2xi,M,Ibdy);
L = Luxx + Luyy;
boundary_L = LinearBound(qder,M,Ibdy);
L(Ibdy.All,:) = boundary_L(Ibdy.All,:);

function out = LinearBound(qder,M,Ibdy)
%%Get the linear matrix for the boundary condition.

%%Assuming the computational domain is a square.
h = M.h; n = M.n;
out = sparse(n^2,n^2);

Q2eta = qder.Q2eta; Q2xi = qder.Q2xi;
Q2xieta = qder.Q2xieta;

J = Q2eta.*Q2xi-Q2xieta.^2;
invJ = 1./J; I = speye(n);e = ones(n,1);

A11 = invJ .* (Q2xieta.^2 + Q2eta.^2);
A21 = -invJ .* (Q2xieta.* (Q2xi + Q2eta));
A22 = invJ .* (Q2xieta.^2 + Q2xi.^2);

BT = A11 - (A21.^2)./A22;
LR = A22 - (A21.^2)./A11;

Dtemp = sparse([1,1,n,n],[2,3,n-2,n-1],[2,-1/2,1/2,-2],n,n);

JLR = h*kron(I,Dtemp)*(spdiags(A11,0,n^2,n^2)*M.D1XC + spdiags(A21,0,n^2,n^2)*M.D1YC);
JTB = h*kron(Dtemp,I)*(spdiags(A22,0,n^2,n^2)*M.D1YC + spdiags(A21,0,n^2,n^2)*M.D1XC);

ARL = 0.5*(spdiags(kron(spdiags([e e],0:1,n,n),I)*LR,0,n^2,n^2)*kron(spdiags([-e e],0:1,n,n),I)... 
    - spdiags(kron(spdiags([e e],-1:0,n,n),I)*LR,0,n^2,n^2)*kron(spdiags([-e e],-1:0,n,n),I));
 
ATB = 0.5*(spdiags(kron(I,spdiags([e e],0:1,n,n))*BT,0,n^2,n^2)*kron(I,spdiags([-e e],0:1,n,n))...
    - spdiags(kron(I,spdiags([e e],-1:0,n,n))*BT,0,n^2,n^2)*kron(I,spdiags([-e e],-1:0,n,n))) ;

out(Ibdy.Top,:) = ATB(Ibdy.Top,:) + JTB(Ibdy.Top,:);
out(Ibdy.Bottom,:) = ATB(Ibdy.Bottom,:) + JTB(Ibdy.Bottom,:);
out(Ibdy.Left,:) = ARL(Ibdy.Left,:) + JLR(Ibdy.Left,:);
out(Ibdy.Right,:) = ARL(Ibdy.Right,:) + JLR(Ibdy.Right,:);

JCs = -h*(kron(Dtemp,I)*spdiags(LR,0,n^2,n^2)*M.D1YC + kron(I,Dtemp)*spdiags(BT,0,n^2,n^2)*M.D1XC);
out(Ibdy.BottomLeft,:) = JCs(Ibdy.BottomLeft,:);
out(Ibdy.BottomRight,:) = JCs(Ibdy.BottomRight,:);
out(Ibdy.TopLeft,:) = JCs(Ibdy.TopLeft,:);
out(Ibdy.TopRight,:) = JCs(Ibdy.TopRight,:);

out = spdiags(invJ,0,n^2,n^2)*out/(h^2);

function NewA = Lxixi(A,B,M,Ibdy)
%function out = Detaeta(u,A)

% % d/eta (A d/eta)
% % Top boundary: x(nx,:), y(nx,:).
% % Bottom Boundary: x(1,:),y(1,:).
h = M.h;int = Ibdy.Interior;
%%set up the operator matrix.
diagup = A(int+1)+A(int);
diagmain = -(A(int+1)+2*A(int)+A(int-1));
diaglow = A(int)+A(int-1);
I = [int; int; int;Ibdy.All];
J = [int; int+1; int-1;Ibdy.All];
S = [B(int).*diagmain; B(int).*diagup; B(int).*diaglow;zeros(size(Ibdy.All))];
NewA = 0.5*sparse(I,J,S); %%%The L matrix for Detaeta (not applied to u yet)
NewA = NewA / (h^2); %%For use in implicit methods.

function NewA = Letaeta(A,B,M,Ibdy)
h = M.h; n = M.n;int = Ibdy.Interior;

%%set up the operator matrix.
diagup = A(int+n)+A(int);
diagmain = -(A(int+n)+2*A(int)+A(int-n));
diaglow = A(int)+A(int-n);
I = [int; int; int; Ibdy.All];
J = [int; int+n; int-n;Ibdy.All];
S = [B(int).*diagmain;B(int).*diagup;B(int).*diaglow;zeros(size(Ibdy.All))];
NewA = 0.5*sparse(I,J,S); %%%The L matrix for Dxixi (not applied to u yet)
NewA = NewA / (h^2);

function NewA = Lxieta(A,B,M,Ibdy)
h = M.h; n = M.n;
int = Ibdy.Interior;

%%set up the operator matrix.
idxA1 = A(int - (n+1) + n);
idxA2 = -A(int - (n-1) + n);
idxA3 = -A(int + (n-1) - n);
idxA4 = A(int + (n+1) - n);

I_A = [int;int;int;int;Ibdy.All];
J_A = [int-(n+1); int-(n-1); int+(n-1); int+(n+1);Ibdy.All];
S_A = [B(int).*idxA1;B(int).*idxA2;B(int).*idxA3;B(int).*idxA4; zeros(size(Ibdy.All))];
NewA = sparse(I_A, J_A, S_A);
NewA = NewA/(4*h^2);

function NewA = Letaxi(A,B,M,Ibdy)
%functino out = Detapsi(u,A)
% d/psi (A d/eta)
h = M.h; n = M.n; 
int = Ibdy.Interior;
%%set up the operator matrix.
idxA1 = A(int - (n+1) + 1);
idxA2 = -A(int - (n-1) -1);
idxA3 = -A(int + (n-1) + 1);
idxA4 = A(int + (n+1) - 1);

I_A = [int;int;int;int;Ibdy.All];
J_A = [int-(n+1); int-(n-1); int+(n-1); int+(n+1);Ibdy.All];
S_A = [B(int).*idxA1;B(int).*idxA2;B(int).*idxA3;B(int).*idxA4;zeros(size(Ibdy.All))];
NewA = sparse(I_A, J_A, S_A);
NewA = NewA / (4*h^2);

function out = PMA(q,adaptfun,M,Ibdy,MeshPar)
%%Input parameters:
%%t- current time, q -- current mesh.
%%adaptfun -- function (u or v) to adapt to
%%M -- derivative matrices. Ibdy -- boundary indices
%%MeshPar -- mesh parameters.
n = M.n; h = M.h; 
qder = meshder(q,M,Ibdy);
%%%Solution derivatives for the monitor function (if based on v)
u_xic = M.D1XC*adaptfun; u_etac = M.D1YC*adaptfun;
J = qder.Q2eta.*qder.Q2xi-qder.Q2xieta.^2;
%%%Smooth the jacobian at the corners
num_int = 2:3; num_int_m = num_int - 1;
n_int = 1:3; nm_int = n_int - 1;
%%%Bottom Left
blidx = [num_int,n+n_int,2*n+n_int];
J(Ibdy.BottomLeft) = sum(J(blidx))/length(blidx);
%%%Bottom Right
bridx = [n-num_int_m,2*n-nm_int,3*n-nm_int];
J(Ibdy.BottomRight) = sum(J(bridx))/length(bridx);
%%%Top Left
tlidx = [(n^2)-n+num_int,(n*(n-1))-n+n_int,(n*(n-2))-n+n_int];
J(Ibdy.TopLeft) = sum(J(tlidx))/length(tlidx);
%%%Top Right
tridx = [(n^2)-num_int_m,(n*(n-1))-nm_int,(n*(n-2))-nm_int];
J(Ibdy.TopRight) = sum(J(tridx))/length(tridx);
invJ = 1./J;
Q2eta = invJ.*qder.Q2eta; Q2xieta = invJ.*qder.Q2xieta;
Q2xi = invJ.*qder.Q2xi;

%%Arc-length monitor function
u_x = (Q2eta.*u_xic - Q2xieta.*u_etac);
u_y = (-Q2xieta.*u_xic + Q2xi.*u_etac);
monitor = sqrt(1+0.95*(u_x.^2 + u_y.^2)); monitor = monitor.^2;
for i = 1:MeshPar.monsmooth
    monitor = smooth_mon(monitor,M);
end
Mc = sum(monitor .* abs(J))*h^2;
monitor = monitor + MeshPar.mcav*Mc;

q_rhs = (monitor.*abs(J)).^(1/2);
a = dct2(reshape(q_rhs,n,n)/MeshPar.epsilon);
a = a./(1-MeshPar.gamma*M.Leig);
a = idct2(a);
out = reshape(a,n^2,1);

function out = loop_pma(q,adaptfun,M,Ibdy,MeshPar,Hs,nj,dt)
p_time = dt:dt:125*dt; thresh = 1e-3; count = 0;
errortol = 1e-6; numloops = 20;
%%Start by evolving just the PMA to get a good intially adapted mesh
for k = 1:length(p_time)
   newq = q + dt*PMA(q,adaptfun,M,Ibdy,MeshPar);
   maxH = max(abs(findH(newq,Hs,nj,M,Ibdy))); %%convex domains
   if maxH > thresh
       count = count + 1;
       newq(Ibdy.All) = newton(newq,Hs,nj,Ibdy,M,errortol,numloops);
   end
   q = newq;
end
out = q;

function [u0,error] = newton(initu,Hs,nj,Ibdy,M,errortol,numloops)
error = 10;
alpha0 = 1; %initial guess for alpha.
%Parameters for damped netwons iterations
tau = 0.1; sigma = 0.01; alphamin = 0.01; %Min alpha value.
loopcount = 0;
repeat = 1;
repeat2 = 1;
u0 = initu(Ibdy.All);
uh = initu;
while repeat
    [F,Jac] = findH(uh,Hs,nj,M,Ibdy);
    psi = Jac \ -F;
    g0 = 1/2*sum(psi.^2);
    if loopcount > 1
        uh(Ibdy.All) = u0 + alpha0*psi; %%Update the boundary only
        [fsh,~] = findH(uh,Hs,nj,M,Ibdy);
        mut = 1/2*sum(prevpsi.^2).^(1/2);
        mub = 1/2*sum((psi-prevjac\fsh).^2).^(1/2);
        mu = mut/mub * alpha0;
        alpha0 = max(alphamin,min(mu,1));
    end
    
    while repeat2
        %Finding the optimal alpha.
        uh(Ibdy.All) = u0 + alpha0*psi;
        [fsh,~] = findH(uh,Hs,nj,M,Ibdy);
        gl = 1/2*sum((Jac\fsh).^2);
        if gl <= (1-2*alpha0*sigma)*g0
            break;
        else
            alpha0 = max(tau*alpha0,(alpha0^2*g0)/((2*alpha0-1)*g0+gl));
        end
    end
    prevpsi = psi;
    prevjac = Jac;
    error = max(abs(F));
    
    u0 = uh(Ibdy.All);
    
    loopcount = loopcount + 1;
    repeat = ((error>errortol) && (loopcount < numloops));
end

function [H,J] = findH(allq,Hs,nj,M,Ibdy)
%Find the hamilton-jacobi function of the signed distance.
%Defined on page 7.
%Upwind Discretizations found on 9-10.

n = M.n; H = zeros(n^2,1);
% Forward and backward derivatives.
D1XM = M.D1XM; D1XP = M.D1XP;
D1YM = M.D1YM; D1YP = M.D1YP;

qxm = D1XM*allq;qxp = D1XP*allq;
qym = D1YM*allq;qyp = D1YP*allq;

[H(Ibdy.All), iC] = max(max(nj(:,1),0)*qxm(Ibdy.All)' + min(nj(:,1),0)*qxp(Ibdy.All)'...
    + max(nj(:,2),0)*qym(Ibdy.All)' + min(nj(:,2),0)*qyp(Ibdy.All)' - repmat(Hs,[1,length(Ibdy.All)]) );
H = H(Ibdy.All);

% Analytical Jacobian.
n1 = zeros(n^2,1); n2 = n1;
n1(Ibdy.All) = nj(iC,1);  n2(Ibdy.All) = nj(iC,2);
J = spdiags(max(n1,0),0,n^2,n^2)*(M.D1XM) + spdiags(min(n1,0),0,n^2,n^2)*(M.D1XP)+...
    spdiags(max(n2,0),0,n^2,n^2)*(M.D1YM) + spdiags(min(n2,0),0,n^2,n^2)*(M.D1YP);
J = J(Ibdy.All,Ibdy.All);

function out = smooth_mon(Mon,M)
n = M.n; idx = 2:n-1; 
Mon = reshape(Mon,n,n);out = zeros(size(Mon));
out(idx,idx) = Mon(idx,idx) + 2/16*(Mon(idx+1,idx)+Mon(idx-1,idx)+Mon(idx,idx-1)+Mon(idx,idx+1)) + ...
    1/16*(Mon(idx+1,idx+1)+Mon(idx-1,idx-1)+Mon(idx+1,idx-1)+Mon(idx-1,idx+1));

% Boundary smoothing except corners;
out(2:end-1,n) = (1/12)*(4*Mon(2:end-1,n) +2*Mon(1:end-2,n) +2*Mon(3:end,n) + 2*Mon(2:end-1,n-1) + Mon(3:end,n-1)+Mon(1:end-2,n-1));
out(2:end-1,1) = (1/12)*(4*Mon(2:end-1,1) +2*Mon(1:end-2,1) +2*Mon(3:end,1) + 2*Mon(2:end-1,2) + Mon(3:end,2)+Mon(1:end-2,2));
out(n,2:end-1) = (1/12)*(4*Mon(n,2:end-1) +2*Mon(n,1:end-2) +2*Mon(n,3:end) + 2*Mon(n-1,2:end-1) + Mon(n-1,3:end)+Mon(n-1,1:end-2));
out(1,2:end-1) = (1/12)*(4*Mon(1,2:end-1) +2*Mon(1,1:end-2) +2*Mon(1,3:end) + 2*Mon(2,2:end-1) + Mon(2,3:end)+Mon(2,1:end-2));

% Smoothing the corners.

out(1,1) = (1/9)*( 4*Mon(1,1) + 2*Mon(1,2) + 2*Mon(2,1) + Mon(2,2) );
out(1,n) = (1/9)*( 4*Mon(1,n) + 2*Mon(2,n) + 2*Mon(1,n-1) + Mon(2,n-1) );
out(n,n) = (1/9)*( 4*Mon(n,n) + 2*Mon(n-1,n) + 2*Mon(n,n-1) + Mon(n-1,n-1) );
out(n,1) = (1/9)*( 4*Mon(n,1) + 2*Mon(n-1,1) + 2*Mon(n,2) + Mon(n-1,2) );

out = reshape(out,n^2,1);

function QDer = meshder(q,M,Ibdy)
QDer.Q1xi = M.D1XC*q; QDer.Q1eta = M.D1YC*q;
n = M.n; fac = 25/(6*M.h);
%Find the RHS of Q (i.e. second derivatives of Q and the boundary terms)
ff1 = zeros(n^2,1); ff1(Ibdy.Left) = -QDer.Q1xi(Ibdy.Left); 
ff1(Ibdy.Right) = QDer.Q1xi(Ibdy.Right);QDer.Q2xi = M.D2XXN*q + fac*ff1;
ff1 = zeros(n^2,1); ff1(Ibdy.Bottom) = -QDer.Q1eta(Ibdy.Bottom); 
ff1(Ibdy.Top) = QDer.Q1eta(Ibdy.Top); QDer.Q2eta = M.D2YYN*q + fac*ff1;
%Mixed derivative of Q
QDer.Q2xieta = M.D2XY4*q; 

function [last_v,last_u,rho] = findSchnakenSteadyStateFullSystem(A)
%%Function to find the steady state profile of the Schnakenburg system. 
%%Input- A: the value A from the Schnakenburg system 
%%Ouput - last_v: 1D profile for v. last_u: 1D profile for u. rho:
%%coordinate system used for the steady state profile. 
n = 500;
a = 0; b = 15; %Domain endpoints

%%%Create an initial guess for the BVP
x = linspace(a,b,n);
solinit = bvpinit(x,@(x) initcond(x,A));
options = [];
%%Solve the for the steady state using BVP4C
sol = bvp4c(@emdenode,@emdenbc,solinit,options,A);
%%Evaluate and export the solution on the mesh. 
y = deval(sol,x);
last_u = y(1,:);
last_v = y(3,:);
rho = x;

function dydx = emdenode(x,y,A)
%%Setting the right hand side of the system for BVP4C
if (x==0)
    dydx = [y(2), 0.5*(y(1)*y(3)^2), y(4), 0.5*(y(3)-y(1)*y(3)^2)];
else
    dydx = [y(2), -y(2)/x + y(1)*y(3)^2, y(4), -y(4)/x + y(3) - y(1)*y(3)^2];
end

function res = emdenbc(ya,yb,A)
%%Boundary condtions for the system. 
%%u'(0) = 0; u(L) = Slog(p) + chi(S)
%%v'(0) = 0; v(L) = 0;
L = 15;
S = A / 2;

res = [ya(2), yb(2) - (S/L), ya(4), yb(3)];

function yinit = initcond(x,A)
%%Set the initial guess for the BVP solver.
S = A / 2;
yinit = [S*log(1+x),0,2*sech(x).^2,0];

function Ibdy = get_boundary_indices(endpt1,endpt2,X,Y)
%%Boundary indices for the domain. 
n = numel(X);allidx = 1:n;

X = reshape(X',[],1); clear x;
Y = reshape(Y',[],1); clear y;

Ibdy.All = find( (X==endpt1) | (X==endpt2) | (Y==endpt1) | (Y==endpt2));
Ibdy.Interior = setdiff(allidx,Ibdy.All)';

Ibdy.Top = find(Y==endpt2);Ibdy.Bottom = find(Y==endpt1);
Ibdy.Left = find(X==endpt1);Ibdy.Right = find(X==endpt2);

Ibdy.BottomLeft = find((X==endpt1)& (Y==endpt1));
Ibdy.BottomRight = find((X==endpt2)&(Y==endpt1));
Ibdy.TopLeft = find((X==endpt1)&(Y==endpt2));
Ibdy.TopRight = find((X==endpt2)&(Y==endpt2));

function M = make_D_matrices(n,h)
%%%Make the derivative matrices. Input: n(grid size) h(grid step)
%%%Output - struct of derivative matrices
e = ones(n,1); I = speye(n); M.n = n; M.h = h; 
%Forward differencing
DP = spdiags([-3*e 4*e -e],[0 1 2],n,n);
DP(n-1,n-2) = 0; DP(n-1,n-1) = -2; DP(n-1,n) = 2;
DP(n,n) = 3; DP(n,n-1)=-4; DP(n,n-2) = 1;
DP = DP/(2*h);
M.D1XP = kron(I,DP);M.D1YP = kron(DP,I);
%Backward differencing
DM = spdiags([e -4*e 3*e],[-2 -1 0],n,n);
DM(1,1) = -3; DM(1,2) = 4; DM(1,3) = -1;
DM(2,1) = -2; DM(2,2) = 2;
DM = DM/(2*h);
M.D1XM = kron(I,DM); M.D1YM = kron(DM,I);
%Central differencing
DC = spdiags([-e 0*e e],-1:1,n,n);
DC(1,1) = -3; DC(1,2) = 4; DC(1,3) = -1;
DC(end,end) = 3; DC(end,end-1) = -4; DC(end,end-2) = 1;
DC = DC/(2*h);
D2 = spdiags([e -2*e e],-1:1,n,n);
D2 = D2/(h*h);
M.D2XX = kron(I,D2);
M.D2YY = kron(D2,I);
M.D2XY = kron(DC,DC);

%%%% Fourth Order Neumann.
z = zeros(n,1);
Neu_D24 = spdiags([-e 16*e -30*e 16*e -e],-2:2,n,n);
Neu_D24(1,1)=-415/6;Neu_D24(1,2)=96;Neu_D24(1,3)=-36;Neu_D24(1,4)=32/3;Neu_D24(1,5)=-3/2;
Neu_D24(2,1)= 10;Neu_D24(2,2)=-15;Neu_D24(2,3)=-4;Neu_D24(2,4)=14;Neu_D24(2,5)=-6; Neu_D24(2,6) = 1;
Neu_D24(end,end)=-415/6;Neu_D24(end,end-1)=96;Neu_D24(end,end-2)=-36;
Neu_D24(end,end-3)=32/3;Neu_D24(end,end-4)=-3/2;
Neu_D24(end-1,end) = 10; Neu_D24(end-1,end-1) = -15; Neu_D24(end-1,end-2) = -4;
Neu_D24(end-1,end-3)=14;Neu_D24(end-1,end-4)=-6; Neu_D24(end-1,end-5) = 1;
Neu_D24 = Neu_D24/(12*h^2);
M.D2XXN = kron(I,Neu_D24);M.D2YYN = kron(Neu_D24,I);
%%Fourth order mixed
D114 = spdiags([e -8*e z 8*e -e],-2:2,n,n);
D114(1,1) = -25; D114(1,2) = 48; D114(1,3) = -36; D114(1,4)= 16; D114(1,5)= -3;
D114(2,1) = -3; D114(2,2) = -10; D114(2,3) = 18; D114(2,4) = -6; D114(2,5) = 1;
D114(end-1,end) = 3; D114(end-1,end-1) = 10; D114(end-1,end-2) = -18; D114(end-1,end-3) = 6; D114(end-1,end-4) = -1;
D114(end,end) = 25; D114(end,end-1) = -48; D114(end,end-2) = 36; D114(end,end-3)= -16; D114(end,end-4)= 3;
D114 = D114/(12*h);

M.D2XY4 = kron(D114,D114);M.D1XC = kron(I,D114); M.D1YC = kron(D114,I);

Leig  = (((2*cos(pi*(0:n-1)'/(n-1)))-2)*ones(1,n)) + ...
    (ones(n,1)*((2*cos(pi*(0:n-1)/(n-1)))-2));
M.Leig = Leig/h^2; %size nx x ny

