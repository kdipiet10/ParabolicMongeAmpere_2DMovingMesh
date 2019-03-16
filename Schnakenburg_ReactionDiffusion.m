function Schnakenburg_ReactionDiffusion
%%% Implicit time stepping
%%% done in an alternating fashion -- PDE solved with Crank Nicholson
%%% Solving the Schnakenburg model on a unit disk, moving mesh
%%% v_t = eps^2 lap(v) - v + uv^2 + d_n(v) = 0
%%% u_t = 1/(eps^2)[Dlap(u) + a - eps^(-2)uv^2] + coupling_u, d_n(u) = 0
close all;
a = -1; b = -a;
nx = 45; ny = 45; 
%Defining the PMA parameters eps(I - gammaDelta)Q_t = (M(Q)H(Q))^(1/2)
gamma = 0.2; epsilon = 0.2; monsmooth = 4; mcav = 1; %%for the initial
%mesh %% good for the arclength monitor function
MeshPar = struct('epsilon',epsilon,'gamma',gamma,'monsmooth',monsmooth,...
    'mcav',mcav);
%Defining the PDE Parameters:
eps = 0.03; tau = eps^2; apde = 8.8; %%Splitting spot.
PDEPar = struct('eps',eps,'apde',apde,'tau',tau);
x0 = 0.3; y0 = 0.3;   %%Initial center of the spot. 
%%Square computational domain.
[x1,x2] = ndgrid(linspace(a,b,nx),linspace(a,b,ny));
hx = x1(2,1) - x1(1,1); hy = x2(1,2)-x2(1,1);
M = make_D_matrices(nx,ny,hx,hy);Ibdy = get_boundary_indices(a,b,x2,x1);
qstart = reshape(0.5*x1.^2 + 0.5*x2.^2,nx*ny,1); %Initial uniform mesh. 
X = struct('hx',hx,'hy',hy,'x1',x1,'x2',x2,'nx',nx,'ny',ny);

%%Represent the target set as a discrete number of points.
Targ.NY = 5*(nx+ny); theta = linspace(0,2*pi,Targ.NY)';
r1 =@(s) 1.0 + 0*s; Targ.xt = r1(theta).*cos(theta);
Targ.yt = r1(theta).*sin(theta);
j = 1:Targ.NY; int = (2*pi*j)/Targ.NY;
%%Admissable directions for mapping the signed distance function.
Targ.nj = [cos(int)' sin(int)'];
Targ.Hs = max([Targ.xt Targ.yt]*Targ.nj')';
%%%Initial density functions.
density.g = @(x,y) 1 + 0*x;density.f = @(x,y) 1 + 0*x;
density.gx = @(x,y) 0*x;density.gy = @(x,y) 0*x;
q0 = SignedDistance(qstart,X,Targ,M,Ibdy,density);
xorig = M.D1XC*q0; yorig = M.D1YC*q0;
%%%Make the initial condition with the steady state profile.
[v_ss,u_ss,rho] = findSchnakenSteadyStateFullSystem(apde);
num_points = size(u_ss,2);
radius = eps*rho(2:end);
theta = linspace(0,2*pi,num_points);
theta = theta(1:end-1);
[Xr,T] = meshgrid(radius,theta);
xx = Xr.*cos(T); yy = Xr.*sin(T);
uu = zeros(size(xx));
vv = zeros(size(xx));

for j = 1:(num_points-1)
    uu(j,:) = u_ss(2:end);
    vv(j,:) = v_ss(2:end);
end

FFu = scatteredInterpolant(xx(:),yy(:),uu(:));
FFv = scatteredInterpolant(xx(:),yy(:),vv(:));
v0 = FFv(xorig-x0,yorig-y0);
u0 = FFu(xorig-x0,yorig-y0);
sol = [v0;u0]; 
%% Finding a good initial mesh
dtmesh = 1e-5;  %time step for finding initial mesh
MeshPar.gamma = 0.075; MeshPar.epsilon = 0.075;
q0 = loop_pma(q0,v0,M,Ibdy,MeshPar,Targ.Hs,Targ.nj,dtmesh);
MeshPar.gamma = 0.2; MeshPar.epsilon = 0.2;
%% Update mesh every 1 sec time point 
Lap = build_lap(q0,M,Ibdy); Id = speye(size(Lap));  
Z = zeros(size(Id)); 
IA = [Id, Z; Z, Id]; 
L = [eps^2*Lap - Id, Z; Z, (1/tau)*Lap];
dt = 5e-2; tfinal = 115; time = 0; 
count = 0; count2 = 1; 
allsol = []; alltime = []; 
allq = []; 
while time < tfinal
    %%Crank Nicholson Time stepping
    dt2 = dt/2; 
    [sol,~] = dampednewtons(@(x) (IA-dt2*L)*x - (sol + dt2*(L*sol) + dt2*rhs(sol,M,PDEPar) + dt2*rhs(x,M,PDEPar)),...
        @(x)(IA-dt2*L)-dt2*Jac(x,M,PDEPar),sol,5e-3,22);
    if mod(count,20) == 0
        %% Add an update of the mesh here. 
        %%Old mesh
        xf1 = M.D1XC*q0; yf1 = M.D1YC*q0; 
        v0 = sol(1:nx*ny);
        u0 = sol(nx*ny+1:end);
        %%%Updated mesh
        q0 = loop_pma(q0,v0,M,Ibdy,MeshPar,Targ.Hs,Targ.nj,dtmesh);
        %q0 = quni; %%just a uniform mesh %%use for testing
        newx = M.D1XC*q0; newy = M.D1YC*q0; 
        Lap = build_lap(q0,M,Ibdy);
        L = [eps^2*Lap - Id, Z; Z, (1/tau)*Lap];
        %% Taylor expansion to interpolate onto the new mesh
        u0x = M.D1XC*u0; u0y = M.D1YC*u0; 
        v0x = M.D1XC*v0; v0y = M.D1YC*v0; 
        diffx = xf1 - newx; diffy = yf1 - newy; 
        v = v0 + v0x.*diffx + v0y.*diffy; 
        u = u0 + u0x.*diffx + u0y.*diffy; 
        sol = [v;u];
        xx1 = reshape(newx,nx,ny); yy1 = reshape(newy,nx,ny);
        %disp('update')
        figure(1)
        plot(xx1,yy1,'b',xx1',yy1','b');
        title(time+dt)
        figure(2)
        pcolor(xx1,yy1,reshape(v,nx,ny))
        title([time+dt max(v)])
    end
    count = count + 1; 
    time = time + dt; 
end


function out = PMA(q,adaptfun,M,Ibdy,MeshPar)
%%Input parameters:
%%t- current time, q -- current mesh.
%%adaptfun -- function (u or v) to adapt to
%%M -- derivative matrices. Ibdy -- boundary indices
%%MeshPar -- mesh parameters.
nx = M.nx; ny = M.ny;
eps = MeshPar.epsilon; gamma = MeshPar.gamma;
monsmooth = MeshPar.monsmooth; mcav = MeshPar.mcav;
hx = M.hx; hy = M.hy; %gamma = 0.01;
qder = meshder(q,M,Ibdy);
Q2eta = qder.Q2eta; Q2xi = qder.Q2xi;
Q2xieta = qder.Q2xieta;
%%%Solution derivatives for the monitor function
u_xic = M.D1XC*adaptfun; u_etac = M.D1YC*adaptfun;
J = Q2eta.*Q2xi-Q2xieta.^2;
%%%Smooth the jacobian at the corners
num_int = 2:3; num_int_m = num_int - 1;
n_int = 1:3; nm_int = n_int - 1;
%%%Bottom Left
blidx = [num_int,nx+n_int,2*nx+n_int];
J(Ibdy.BottomLeft) = sum(J(blidx))/length(blidx);
%%%Bottom Right
bridx = [nx-num_int_m,2*nx-nm_int,3*nx-nm_int];
J(Ibdy.BottomRight) = sum(J(bridx))/length(bridx);
%%%Top Left
tlidx = [(nx*ny)-nx+num_int,(nx*(ny-1))-nx+n_int,(nx*(ny-2))-nx+n_int];
J(Ibdy.TopLeft) = sum(J(tlidx))/length(tlidx);
%%%Top Right
tridx = [(nx*ny)-num_int_m,(nx*(ny-1))-nm_int,(nx*(ny-2))-nm_int];
J(Ibdy.TopRight) = sum(J(tridx))/length(tridx);
invJ = 1./J;
Q2eta = invJ.*Q2eta; Q2xieta = invJ.*Q2xieta; Q2xi = invJ.*Q2xi;
%%arc-length monitor function
u_x = (Q2eta.*u_xic - Q2xieta.*u_etac);
u_y = (-Q2xieta.*u_xic + Q2xi.*u_etac);
alpha = 0.95;
monitor = sqrt(1+alpha*(u_x.^2 + u_y.^2)); monitor = monitor.^2;
for i = 1:monsmooth
    monitor = smooth_mon(monitor,M); 
end
Mc = sum(monitor .* abs(J))*hx*hy;
monitor = monitor + mcav*Mc;

q_rhs = (monitor.*abs(J)).^(1/2);
a = dct2(reshape(q_rhs,nx,ny)/eps);
a = a./(1-gamma*M.Leig);
a = idct2(a);
out = reshape(a,nx*ny,1);

function out = loop_pma(q,adaptfun,M,Ibdy,MeshPar,Hs,nj,dt)
p_time = dt:dt:125*dt; thresh = 1e-3; count = 0;
errortol = 1e-6; numloops = 20;
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

function out = rhs(sol,M,PDEPar)

nx = M.nx; ny = M.ny;
out = zeros(size(sol));
v = sol(1:nx*ny); u = sol(nx*ny+1:end);
out(1:nx*ny) = u.*v.^2; 
out(nx*ny+1:end) = (PDEPar.apde - u.*(v.^2)/(PDEPar.eps^2))/PDEPar.tau;

function J = Jac(sol,M,PDEPar)
nx = M.nx; ny = M.ny;
nx2 = nx^2; ny2 = ny^2; 
v = sol(1:nx*ny); u = sol(nx*ny+1:end);
J = [spdiags(2*v.*u,0,nx2,ny2), spdiags(v.^2,0,nx2,ny2);...
    spdiags(-2*v.*u/(PDEPar.eps^2 * PDEPar.tau),0,nx2,ny2),...
    spdiags(-v.^2/(PDEPar.eps^2 * PDEPar.tau),0,nx2,ny2) ];

function L = build_lap(q,M,Ibdy)
nx = M.nx; ny = M.ny;
qder = meshder(q,M,Ibdy); 
Q2eta = qder.Q2eta; Q2xi = qder.Q2xi; Q2xieta = qder.Q2xieta;
J = Q2eta.*Q2xi-Q2xieta.^2;
%%%Smooth the jacobian at the corners
num_int = 2:3; num_int_m = num_int - 1;
n_int = 1:3; nm_int = n_int - 1;
%%Bottom Left
blidx = [num_int,nx+n_int,2*nx+n_int];
J(Ibdy.BottomLeft) = sum(J(blidx))/length(blidx);
%%%Bottom Right
bridx = [nx-num_int_m,2*nx-nm_int,3*nx-nm_int];
J(Ibdy.BottomRight) = sum(J(bridx))/length(bridx);
%%%Top Left
tlidx = [(nx*ny)-nx+num_int,(nx*(ny-1))-nx+n_int,(nx*(ny-2))-nx+n_int];
J(Ibdy.TopLeft) = sum(J(tlidx))/length(tlidx);
%%%Top Right
tridx = [(nx*ny)-num_int_m,(nx*(ny-1))-nm_int,(nx*(ny-2))-nm_int];
J(Ibdy.TopRight) = sum(J(tridx))/length(tridx);

invJ = 1./J;Q2eta = invJ.*Q2eta; Q2xieta = invJ.*Q2xieta; Q2xi = invJ.*Q2xi;
Luxx = Lxixi(Q2eta,Q2eta,M,Ibdy)-Lxieta(Q2xieta,Q2eta,M,Ibdy) -...
    Letaxi(Q2eta,Q2xieta,M,Ibdy) + Letaeta(Q2xieta,Q2xieta,M,Ibdy);
Luyy = Lxixi(Q2xieta,Q2xieta,M,Ibdy)-Lxieta(Q2xi,Q2xieta,M,Ibdy) - ...
    Letaxi(Q2xieta,Q2xi,M,Ibdy) + Letaeta(Q2xi,Q2xi,M,Ibdy);
L = Luxx + Luyy;
boundary_L = LinearBound(qder,M,Ibdy);
L(Ibdy.All,:) = boundary_L(Ibdy.All,:);

function [H,J] = findH(allq,Hs,nj,M,Ibdy)
%Find the hamilton-jacobi function of the signed distance.
nx = M.nx; ny = M.ny;H = zeros(nx*ny,1);
% Forward and backward derivatives.
D1XM = M.D1XM; D1XP = M.D1XP;
D1YM = M.D1YM; D1YP = M.D1YP;

qxm = D1XM*allq; qxp = D1XP*allq;
qym = D1YM*allq;qyp = D1YP*allq;

[H(Ibdy.All), iC] = max(max(nj(:,1),0)*qxm(Ibdy.All)' + min(nj(:,1),0)*qxp(Ibdy.All)'...
    + max(nj(:,2),0)*qym(Ibdy.All)' + min(nj(:,2),0)*qyp(Ibdy.All)' - repmat(Hs,[1,length(Ibdy.All)]) );
H = H(Ibdy.All);
% Analytical Jacobian.
n1 = zeros(nx*ny,1); n2 = n1;
n1(Ibdy.All) = nj(iC,1);  n2(Ibdy.All) = nj(iC,2);
J = spdiags(max(n1,0),0,nx*ny,nx*ny)*(M.D1XM) + spdiags(min(n1,0),0,nx*ny,nx*ny)*(M.D1XP)+...
    spdiags(max(n2,0),0,nx*ny,nx*ny)*(M.D1YM) + spdiags(min(n2,0),0,nx*ny,nx*ny)*(M.D1YP);
J = J(Ibdy.All,Ibdy.All);

function out = LinearBound(qder,M,Ibdy)
%%Get the linear matrix for the boundary condition.
h = M.hx; n = M.nx;
out = sparse(n^2,n^2);

Q2eta = qder.Q2eta; Q2xi = qder.Q2xi;
Q2xieta = qder.Q2xieta;

J = Q2eta.*Q2xi-Q2xieta.^2;
invJ = 1./J;
I = speye(n);
e = ones(n,1);

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
nx = M.nx; dx = M.hx; int = Ibdy.Interior;
%%set up the operator matrix.
diagup = A(int+1)+A(int);
diagmain = -(A(int+1)+2*A(int)+A(int-1));
diaglow = A(int)+A(int-1);
I = [int; int; int;Ibdy.All];
J = [int; int+1; int-1;Ibdy.All];
S = [B(int).*diagmain; B(int).*diagup; B(int).*diaglow;zeros(size(Ibdy.All))];
NewA = 0.5*sparse(I,J,S); %%%The L matrix for Detaeta (not applied to u yet)
NewA = NewA / (dx^2); %%For use in implicit methods.

function NewA = Letaeta(A,B,M,Ibdy)
dy = M.hy;ny = M.ny; int = Ibdy.Interior;
%%set up the operator matrix.
diagup = A(int+ny)+A(int);
diagmain = -(A(int+ny)+2*A(int)+A(int-ny));
diaglow = A(int)+A(int-ny);
I = [int; int; int; Ibdy.All];
J = [int; int+ny; int-ny;Ibdy.All];
S = [B(int).*diagmain;B(int).*diagup;B(int).*diaglow;zeros(size(Ibdy.All))];
NewA = 0.5*sparse(I,J,S); 
NewA = NewA / (dy^2);

function NewA = Lxieta(A,B,M,Ibdy)
dx = M.hx; nx = M.nx; dy = M.hy; ny = M.ny;
int = Ibdy.Interior;
%%set up the operator matrix.
idxA1 = A(int - (nx+1) + nx);
idxA2 = -A(int - (nx-1) + nx);
idxA3 = -A(int + (nx-1) - nx);
idxA4 = A(int + (nx+1) - nx);

I_A = [int;int;int;int;Ibdy.All];
J_A = [int-(ny+1); int-(ny-1); int+(ny-1); int+(ny+1);Ibdy.All];
S_A = [B(int).*idxA1;B(int).*idxA2;B(int).*idxA3;B(int).*idxA4; zeros(size(Ibdy.All))];
NewA = sparse(I_A, J_A, S_A);
NewA = NewA/(4*dx*dy);

function NewA = Letaxi(A,B,M,Ibdy)
%functino out = Detapsi(u,A)
% d/psi (A d/eta)
dx = M.hx; nx = M.nx;
dy = M.hy; ny = M.ny;
int = Ibdy.Interior;
%%set up the operator matrix.
idxA1 = A(int - (ny+1) + 1);
idxA2 = -A(int - (ny-1) -1);
idxA3 = -A(int + (ny-1) + 1);
idxA4 = A(int + (ny+1) - 1);

I_A = [int;int;int;int;Ibdy.All];
J_A = [int-(nx+1); int-(nx-1); int+(nx-1); int+(nx+1);Ibdy.All];
S_A = [B(int).*idxA1;B(int).*idxA2;B(int).*idxA3;B(int).*idxA4;zeros(size(Ibdy.All))];
NewA = sparse(I_A, J_A, S_A);
NewA = NewA / (4*dx*dy);

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

function out = smooth_mon(Mon,M)
nx = M.nx; ny = M.ny;
%hx = M.hx; hy = M.hy;
idx = 2:nx-1; idy = 2:ny-1;
Mon = reshape(Mon,nx,ny);
out = zeros(size(Mon));
out(idx,idy) = Mon(idx,idy) + 2/16*(Mon(idx+1,idy)+Mon(idx-1,idy)+Mon(idx,idy-1)+Mon(idx,idy+1)) + ...
    1/16*(Mon(idx+1,idy+1)+Mon(idx-1,idy-1)+Mon(idx+1,idy-1)+Mon(idx-1,idy+1));

% Boundary smoothing except corners;
out(2:end-1,ny) = (1/12)*(4*Mon(2:end-1,ny) +2*Mon(1:end-2,ny) +2*Mon(3:end,ny) + 2*Mon(2:end-1,ny-1) + Mon(3:end,ny-1)+Mon(1:end-2,ny-1));
out(2:end-1,1) = (1/12)*(4*Mon(2:end-1,1) +2*Mon(1:end-2,1) +2*Mon(3:end,1) + 2*Mon(2:end-1,2) + Mon(3:end,2)+Mon(1:end-2,2));
out(nx,2:end-1) = (1/12)*(4*Mon(nx,2:end-1) +2*Mon(nx,1:end-2) +2*Mon(nx,3:end) + 2*Mon(nx-1,2:end-1) + Mon(nx-1,3:end)+Mon(nx-1,1:end-2));
out(1,2:end-1) = (1/12)*(4*Mon(1,2:end-1) +2*Mon(1,1:end-2) +2*Mon(1,3:end) + 2*Mon(2,2:end-1) + Mon(2,3:end)+Mon(2,1:end-2));

% Smoothing the corners.

out(1,1) = (1/9)*( 4*Mon(1,1) + 2*Mon(1,2) + 2*Mon(2,1) + Mon(2,2) );
out(1,ny) = (1/9)*( 4*Mon(1,ny) + 2*Mon(2,ny) + 2*Mon(1,ny-1) + Mon(2,ny-1) );
out(nx,ny) = (1/9)*( 4*Mon(nx,ny) + 2*Mon(nx-1,ny) + 2*Mon(nx,ny-1) + Mon(nx-1,ny-1) );
out(nx,1) = (1/9)*( 4*Mon(nx,1) + 2*Mon(nx-1,1) + 2*Mon(nx,2) + Mon(nx-1,2) );

out = reshape(out,nx*ny,1);

function QDer = meshder(q,M,Ibdy)
QDer.Q1xi = M.D1XC*q; QDer.Q1eta = M.D1YC*q;
nx = M.nx; ny = M.ny;
facx = 25/(6*M.hx);facy = 25/(6*M.hy);
%Find the RHS of Q (i.e. second derivatives of Q and the boundary terms)
ff1 = zeros(nx*ny,1); ff1(Ibdy.Left) = -QDer.Q1xi(Ibdy.Left); 
ff1(Ibdy.Right) = QDer.Q1xi(Ibdy.Right);QDer.Q2xi = M.D2XXN*q + facx*ff1;
ff1 = zeros(nx*ny,1); ff1(Ibdy.Bottom) = -QDer.Q1eta(Ibdy.Bottom); 
ff1(Ibdy.Top) = QDer.Q1eta(Ibdy.Top); QDer.Q2eta = M.D2YYN*q + facy*ff1;
%Mixed derivative of Q
QDer.Q2xieta = M.D2XY4*q; %Q2xieta(Ibdy.All) = 0;

function M = make_D_matrices(nx,ny,hx,hy)
M.nx = nx; M.ny = ny; M.hx = hx; M.hy = hy; 
ex = ones(nx,1); ey = ones(ny,1);
Ix = speye(nx); Iy = speye(ny);

DPX = spdiags([-3*ex 4*ex -ex],[0 1 2],nx,nx);
DPX(nx-1,nx-2) = 0; DPX(nx-1,nx-1) = -2; DPX(nx-1,nx) = 2;
DPX(nx,nx) = 3; DPX(nx,nx-1)=-4; DPX(nx,nx-2) = 1;
DPX = DPX/(2*hx);

DPY = spdiags([-3*ey 4*ey -ey],[0 1 2],ny,ny);
DPY(ny-1,ny-2)=0; DPY(ny-1,ny-1)= -2; DPY(ny-1,ny)=2;
DPY(ny,ny)=3; DPY(ny,ny-1)=-4; DPY(ny,ny-2)=1;
DPY = DPY/(2*hy);

M.D1XP = kron(Iy,DPX); M.D1YP = kron(DPY,Ix);

DMX = spdiags([ex -4*ex 3*ex],[-2 -1 0],nx,nx);
DMX(1,1) = -3; DMX(1,2) = 4; DMX(1,3) = -1;
DMX(2,1) = -2; DMX(2,2) = 2;
DMX = DMX/(2*hx);

DMY = spdiags([ey -4*ey 3*ey],[-2 -1 0],ny,ny);
DMY(1,1) = -3; DMY(1,2) = 4; DMY(1,3) = -1;
DMY(2,1) = -2; DMY(2,2) = 2;
DMY = DMY/(2*hy);

M.D1XM = kron(Iy,DMX);M.D1YM = kron(DMY,Ix);

DCX = spdiags([-ex 0*ex ex],-1:1,nx,nx);
DCX(1,1) = -3; DCX(1,2) = 4; DCX(1,3) = -1;
DCX(end,end) = 3; DCX(end,end-1) = -4; DCX(end,end-2) = 1;
DCX = DCX/(2*hx);

DCY = spdiags([-ey 0*ey ey],-1:1,ny,ny);
DCY(1,1) = -3; DCY(1,2) = 4; DCY(1,3) = -1;
DCY(end,end) = 3; DCY(end,end-1) = -4; DCY(end,end-2) = 1;
DCY = DCY/(2*hy);

D2X = spdiags([ex -2*ex ex],-1:1,nx,nx);
D2X = D2X/(hx*hx);
D2Y = spdiags([ey -2*ey ey],-1:1,ny,ny);
D2Y = D2Y/(hy*hy);

M.D2XX = kron(Iy,D2X);M.D2YY = kron(D2Y,Ix);M.D2XY = kron(DCY,DCX);

%%%% Fourth Order Neumann.
zx = zeros(nx,1);
Neu_D24X = spdiags([-ex 16*ex -30*ex 16*ex -ex],-2:2,nx,nx);
Neu_D24X(1,1)=-415/6;Neu_D24X(1,2)=96;Neu_D24X(1,3)=-36;Neu_D24X(1,4)=32/3;Neu_D24X(1,5)=-3/2;
Neu_D24X(2,1)= 10;Neu_D24X(2,2)=-15;Neu_D24X(2,3)=-4;Neu_D24X(2,4)=14;Neu_D24X(2,5)=-6; Neu_D24X(2,6) = 1;
Neu_D24X(end,end)=-415/6;Neu_D24X(end,end-1)=96;Neu_D24X(end,end-2)=-36;
Neu_D24X(end,end-3)=32/3;Neu_D24X(end,end-4)=-3/2;
Neu_D24X(end-1,end) = 10; Neu_D24X(end-1,end-1) = -15; Neu_D24X(end-1,end-2) = -4;
Neu_D24X(end-1,end-3)=14;Neu_D24X(end-1,end-4)=-6; Neu_D24X(end-1,end-5) = 1;
Neu_D24X = Neu_D24X/(12*hx*hx);

zy = zeros(ny,1);
Neu_D24Y = spdiags([-ey 16*ey -30*ey 16*ey -ey],-2:2,ny,ny);
Neu_D24Y(1,1)=-415/6;Neu_D24Y(1,2)=96;Neu_D24Y(1,3)=-36;Neu_D24Y(1,4)=32/3;Neu_D24Y(1,5)=-3/2;
Neu_D24Y(2,1)= 10;Neu_D24Y(2,2)=-15;Neu_D24Y(2,3)=-4;Neu_D24Y(2,4)=14;Neu_D24Y(2,5)=-6; Neu_D24Y(2,6) = 1;
Neu_D24Y(end,end)=-415/6;Neu_D24Y(end,end-1)=96;Neu_D24Y(end,end-2)=-36;
Neu_D24Y(end,end-3)=32/3;Neu_D24Y(end,end-4)=-3/2;
Neu_D24Y(end-1,end) = 10; Neu_D24Y(end-1,end-1) = -15; Neu_D24Y(end-1,end-2) = -4;
Neu_D24Y(end-1,end-3)=14;Neu_D24Y(end-1,end-4)=-6; Neu_D24Y(end-1,end-5) = 1;
Neu_D24Y = Neu_D24Y/(12*hy*hy);

M.D2XXN = kron(Iy,Neu_D24X);M.D2YYN = kron(Neu_D24Y,Ix);

D114X = spdiags([ex -8*ex zx 8*ex -ex],-2:2,nx,nx);
D114X(1,1) = -25; D114X(1,2) = 48; D114X(1,3) = -36; D114X(1,4)= 16; D114X(1,5)= -3;
D114X(2,1) = -3; D114X(2,2) = -10; D114X(2,3) = 18; D114X(2,4) = -6; D114X(2,5) = 1;
D114X(end-1,end) = 3; D114X(end-1,end-1) = 10; D114X(end-1,end-2) = -18; D114X(end-1,end-3) = 6; D114X(end-1,end-4) = -1;
D114X(end,end) = 25; D114X(end,end-1) = -48; D114X(end,end-2) = 36; D114X(end,end-3)= -16; D114X(end,end-4)= 3;
D114X = D114X/(12*hx);

D114Y = spdiags([ey -8*ey zy 8*ey -ey],-2:2,ny,ny);
D114Y(1,1) = -25; D114Y(1,2) = 48; D114Y(1,3) = -36; D114Y(1,4)= 16; D114Y(1,5)= -3;
D114Y(2,1) = -3; D114Y(2,2) = -10; D114Y(2,3) = 18; D114Y(2,4) = -6; D114Y(2,5) = 1;
D114Y(end-1,end) = 3; D114Y(end-1,end-1) = 10; D114Y(end-1,end-2) = -18; D114Y(end-1,end-3) = 6; D114Y(end-1,end-4) = -1;
D114Y(end,end) = 25; D114Y(end,end-1) = -48; D114Y(end,end-2) = 36; D114Y(end,end-3)= -16; D114Y(end,end-4)= 3;
D114Y = D114Y/(12*hy);

M.D2XY4 = kron(D114Y,D114X);M.D1XC = kron(Iy,D114X);M.D1YC = kron(D114Y,Ix);

Leig  = (((2*cos(pi*(0:nx-1)'/(nx-1)))-2)*ones(1,ny)) + ...
    (ones(nx,1)*((2*cos(pi*(0:ny-1)/(ny-1)))-2));
M.Leig = Leig/(hx*hy); %size nx x ny
M1 = spdiags([ex 2*ex 1*ex],[-1 0 1],nx,ny);
%smoothing matrix
M.Sm = blktridiag((2/16)*M1,(1/16)*M1,(1/16)*M1,ny);

function Ibdy = get_boundary_indices(endpt1,endpt2,X,Y)

nx = numel(X);
allidx = 1:nx;

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

