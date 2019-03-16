function q = SignedDistance(qstart,X,Targ,M,Ibdy,density)

%%%% Code for solving the nonlinear Monge Ampere equation. Assumes that we
%%%% are using a signed distance function for the Hamilton Jacobi problem
%%%% H(grad(q(x)) = 0

%%% Solves the nonlinear equation using Newton's method. 

%%%Input: qstart -- original mesh, X -- the computational domain. Targ --
%%%the physical, target domain. M -- the mesh derivatives, a structure.
%%%Ibdy -- boundary indices, a structure. f -- computational domain density
%%%function, g -- physical domain density function. density -- manual
%%%density function. 
[q,~] = newton(qstart,Targ.Hs,Targ.nj,Ibdy,M,1e-6,20,X,density);

function [F,J] = rhs(u0,Hs,nj,Ibdy,M,X,density)
n2 = M.nx*M.ny;
ufix = u0(ceil(n2/2));

[F,J] = MA(u0,ufix,M,Ibdy,X,density);
[Ftemp,JTemp] = findH(u0,Hs,nj,M,Ibdy);
F(Ibdy.All) = Ftemp(Ibdy.All);
J(Ibdy.All,:) = JTemp(Ibdy.All,:);

function [out, J] = MA(u,ufix,M,Ibdy,X,density)
%Finding the Monge-Ampere equation on the interior points. 

nx = M.nx; ny = M.ny;  x1 = X.x1; x2 = X.x2; 
x1 = reshape(x1,nx*ny,1); x2 = reshape(x2,nx*ny,1);

uxx = M.D2XX * u;   uyy = M.D2YY * u;   uxy = M.D2XY * u;

f = density.f(x1,x2);g = density.g(x1,x2);
gx = density.gx(x1,x2); gy = density.gy(x1,x2);

MAS = uxx .* uyy - uxy.^2 - ufix - f./g;

Z = sparse(nx*ny,nx*ny);
fix = ceil((nx*ny)/2);
Z(:,fix) = 1;
JAM = spdiags(uxx,0,nx*ny,nx*ny)*M.D2YY + spdiags(uyy,0,nx*ny,nx*ny)*M.D2XX - ...
      2.*spdiags(uxy,0,nx*ny,nx*ny)*M.D2XY - Z;
  
temp_Jx = -gx.*f./(g.^2);temp_Jy = -gy.*f./(g.^2);
  
J = JAM - spdiags(temp_Jx,0,nx*ny,nx*ny)*M.D1XC - spdiags(temp_Jy,0,nx*ny,nx*ny)*M.D1YC; 
out = MAS; 

function [H,J] = findH(q,Hs,nj,M,Ibdy)
%Find the hamilton-jacobi function of the signed distance.
%Defined on page 7.
%Upwind Discretizations found on 9-10.

nx = M.nx; ny = M.ny; H = zeros(nx*ny,1);

% Forward and backward derivatives.
qxm = M.D1XM*q;qxp = M.D1XP*q; qym = M.D1YM*q;qyp = M.D1YP*q;

[H(Ibdy.All), iC] = max(max(nj(:,1),0)*qxm(Ibdy.All)' + min(nj(:,1),0)*qxp(Ibdy.All)'...
    + max(nj(:,2),0)*qym(Ibdy.All)' + min(nj(:,2),0)*qyp(Ibdy.All)' - repmat(Hs,[1,length(Ibdy.All)]) );

n1 = zeros(nx*ny,1); n2 = n1;
n1(Ibdy.All) = nj(iC,1);  n2(Ibdy.All) = nj(iC,2);

J = spdiags(max(n1,0),0,nx*ny,nx*ny)*(M.D1XM) + spdiags(min(n1,0),0,nx*ny,nx*ny)*(M.D1XP)+...
    spdiags(max(n2,0),0,nx*ny,nx*ny)*(M.D1YM) + spdiags(min(n2,0),0,nx*ny,nx*ny)*(M.D1YP);

 
 function [u0,error] = newton(initu,Hs,nj,Ibdy,M,errortol,numloops,X,density)
error = 10;
alpha0 = 1; %initial guess for alpha.

%Parameters for damped netwons iterations
tau = 0.1; sigma = 0.01; alphamin = 0.01; %Min alpha value.

loopcount = 0;
repeat = 1;
repeat2 = 1;
u0 = initu;

while repeat
    [F,Jac] = rhs(u0,Hs,nj,Ibdy,M,X,density);
    psi = Jac \ -F;
    g0 = 1/2*sum(psi.^2);
    
    if loopcount > 1
        uh = u0 + alpha0*psi;
        [fsh,~] = rhs(uh,Hs,nj,Ibdy,M,X,density);
        mut = 1/2*sum(prevpsi.^2).^(1/2);
        mub = 1/2*sum((psi-prevjac\fsh).^2).^(1/2);
        mu = mut/mub * alpha0;
        alpha0 = max(alphamin,min(mu,1));
    end
    
    while repeat2
        %Finding the optimal alpha.
        uh = u0 + alpha0*psi;
        [fsh,~] = rhs(uh,Hs,nj,Ibdy,M,X,density);
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
    u0 = uh;
    
    loopcount = loopcount + 1;
    repeat = ((error>errortol) && (loopcount < numloops));
end


