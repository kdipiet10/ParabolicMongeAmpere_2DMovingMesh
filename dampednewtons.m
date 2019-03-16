function [v,error] = dampednewtons(fun,jac,initguess,errortol,numloops)

%General verson of damped newtons method. Need to change the function and
%the derivative dependent on the problem. 

%input:
%fun and jac should be function handles describing the function and it's 
%jacobian. (take in just a single vector)
%initv - the initial guess for Newtons method. 
%x discritized space for v. 
%errortol - the residual error tolerance. 
%numloops - maximum number of loop iterations. 

%output - the final v vector 
%error - the final error

lambda0 = 1; %initial guess for lambda 

tau = 0.1;  
sigma = 0.01; 
lambdamin = 0.01; %minimum lambda. 

loopcount = 0; 
repeat = 1;  
repeat2 = 1; 
error = 0; 
v = initguess; 
                                         
while repeat
    %Solve for psi given an initial guess v 
    ff = fun(v);
    jacobian = jac(v); 
    psi = jacobian \ -ff;
    g0 = 1/2*sum(psi.^2); 
    %Predicting lambda. 
    if loopcount > 1 
        %finding the best intial lambda based on the previous. 
        sh = v + lambda0*psi; 
        fsh = fun(sh); 
        mut = 1/2*sum(prevpsi.^2).^(1/2);
        mub = 1/2*sum((psi-prevjac\fsh).^2).^(1/2); 
        mu = mut/mub * lambda0;
        lambda0 = max(lambdamin,min(1,mu)); 
    end 
    while repeat2
        sh = v + lambda0*psi;
        fsh = fun(sh); 
        gl = 1/2*sum((jacobian\fsh).^2); 
        if gl <= (1-2*lambda0*sigma)*g0
            break; 
        elseif lambda0 < lambdamin
            break; 
        else
            lambda0 = max(tau*lambda0,(lambda0^2*g0)/((2*lambda0-1)*g0+gl));
        end
    end 
    if lambda0 < lambdamin
        %disp('no convergence')
        break;
    end 
    prevpsi = psi; 
    prevjac = jacobian; 
    error = max(abs(ff)); 
    v = sh; 

    loopcount = loopcount + 1; 
    repeat = ((error>errortol) && (loopcount < numloops)); 
end 