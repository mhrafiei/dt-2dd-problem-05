% transform angles A between 0 and pi/2 

function A = fun_wrap2halfpi(A)

A = wrapTo2Pi(A);

% condition 1 
ind    = A>pi/2 & A<=pi;
A(ind) = A(ind) - pi/2;

% condition 2 
ind    = A>pi   & A<=3*pi/2;
A(ind) = A(ind) - pi;

% condition 3 
ind    = A>3*pi/2   & A<=2*pi;
A(ind) = A(ind) - 3*pi/2;



end