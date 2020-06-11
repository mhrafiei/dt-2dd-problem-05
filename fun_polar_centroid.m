function [ac,rc] = fun_polar_centroid(A,R)

[X,Y]   = pol2cart(A,R);

x       = mean(X,2);
y       = mean(Y,2);

[ac,rc]  = cart2pol(x,y);
ac(rc<0) = ac(rc<0)- pi;
rc       = abs(rc);
ac       = wrapTo2Pi(ac);


end

