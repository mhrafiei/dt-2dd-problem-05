% % A and R must be column vectors (no matrix); B is the angle between the
% % principle axis and the global axis 
% A and R are row vectors

% https://en.wikipedia.org/wiki/Second_moment_of_area#cite_note-6
% Hally, David (1987). Calculation of the Moments of Polygons (PDF) (Technical report). Canadian National Defense. Technical Memorandum 87/209.
% Steger, Carsten (1996). "On the Calculation of Arbitrary Moments of Polygons" (PDF).
% Soerjadi, Ir. R. "On the Computation of the Moments of a Polygon, with some Applications".

function B = fun_productmoment(A,R)

N       = length(A);

% % a       = nchoosek(1:N,N-1);b = perms(1:N);
% % ind     = ismember(b(:,1:N-1),a,'rows');
% % ind     = b(ind,1:N-1);
% 
% ind      = nan(N,N-1);
% ind_case = 1:N;
% for i0 = 1:N
%     ind(i0,:) = ind_case([1:i0-1,i0+1:end]);
% end
% 
% Z       = sum((R.^2) .* prod(exp(A(ind).*2*1i),2)  ,1);
% B2(1,1) = real(-0.5 * (1i * log(+ ((prod(exp(A.*1i),1) * ( sum(exp(A.*2*1i).*R.^2,1) * Z ).^(0.5) )/(Z)) )  ));
% B2(2,1) = real(-0.5 * (1i * log(- ((prod(exp(A.*1i),1) * ( sum(exp(A.*2*1i).*R.^2,1) * Z ).^(0.5) )/(Z)) )  ));
% 
% B       = sort(B2);
% % B       = (sort(wrapTo2Pi(B2)));
% % B       = round(B(1)*1000)/1000;
% 
% syms B
% x = R.*cos(A+B);
% y = R.*sin(A+B);
% 
% % Ixy = nan(N,1);
% 
% x = [x;x(1)];
% y = [y;y(1)];
% 
% for i0 = 1:N
%    Ixy(i0,1) = (x(i0)*y(i0+1) - x(i0+1)*y(i0) )*( x(i0)*y(i0+1) + 2*x(i0)*y(i0) + 2*x(i0+1)*y(i0+1)  + x(i0+1)*y(i0) );   
% end
% q = solve(sum(Ixy)==0,B);
% B2 = fun_wrap2halfpi(real(eval(q)));
% B = B2(1);
% 
% B = q;
% B = real(eval(q));
% B = B(1);

% comb_all = nchoosek(1:N,2);
% 
% R.^2.*exp(A(1,comb_all(end:-1:1,:)) .*2i) + R(comb_all(:,1)).*R(comb_all(:,2)).*exp(A)

%%

comb_all = nchoosek(1:N,2);

comb_normal = nan(N-1,2);

for i0 = 1:N-1
   ind = find(comb_all(:,1)==i0);
   comb_normal(i0,:) = comb_all(ind(1),:);
end

ind = find(comb_all(:,1)==1);
comb_abnormal = comb_all(ind(end),:);

comb_normal   = [comb_normal   ; comb_normal(:,end:-1:1)  ];
comb_abnormal = [comb_abnormal ; comb_abnormal(:,end:-1:1)];

comb_all      = [comb_normal;comb_abnormal];

comb_bin = zeros(size(comb_all));
for i0 = 1:N*2
    comb_bin(i0,comb_all(i0,:)) = 1; 
end
comb_bin = logical(comb_bin);


if N==3
    
end

if N>=4
    
    % part 1
    
    mult_val_p1.sign_normal_orange   = sign([-ones(N-1,1);+ones(N-1,1)]);
    mult_val_p1.sign_abnormal_orange = sign([+1;-1]);
    
    mult_val_p1.sign_normal_green   = sign([+ones(N-1,1);-ones(N-1,1)]);
    mult_val_p1.sign_abnormal_green = sign([-1;+1]);
    
    mult_val_p1.sign_normal_red     = sign([ones(N-1,1);-ones(N-1,1)]);
    mult_val_p1.sign_abnormal_red   = sign([-1;+1]);
    
    mult_val_p1.rpower_orange       = repmat([3,1],2*(N),1);
    mult_val_p1.rpower_green        = repmat([3,1],2*(N),1);
    mult_val_p1.rpower_red          = repmat([2,2],2*(N),1);
    
    mult_val_p1.angle_imlt_orange   = ones(N*2,N)*3i;
    mult_val_p1.angle_imlt_green    = ones(N*2,N)*3i;
    mult_val_p1.angle_imlt_red      = ones(N*2,N)*3i;
    for i0 = 1:2*N
        mult_val_p1.angle_imlt_orange(i0,comb_all(i0,:)) = [2i,2i];
        mult_val_p1.angle_imlt_green(i0,comb_all(i0,:))  = [0i,4i];
        mult_val_p1.angle_imlt_red(i0,comb_all(i0,:))    = [1i,3i];
    end
    

    p1_orange = sum( ([mult_val_p1.sign_normal_orange;mult_val_p1.sign_abnormal_orange]) .*( prod(R(comb_all).^mult_val_p1.rpower_orange,2) .* exp(sum(mult_val_p1.angle_imlt_orange .* repmat(A',2*N,1),2)) ),1);
    p1_green  = sum( ([mult_val_p1.sign_normal_green;mult_val_p1.sign_abnormal_green])   .*( prod(R(comb_all).^mult_val_p1.rpower_green,2)  .* exp(sum(mult_val_p1.angle_imlt_green  .* repmat(A',2*N,1),2)) ),1);
    p1_red    = sum( ([mult_val_p1.sign_normal_red;mult_val_p1.sign_abnormal_red])       .*( prod(R(comb_all).^mult_val_p1.rpower_red,2)    .* exp(sum(mult_val_p1.angle_imlt_red    .* repmat(A',2*N,1),2)) ),1);
    
    p1        = p1_orange + p1_green + p1_red;
    
    % part 2
    
    mult_val_p2.sign_normal_orange   = sign([-ones(N-1,1);+ones(N-1,1)]);
    mult_val_p2.sign_abnormal_orange = sign([+1;-1]);
    
    mult_val_p2.sign_normal_green   = sign([+ones(N-1,1);-ones(N-1,1)]);
    mult_val_p2.sign_abnormal_green = sign([-1;+1]);
    
    mult_val_p2.sign_normal_red     = sign([-ones(N-1,1);+ones(N-1,1)]);
    mult_val_p2.sign_abnormal_red   = sign([+1;-1]);
    
    mult_val_p2.rpower_orange       = repmat([3,1],2*(N),1);
    mult_val_p2.rpower_green        = repmat([3,1],2*(N),1);
    mult_val_p2.rpower_red          = repmat([2,2],2*(N),1);
    
    mult_val_p2.rpower_orange       = repmat([3,1],2*(N),1);
    mult_val_p2.rpower_green        = repmat([3,1],2*(N),1);
    mult_val_p2.rpower_red          = repmat([2,2],2*(N),1);
    
    mult_val_p2.angle_imlt_orange   = ones(N*2,N)*3i;
    mult_val_p2.angle_imlt_green    = ones(N*2,N)*3i;
    mult_val_p2.angle_imlt_red      = ones(N*2,N)*3i;
    for i0 = 1:2*N
        mult_val_p2.angle_imlt_orange(i0,comb_all(i0,:)) = [6i,2i];
        mult_val_p2.angle_imlt_green(i0,comb_all(i0,:))  = [4i,4i];
        mult_val_p2.angle_imlt_red(i0,comb_all(i0,:))    = [5i,3i];
    end
    

    p2_orange = sum( ([mult_val_p2.sign_normal_orange;mult_val_p2.sign_abnormal_orange]) .*( prod(R(comb_all).^mult_val_p2.rpower_orange,2) .* exp(sum(mult_val_p2.angle_imlt_orange .* repmat(A',2*N,1),2)) ),1);
    p2_green  = sum( ([mult_val_p2.sign_normal_green;mult_val_p2.sign_abnormal_green])   .*( prod(R(comb_all).^mult_val_p2.rpower_green,2)  .* exp(sum(mult_val_p2.angle_imlt_green  .* repmat(A',2*N,1),2)) ),1);
    p2_red    = sum( ([mult_val_p2.sign_normal_red;mult_val_p2.sign_abnormal_red])       .*( prod(R(comb_all).^mult_val_p2.rpower_red,2)    .* exp(sum(mult_val_p2.angle_imlt_red    .* repmat(A',2*N,1),2)) ),1);
    
    p2        = p2_orange + p2_green + p2_red;
    
    B         = fun_wrap2halfpi(real(-(log(  p1/p2 ) * 1i)/4));
    
end

end

% syms B
% x = R.*cos(A+B);
% y = R.*sin(A+B);
% 
% % Ixy = nan(N,1);
% 
% x = [x;x(1)];
% y = [y;y(1)];
% 
% for i0 = 1:N
%    Ixy(i0,1) = (x(i0)*y(i0+1) - x(i0+1)*y(i0) )*( x(i0)*y(i0+1) + 2*x(i0)*y(i0) + 2*x(i0+1)*y(i0+1)  + x(i0+1)*y(i0) );   
% end
% q = solve(sum(Ixy)==0,B);
% % B = fun_wrap2halfpi(real(eval(q)));
% % B = B2(1);
% % 
% B = q;

