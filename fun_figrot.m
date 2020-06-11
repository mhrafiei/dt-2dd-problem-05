function fun_figrot(a,r,t,max_val)
leglag     = '%010d';

[x,y] = pol2cart(a,r);
x = x';
y = y';
a = a';
r = r';

n = length(a);

fig = figure('Name','progress','units','normalized','outerposition',[0 0 1 1]);
polarplot(a,r,'.r','markersize',30)
hold on

set(gca,'fontname','times','fontsize',30)

ind = nchoosek(1:n,2);

d   = floor(fun_norm([x(ind(:,1))-x(ind(:,2)),y(ind(:,1))-y(ind(:,2))])*10^10);

ind_min = find(d <12);

ind = ind(ind_min,:);

for i0 = 1:size(ind,1)
    
    polarplot([a(ind(i0,1));a(ind(i0,2))],[r(ind(i0,1));r(ind(i0,2))],'--r','linewidth',1)
    
end

r0 = 1.5*10^-10;

round_vals = unique(t(:));
round_asgn = [[0,1,2,3,4,5]  * pi/3,[1,3,5,7,9,11] * pi/6];
color_vals = ['r-';'k-';'b-';'g-';'m-';'c-';'r:';'k:';'b:';'g:';'m:';'c:'];
line_vals  = [1,1,1,1,1,1,2,2,2,2,2,2];

for i0 = 1:n
    t0 = t(i0);
    t1 = t0;
    t2 = t0+pi/2;
    t3 = t0+pi;
    r1 = r0;
    r2 = 2*r0;
    r3 = r0;
    
    [r1,a1]=fun_polar_loc2glob(r(i0),a(i0),r1,t1);
    [r2,a2]=fun_polar_loc2glob(r(i0),a(i0),r2,t2);
    [r3,a3]=fun_polar_loc2glob(r(i0),a(i0),r3,t3);
    
    ind_color = find(t(i0)==round_asgn);
    polarplot([a(i0);a1],[r(i0);r1],'r','linewidth',2)
    polarplot([a(i0);a2],[r(i0);r2],'r','linewidth',2)
    polarplot([a(i0);a3],[r(i0);r3],'r','linewidth',2)

end

% leg_vals = {};
% for i0 = 1:length(round_vals)
%     leg_vals(i0) = {num2str(round(rad2deg(round_vals(i0))))};
% end
% 
% legend(leg_vals)

R = max(r);

set(gca,'fontname','times','fontsize',40)
pax = gca;
pax.ThetaAxisUnits = 'radians';


rticks([0 R/2 R])
rticklabels({' ',' ',' '})

ttt = datetime;
ttt.Format = 'dd-MMM-uuuu-HH-mm-ss-ms';

if ~isnan(max_val)
    rlim([0,max_val*1.2])
end

folder_name = fullfile(pwd,[num2str(length(a),leglag)],'figures');
if exist(folder_name,'dir')==0
    mkdir(folder_name)
end

print(gcf, '-dtiff', '-r100', fullfile(folder_name,['fig3e_' num2str(length(a)) '_' char(ttt) '.tif']));

end

function y = fun_norm(x)

y = sqrt(sum(x.^2,2));

end