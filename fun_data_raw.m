function fun_data_raw(opt)

% %% parameters
% opt.num_led        = '%010d';
% opt.num_cnf        = 100;
% opt.sze_btc        = 12;
% opt.num_dsl        = 5;
% 
% opt.dst_mag        = 3*.249e-9;
% opt.cll_res        = 5e-6;
% opt.cll_rad        = (sqrt(2)/4)*opt.cll_res*2;
% opt.frg_lbd        = -1.3387e+09;
% opt.frg_ubd        = +1.3387e+09;
% 
% opt.nam_fld        = fullfile(pwd,num2str(opt.num_dsl,opt.num_led));
% opt.nam_tim        = datetime;
% opt.nam_tim.Format = 'yyyy_MM_dd_HH_mm_ss_SSS';
% opt.nam_sfd        = fullfile(pwd,num2str(opt.num_dsl,opt.num_led),['batch_',char(opt.nam_tim)]);
% 
% opt.gna_lbd        = zeros(1,opt.num_dsl);
% opt.gna_ubd        = ones(1, opt.num_dsl)*2*pi;
% opt.gna_Avl        = [];
% opt.gna_bvl        = [];
% opt.gna_Aeq        = [];
% opt.gna_beq        = [];
% opt.gna_bin        = 10;
% opt.gna_mkv        = 1000; % it must be at least as much as the PopulationSize
% opt.gna_pns        = 20;
% opt.gna_gns        = 20;
% opt.gna_opt        = optimoptions('ga','ConstraintTolerance',1e-128, ...
%                                   'Display','off', ...
%                                   'FunctionTolerance',1e-128, ...
%                                   'PopulationSize',opt.gna_pns, ...
%                                   'UseParallel',false, ...
%                                   'MaxGenerations',opt.gna_gns); %'InitialPopulationMatrix',x0, ...
% 
% %% functions
% opt.fun_scl        = @(x,lb,ub) ((x-(ones(size(x,1),1)*min(x)))./ ...
%                                 ((ones(size(x,1),1)*max(x))- ...
%                                  (ones(size(x,1),1)* ...
%                                   min(x))))*(ub-lb)+lb;
% 
% % a, r, and g are for target in pcs2 and t is for the base
% opt.fun_fgd        = @(a,r,g,t) (cos(2*g).*(cos(3*a - 2*t) + ...
%                                  cos(a - 2*t)))./(2*r) + ...
%                                 (4*cos(2*t).*cos(a).^2.*cos(g).* ...
%                                  sin(a).*sin(g))./r;
% 
% opt.fun_fgi        = @(a,r,g,t) ((cos(2*t).*(cos(3*a - 2*g) + ...
%                                   cos(a - 2*g)))./2 + ...
%                                   4*cos(2*g).*cos(a).^2.*sin(a).*cos(t).* ...
%                                   sin(t))./r;

%% restart random generator
rng('shuffle')    ;
s = rng;

%% create directories
if exist(opt.nam_fld,'dir') == 0
    mkdir(opt.nam_fld)
end
if exist(opt.nam_sfd,'dir') == 0
    mkdir(opt.nam_sfd)
end

%% batch informain
opt.num_btc                = ceil(opt.num_cnf/opt.sze_btc);
opt.ind_btc                = nan(1,opt.num_btc*opt.sze_btc);
opt.ind_btc(1:opt.num_cnf) = 1:opt.num_cnf;
opt.ind_btc                = reshape(opt.ind_btc, opt.sze_btc, opt.num_btc);

%% dislocation indexing
opt                        = fun_pdf(opt);

%% all possible points in hegzagone
[opt.azm_hgz,opt.rad_hgz, opt.sze_hxg] = fun_hexagon(opt);

%% radius lower and upper bounds in pcs 2
opt.pc2_rlb        = opt.dst_mag;
opt.pc2_rub        = 2 * (opt.sze_hxg * opt.dst_mag);

%% radius lower and upper bounds in pcs 3
opt.pc3_rlb        = 0;
opt.pc3_rub        = opt.sze_hxg * opt.dst_mag + opt.dst_mag;

%% batch loop
parfor i0 = 1:opt.num_btc
    
    fun_ass_01(opt,i0)
    
end

disp(['        '])
disp('Batch Configurations Have Been Completed!')
disp(['        '])

end

%-----------
%% functions 
%-----------
function fun_ass_01(opt,i0)
%% restart random generator
rng('shuffle')    ;
s = rng;

%% modify batcg index
bch_num      = size(opt.ind_btc,2);
opt.ind_btc  = opt.ind_btc(~isnan(opt.ind_btc(:,i0)),i0);

%% generate each dtapoint in the batch
pc3_azm = nan(length(opt.ind_btc),opt.num_dsl);
pc3_rad = nan(length(opt.ind_btc),opt.num_dsl);
pc3_rot = nan(length(opt.ind_btc),opt.num_dsl);
pc2_frc = nan(length(opt.ind_btc),opt.num_dsl * (opt.num_dsl - 1));
pc3_frc = nan(length(opt.ind_btc),opt.num_dsl);
fit_val = nan(length(opt.ind_btc), 1);

for i1 = 1:length(opt.ind_btc)
    % randomly select opt.num_dsl number points from hexagon and add rand
    % to all azimuth
    ind_points    = randperm(length(opt.azm_hgz),opt.num_dsl);
    a             = wrapTo2Pi(opt.azm_hgz(1,ind_points) + rand*2*pi);
    r             = opt.rad_hgz(1,ind_points);
    
    % transformation (centeralized and product moment transformed)
    [pc3_azm(i1,:), pc3_rad(i1,:), B(i1,1)] = fun_transform(a,r);
    
    % use markove chain for the rotation's initial point
    pc3_rot0      = fun_mkv(opt.fun_fgd, ...
        opt.gna_pns, ...
        opt.gna_mkv, ...
        opt.num_dsl, ...
        opt.ind_dsl, ...
        opt.gna_bin, ...
        pc3_azm(i1,:), ...
        pc3_rad(i1,:));
    
    
    % fitness function for ga
    fun = @(t)fun_fit(opt.fun_fgd, ...
        opt.ind_dsl, ...
        opt.gna_bin, ...
        pc3_azm(i1,:), ...
        pc3_rad(i1,:), ...
        t);
    
    % genetic optimization
    opt.gna_opt.InitialPopulationMatrix = pc3_rot0;
    [pc3_rot(i1,:),fit_val(i1,1)] = ga(fun, ...
        opt.num_dsl, ...
        opt.gna_Avl, ...
        opt.gna_bvl, ...
        opt.gna_Aeq, ...
        opt.gna_beq, ...
        opt.gna_lbd, ...
        opt.gna_ubd, ...
        [], ...
        opt.gna_opt);
    
    % round to closes slip system
      pc3_rot(i1,:) = fun_slip(pc3_rot(i1,:));
    
    % compute forces in pcs 2 and 3
    [pc2_frc(i1,:),pc3_frc(i1,:)] = fun_frc(opt.fun_fgd, ...
        opt.ind_dsl, ...
        pc3_azm(i1,:), ...
        pc3_rad(i1,:), ...
        pc3_rot(i1,:));
    
    disp(['btch ' num2str(i0, opt.num_led) ' out of ' num2str(bch_num, opt.num_led) ' | ' ...
          'data ' num2str(i1, opt.num_led) ' out of ' num2str(length(opt.ind_btc),opt.num_led)])
    %fun_figrot(pc3_azm(i1,:), pc3_rad(i1,:), pc3_rot(i1,:), opt.sze_hxg * opt.dst_mag  )
end

%% save data of the batch
nam_tim        = datetime;
nam_tim.Format = 'yyyy_MM_dd_HH_mm_ss_SSS';
file_name      = fullfile(opt.nam_sfd ,['configure_', char(nam_tim), '.mat']);

% pcs3 format | raw & scl
datain_pc3_raw = [pc3_azm, pc3_rad, pc3_rot];
dataou_pc3_raw = pc3_frc;

h_val = opt.fun_scl([opt.pc3_rlb;opt.pc3_rub;pc3_rad(:)],0,1);
datain_pc3_scl = [pc3_azm./2*pi, reshape(h_val(3:end),size(pc3_rad)), pc3_rot./2*pi];
dataou_pc3_scl = pc3_frc; % should not be scaled untill all is availble

order_val      = ['azm','rad','rot'];

% pcs2 format | raw & scl
datain_pc2_raw = zeros(length(opt.ind_btc),opt.num_dsl,opt.num_dsl,4);
datain_pc2_scl = zeros(length(opt.ind_btc),opt.num_dsl,opt.num_dsl,4);
pc2_azm_tgd    = nan(length(opt.ind_btc),opt.num_dsl*(opt.num_dsl-1));
pc2_rad_tgd    = nan(length(opt.ind_btc),opt.num_dsl*(opt.num_dsl-1));
pc2_rot_tgd    = nan(length(opt.ind_btc),opt.num_dsl*(opt.num_dsl-1));
pc2_rot_bse    = nan(length(opt.ind_btc),opt.num_dsl*(opt.num_dsl-1));

for i1 = 1:length(opt.ind_btc)
    [pc2_azm_tgd(i1,:), ...
     pc2_rad_tgd(i1,:), ...
     pc2_rot_tgd(i1,:), ...
     pc2_rot_bse(i1,:)] = fun_pc3to2(opt.ind_dsl, ...
                                     pc3_azm(i1,:), ...
                                     pc3_rad(i1,:), ...
                                     pc3_rot(i1,:));
                                 
end

datain_pc2_raw = [pc2_azm_tgd, pc2_rad_tgd, pc2_rot_tgd, pc2_rot_bse];
h_val          = opt.fun_scl([opt.pc2_rlb;opt.pc2_rub;pc2_rad_tgd(:)],0,1);
datain_pc2_scl = [pc2_azm_tgd./(2*pi), reshape(h_val(3:end),size(pc2_rad_tgd)),pc2_rot_tgd./(2*pi), pc2_rot_bse./(2*pi)];

dataou_pc2_raw = pc2_frc;
dataou_pc2_scl = pc2_frc./opt.frg_ubd;

save(file_name,'datain_pc3_raw', ...
               'dataou_pc3_raw', ...
               'datain_pc3_scl', ...
               'dataou_pc3_scl', ...
               'datain_pc2_raw', ...
               'dataou_pc2_raw', ...
               'datain_pc2_scl', ...
               'dataou_pc2_scl', ...
               '-v7.3')


end

%% index calculation for dislocations 
function opt = fun_pdf(opt)

[x,y]       = meshgrid(1:opt.num_dsl,1:opt.num_dsl);
x           = triu(x).*(1-eye(opt.num_dsl));
y           = triu(y).*(1-eye(opt.num_dsl));
opt.ind_dsl = [[y(y~=0),x(x~=0)];[x(x~=0),y(y~=0)]];

end

%% all possible points in the hexagon
function [a,r,m] = fun_hexagon(opt)

m = 1;
c = 1;
while true
    c = c + 6*m;
    if c >= opt.num_dsl
        break
    end
    m = m + 1;
end

a = [0];
r = [0];

for i0 = 1:m
    a0 = 0:2*pi/(6*i0):2*pi-2*pi/(6*i0);
    a  = [a,a0];
    r  = [r,ones(1,length(a0))*opt.dst_mag*i0];
end

end

%% fitness function
function fit_val = fun_fit(fun_fgd,ind,bin,a,r,t)

r_glb_base = r(ind(:,1));
a_glb_base = a(ind(:,1));
r_glb_trgd = r(ind(:,2));
a_glb_trgd = a(ind(:,2));

[r_loc_trgd,a_loc_trgd] = fun_polar_glob2loc(r_glb_base, ...
    a_glb_base, ...
    r_glb_trgd, ...
    a_glb_trgd);
t_loc_base = t(ind(:,1));
t_loc_trgd = t(ind(:,2));

f       = fun_fgd(a_loc_trgd, r_loc_trgd, t_loc_trgd, t_loc_base) ;

fit_val = std(hist(f,bin));

end

%% force calculations in pcs 2 and 3
function [f_pcs2, f_pcs3] = fun_frc(fun_fgd,ind,a,r,t)
r_glb_base = r(ind(:,1));
a_glb_base = a(ind(:,1));
r_glb_trgd = r(ind(:,2));
a_glb_trgd = a(ind(:,2));

[r_loc_trgd,a_loc_trgd] = fun_polar_glob2loc(r_glb_base, ...
    a_glb_base, ...
    r_glb_trgd, ...
    a_glb_trgd);
t_loc_base = t(ind(:,1));
t_loc_trgd = t(ind(:,2));

f_pcs2       = fun_fgd(a_loc_trgd, r_loc_trgd, t_loc_trgd, t_loc_base) ;

f_pcs3       = nan(1,length(a));
for i0 = 1:length(a)
    
    ind_cse      = find(ind(:,1) == i0 | ind(:,2) == i0);
    f_pcs3(1,i0) = sum(f_pcs2(ind_cse));
    
end

end

%% markov chain
function pc3_rot = fun_mkv(fun_fgd,pop,mkv,n,ind,bin,a,r)

fit_val     = nan(mkv,1);
pc3_rot_mkv = nan(mkv,n);
for i2 = 1:mkv
    t                 = rand(1,n)*2*pi;
    pc3_rot_mkv(i2,:) = t;
    fit_val(i2,1)     = fun_fit(fun_fgd,ind,bin,a,r,t);
end

[fit_val,ind] = sort(fit_val);
pc3_rot       = pc3_rot_mkv(ind(1:pop),:);

end

%% transformation and centeralization 
function [a, r, B] = fun_transform(a,r)


% get the centroids of the main config
[a_lc,r_lc]      = fun_polar_centroid(a,r);

% transform the main config to the centroid
[r_lic0,a_lic0i] = fun_polar_glob2loc(r_lc,a_lc,r,a);

% wrap the transformed angles of the main config
a_lic0           = wrapTo2Pi(a_lic0i);

% % compute the angle between principle axis and the local axis
B                = fun_productmoment(a_lic0',r_lic0');

% shift
a                = wrapTo2Pi(a_lic0    + repmat(B,1,length(a)));
r                = r_lic0;

end

%% round to closest slip system 
function t = fun_slip(t)

% 6 slip + 6 partials
round_vals = [[0,1,2,3,4,5,6]  * pi/3,[1,3,5,7,9,11] * pi/6];
round_asgn = [[0,1,2,3,4,5,0]  * pi/3,[1,3,5,7,9,11] * pi/6];
t_temp = nan(13,length(t));
for i2 = 1:13
    t_temp(i2,:) = abs(t - round_vals(i2));
end

[min_temp, ind_temp] = min(t_temp);
t                    = round_asgn(ind_temp);


end

%% convert pcs 3 to pcs 2
function [a_trgd, r_trgd, t_trgd, t_base] = fun_pc3to2(ind,a,r,t)

r_glb_base = r(ind(:,1));
a_glb_base = a(ind(:,1));
r_glb_trgd = r(ind(:,2));
a_glb_trgd = a(ind(:,2));

[r_trgd,a_trgd] = fun_polar_glob2loc(r_glb_base, ...
    a_glb_base, ...
    r_glb_trgd, ...
    a_glb_trgd);
t_base = t(ind(:,1));
t_trgd = t(ind(:,2));

% control negetives
ind_neg = find(r_trgd <0);
a_trgd(ind_neg) = wrapTo2Pi(a_trgd(ind_neg) + pi);
r_trgd(ind_neg) = abs(r_trgd(ind_neg));
a_trgd          = wrapTo2Pi(a_trgd);

end





