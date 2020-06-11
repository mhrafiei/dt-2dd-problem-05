clc
clear
close all

%% parameters

opt.num_cnf        = 1000;
opt.sze_btc        = 10;
opt.num_dsl        = 4;

opt.num_led        = '%010d';

opt.dst_mag        = 3*.249e-9;
opt.cll_res        = 5e-6;
opt.cll_rad        = (sqrt(2)/4)*opt.cll_res*2;
opt.frg_lbd        = -1.3387e+09;
opt.frg_ubd        = +1.3387e+09;

opt.nam_fld        = fullfile(pwd,num2str(opt.num_dsl,opt.num_led));
opt.nam_tim        = datetime;
opt.nam_tim.Format = 'yyyy_MM_dd_HH_mm_ss_SSS';
opt.nam_sfd        = fullfile(pwd,num2str(opt.num_dsl,opt.num_led),['batch_',char(opt.nam_tim)]);

opt.gna_lbd        = zeros(1,opt.num_dsl);
opt.gna_ubd        = ones(1, opt.num_dsl)*2*pi;
opt.gna_Avl        = [];
opt.gna_bvl        = [];
opt.gna_Aeq        = [];
opt.gna_beq        = [];
opt.gna_bin        = 10;
opt.gna_mkv        = 1000; % it must be at least as much as the PopulationSize
opt.gna_pns        = 20;
opt.gna_gns        = 20;
opt.gna_opt        = optimoptions('ga','ConstraintTolerance',1e-128, ...
                                  'Display','off', ...
                                  'FunctionTolerance',1e-128, ...
                                  'PopulationSize',opt.gna_pns, ...
                                  'UseParallel',false, ...
                                  'MaxGenerations',opt.gna_gns); %'InitialPopulationMatrix',x0, ...

%% functions
opt.fun_scl        = @(x,lb,ub) ((x-(ones(size(x,1),1)*min(x)))./ ...
                                ((ones(size(x,1),1)*max(x))- ...
                                 (ones(size(x,1),1)* ...
                                  min(x))))*(ub-lb)+lb;

% a, r, and g are for target in pcs2 and t is for the base
opt.fun_fgd        = @(a,r,g,t) (cos(2*g).*(cos(3*a - 2*t) + ...
                                 cos(a - 2*t)))./(2*r) + ...
                                (4*cos(2*t).*cos(a).^2.*cos(g).* ...
                                 sin(a).*sin(g))./r;

opt.fun_fgi        = @(a,r,g,t) ((cos(2*t).*(cos(3*a - 2*g) + ...
                                  cos(a - 2*g)))./2 + ...
                                  4*cos(2*g).*cos(a).^2.*sin(a).*cos(t).* ...
                                  sin(t))./r;
%% main functions to run
% fun_data_raw(opt)
fun_data_col(opt)

