function fun_data_col(opt)

opt.nam_fld  = fullfile(pwd,num2str(opt.num_dsl,opt.num_led));
opt.dir_btc  = dir(fullfile(opt.nam_fld,'batch_*'));
bch_num      = length(opt.dir_btc);

for i0 = 1:length(opt.dir_btc)
    opt.dir_cnf = dir(fullfile(opt.nam_fld,opt.dir_btc(i0).name,'configure_*'));
    
    for i1 = 1:length(opt.dir_cnf)
        configure_file = fullfile(opt.dir_cnf(i1).folder, opt.dir_cnf(i1).name);
        try
            load(configure_file )
            if i0 == 1 && i1 == 1
                in_pc2_raw = nan(1,size(datain_pc2_raw,2));
                in_pc2_scl = nan(1,size(datain_pc2_scl,2));
                
                ou_pc2_raw = nan(1,size(dataou_pc2_raw,2));
                ou_pc2_scl = nan(1,size(dataou_pc2_scl,2));
                
                in_pc3_raw = nan(1,size(datain_pc3_raw,2));
                in_pc3_scl = nan(1,size(datain_pc3_scl,2));
                
                ou_pc3_raw = nan(1,size(dataou_pc3_raw,2));
                ou_pc3_scl = nan(1,size(dataou_pc3_scl,2));
                
            end
            
            in_pc2_raw = cat(1,in_pc2_raw,(datain_pc2_raw));
            in_pc2_scl = cat(1,in_pc2_scl,(datain_pc2_scl));
            
            ou_pc2_raw = cat(1,ou_pc2_raw,(dataou_pc2_raw));
            ou_pc2_scl = cat(1,ou_pc2_scl,(dataou_pc2_scl));
            
            in_pc3_raw = cat(1,in_pc3_raw,(datain_pc3_raw));
            in_pc3_scl = cat(1,in_pc3_scl,(datain_pc3_scl));
            
            ou_pc3_raw = cat(1,ou_pc3_raw,(dataou_pc3_raw));
            ou_pc3_scl = cat(1,ou_pc3_scl,(dataou_pc3_scl));
            
            disp(['OK | btch ' num2str(i0, opt.num_led) ' out of ' num2str(bch_num, opt.num_led) ' | ' ...
                'data ' num2str(i1, opt.num_led) ' out of ' num2str(length(opt.dir_cnf),opt.num_led)])
            
        catch
            
            
            disp(['ER | btch ' num2str(i0, opt.num_led) ' out of ' num2str(bch_num, opt.num_led) ' | ' ...
                'data ' num2str(i1, opt.num_led) ' out of ' num2str(length(opt.dir_cnf),opt.num_led)])
            
        end
    end
    
end

in_pc2_raw(1,:)     = [];
in_pc2_scl(1,:)     = [];

ou_pc2_raw(1,:)     = [];
ou_pc2_scl(1,:)     = [];

in_pc3_raw(1,:)     = [];
in_pc3_scl(1,:)     = [];

ou_pc3_raw(1,:)     = [];
ou_pc3_scl(1,:)     = [];

%% check uniqueness
[~,ind_pc3_unq_in] = unique(in_pc3_raw,'rows');
[~,ind_pc3_unq_ou] = unique(ou_pc3_raw,'rows');

ind_good           = unique([ind_pc3_unq_in;ind_pc3_unq_ou]);

in_pc2_raw         = in_pc2_raw(ind_good,:);
in_pc2_scl         = in_pc2_scl(ind_good,:) ;

ou_pc2_raw         = ou_pc2_raw(ind_good,:);
ou_pc2_scl         = ou_pc2_scl(ind_good,:);

in_pc3_raw         = in_pc3_raw(ind_good,:);
in_pc3_scl         = in_pc3_scl(ind_good,:);

ou_pc3_raw         = ou_pc3_raw(ind_good,:);
ou_pc3_scl         = ou_pc3_scl(ind_good,:);

%% save for matlab
save(fullfile(opt.nam_fld,'data_pc2_raw.mat'),'in_pc2_raw','ou_pc2_raw');
save(fullfile(opt.nam_fld,'data_pc2_scl.mat'),'in_pc2_scl','ou_pc2_scl');
save(fullfile(opt.nam_fld,'data_pc3_raw.mat'),'in_pc3_raw','ou_pc3_raw');
save(fullfile(opt.nam_fld,'data_pc3_scl.mat'),'in_pc3_scl','ou_pc3_scl');

%% save for python
% pcs 2 | raw
datain_list = fun_mat2list(in_pc2_raw);
dataou_list = fun_mat2list(ou_pc2_raw);

dict_val = append("{'datain': ",string(datain_list),", 'dataou': ", string(dataou_list), "}");
fid = fopen(fullfile(opt.nam_fld,'data_pc2_raw'),'wt');
fprintf(fid, dict_val);
fclose(fid);

% pcs 2 | scl
datain_list = fun_mat2list(in_pc2_scl);
dataou_list = fun_mat2list(ou_pc2_scl);

dict_val = append("{'datain': ",string(datain_list),", 'dataou': ", string(dataou_list), "}");
fid = fopen(fullfile(opt.nam_fld,'data_pc2_scl'),'wt');
fprintf(fid, dict_val);
fclose(fid);

% pcs 3 | raw
datain_list = fun_mat2list(in_pc3_raw);
dataou_list = fun_mat2list(ou_pc3_raw);

dict_val = append("{'datain': ",string(datain_list),", 'dataou': ", string(dataou_list), "}");
fid = fopen(fullfile(opt.nam_fld,'data_pc3_raw'),'wt');
fprintf(fid, dict_val);
fclose(fid);

% pcs 3 | scl
datain_list = fun_mat2list(in_pc3_scl);
dataou_list = fun_mat2list(ou_pc3_scl);

dict_val = append("{'datain': ",string(datain_list),", 'dataou': ", string(dataou_list), "}");
fid = fopen(fullfile(opt.nam_fld,'data_pc3_scl'),'wt');
fprintf(fid, dict_val);
fclose(fid);

disp(['        '])
disp('Batch Data Collection Has Been Completed!')
disp(['        '])

end






