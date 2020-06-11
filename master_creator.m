clc
clear
close all
warning off

num_cnf  = 100000;
sze_btc  = 100;
num_dsl  = [15,20,25,30,40,45,50];

leglag   = '%010d';

delete('mas_*')
delete('job_*')

job_sub_raw = [];
job_sub_col = [];

for i1 = 1:2 % raw and col
    
    for i0 = 1:length(num_dsl)
        
        if i1 == 1
            script_val = [fileread('info_mas_raw_marcc.txt'), ...
                newline, ['%%%%%%%%%%%%%%%%%%%%%%%%%'], ...
                newline, ['opt.num_cnf        = ' num2str(num_cnf) ';'], ...
                newline, ['opt.sze_btc        = ' num2str(sze_btc) ';'], ...
                newline, ['opt.num_dsl        = ' num2str(num_dsl(i0)) ';'], ...
                newline, fileread('info_opt.txt'), ...
                newline, 'fun_data_raw(opt);'];
            
            Fun_Mcreate(string(script_val),['mas_raw_'  num2str(num_dsl(i0),leglag)])
            
            job_script = string('#!/bin/bash') + newline + ...
                string(['#SBATCH --job-name=raw_' num2str(num_dsl(i0))]) + newline + ...
                string(fileread('info_job_raw.txt')) + newline + ...
                ["matlab -nodisplay -nosplash -nodesktop -r '" + ...
                ['mas_raw_'  num2str(num_dsl(i0),leglag)] + ";'"];
            
            Fun_Bcreate(job_script,['job_raw_'  num2str(num_dsl(i0),leglag)])
            job_sub_raw = [job_sub_raw, newline, 'sbatch ', ['job_raw_'  num2str(num_dsl(i0),leglag) '.sh;']];
        else
            script_val = [fileread('info_mas_col_marcc.txt'), ...
                newline, ['%%%%%%%%%%%%%%%%%%%%%%%%%'], ...
                newline, ['opt.num_cnf        = ' num2str(num_cnf) ';'], ...
                newline, ['opt.sze_btc        = ' num2str(sze_btc) ';'], ...
                newline, ['opt.num_dsl        = ' num2str(num_dsl(i0)) ';'], ...
                newline, fileread('info_opt.txt'), ...
                newline, 'fun_data_col(opt);'];
            
            Fun_Mcreate(string(script_val),['mas_col_'  num2str(num_dsl(i0),leglag)])
            
            job_script = string('#!/bin/bash') + newline + ...
                string(['#SBATCH --job-name=col_' num2str(num_dsl(i0))]) + newline + ...
                string(fileread('info_job_col.txt')) + newline + ...
                ["matlab -nodisplay -nosplash -nodesktop -r '" + ...
                ['mas_col_'  num2str(num_dsl(i0),leglag)] + ";'"];
            
            Fun_Bcreate(job_script,['job_col_'  num2str(num_dsl(i0),leglag)])
            job_sub_col = [job_sub_col, newline, 'sbatch ', ['job_col_'  num2str(num_dsl(i0),leglag) '.sh;']];
        end

    end
end

Fun_Bcreate(job_sub_raw,'job_sub_raw')
Fun_Bcreate(job_sub_col,'job_sub_col')



% This function prints whatever is written in SCRIPT into  NAME.m and save
% it in the directory automaticaly

function Fun_Mcreate(SCRIPT,NAME)  %and use variable names that have meaning
NAME = sprintf('%s.m', NAME);
fid = fopen(NAME, 'wt'); %and use 't' with text files so eol are properly translated
%fprintf(fid,  SCRIPT);
fwrite(fid, SCRIPT);
fclose(fid);
% edit(NAME);

end

% SBATCH creator
function Fun_Bcreate(SCRIPT,NAME)
NAME = sprintf('%s.sh', NAME);
fid = fopen(NAME, 'wt'); %and use 't' with text files so eol are properly translated
%fprintf(fid,  SCRIPT);
fwrite(fid, SCRIPT);
fclose(fid);
% edit(NAME);
end
