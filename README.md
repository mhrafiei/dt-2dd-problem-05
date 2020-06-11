# Creat machine learning data repository for different number of dislocations using MARCC parallel supercomputing facilities

```diff
! READ FIRST! 
```

## 1. On MARCC, clone the repository:
>* git config --global --unset http.proxy
>* git config --global --unset https.proxy
>* git config --global user.email "mrafiei1@jhu.edu"
>* git config --global user.name "mrafiei1"
>* git clone https://github.com/mhrafiei/dt-2dd-problem-05.git

## 2. Edit master_creator_20200504.m for random locations or master_creator_20200511.m for 6 slip systems and 6 partials, and define different values for disl_num in a list (e.g. data repository for disl_num[0], disl_num[1], ...):
>* cd dt-2dd-problem-05/
>* module load matlab
>* matlab -nodisplay -nosplash -nodesktop -r 'master_creator_20200511;'
>* exit

## 3. Submit first round of jobs on MARCC:
>* bash ja_sub_1to1.sh

## 4. Once completed, submit second round of jobs on MARCC: 
>* bash ja_sub_2to5.sh

## 5. For a particular dislocation number, say 10, copy final .mat and .txt data to the data-matlab folder under ml-2dd-problem-05:
>* cp ~/dt-2dd-problem-05/0000000010/results/data_* ~/ml-2dd-problem-05/data-matlab/

## 6. Run code_data.py (check the ReadMe corresponding to ml-2dd-problem-05)
