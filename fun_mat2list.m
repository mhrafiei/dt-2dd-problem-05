function list_val = fun_mat2list(mat_mlt)

L = size(mat_mlt);
if length(L) == 2
    list_val = fun_ass_01(mat_mlt) ;
end

if length(L) == 4
    
    for i0 = 1: L(1)
        
        mat_case1 = squeeze(mat_mlt(i0,:,:,:));
        
        for i1 = 1:L(2)
            mat_case2 = squeeze(mat_case1(i1,:,:));
            l_val1(i1,:) = fun_ass_01(mat_case2) ;
        end
        l_val1(:,end+1) = ',';
        l_val2(i0,:) = ['[',reshape(l_val1',1,size(l_val1,1)*size(l_val1,2)),'],'];
        clear l_val1
        disp([num2str(i0,'%010d') ' out of' num2str(L(1)) ' | ' num2str(round(i0/L(1) * 100,2)) '%'])
    end
    list_val = ['[',reshape(l_val2',1,size(l_val2,1)*size(l_val2,2)),']'];
end


end

function list_val = fun_ass_01(mat)
l           = size(mat);

a           = string(num2str(mat,'%+5.5f\\'));
ind         = strfind(a,'\');
b           = char(a);
ind         = ind{1};
b(:,ind)    = ',';
lef_bracket = repmat('[',l(1),1);
rgt_bracket = repmat(']',l(1),1);
cma_val     = repmat(',',l(1),1);
b           = [lef_bracket,b,rgt_bracket,cma_val];
list_val    = ['[',reshape(b',1,size(b,1) * size(b,2)),']'];
end