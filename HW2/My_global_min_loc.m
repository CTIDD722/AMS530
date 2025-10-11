%% My_global_min_loc.m
%  Author: Yunpeng Chu
%  Date:   2025/10/09
%  Note:   make sure the version of MATLAB is above R2016a


clear; clc;

%% Configuration
Max_worker = 64;            
Result     = 'results.txt'; 

% Test list
tests = { ...
    struct('name','Example_P3_N4', 'P',3,'N',4, ...
           'given',{ { [1 9 3 4], [5 6 7 2], [9 8 6 1] } }, 'showDetails',true), ...
    struct('name','Random_P4_N10',  'P',4,'N',10,'given',[],'showDetails',true), ...
    struct('name','Random_P4_N100', 'P',4,'N',100,'given',[],'showDetails',false), ...
    struct('name','Random_P8_N10',  'P',8,'N',10,'given',[],'showDetails',true), ...
    struct('name','Random_P8_N100', 'P',8,'N',100,'given',[],'showDetails',false) ...
};

%% Pool bootstrap
cl   = parcluster('local');
want = min(Max_worker, cl.NumWorkers);
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool(cl, want);               
else
    if pool.NumWorkers ~= want
        delete(pool);
        pool = parpool(cl, want);            
    end
end
fprintf('[INFO] Pool ready with %d workers (max local %d).\n', pool.NumWorkers, cl.NumWorkers);

%% results header
fid = fopen(Result, 'w');
assert(fid>0, 'Cannot open results.txt for writing.');
fprintf(fid, 'Case,P,N,Correct,Time_sec\n');

%% Test
for t = 1:numel(tests)
    T = tests{t};
    P = T.P;  N = T.N;

    if P > pool.NumWorkers
        warning('Requested P=%d exceeds pool size=%d. Using P=%d instead.', ...
                P, pool.NumWorkers, pool.NumWorkers);
        P = pool.NumWorkers;
    end

    % matrix building
    if ~isempty(T.given)
        assert(numel(T.given)==P, 'Provided rows do not match effective P.');
        A = zeros(P, N, 'int32');
        for p = 1:P
            A(p,:) = int32(T.given{p});
        end
    else
        rng(20251009 + P*1000 + N); % reproducible
        A = int32(randi([0, 9999], P, N));        
    end

    [~, idxMin1] = min(double(A), [], 1);         
    truth0 = int32(idxMin1 - 1);                 
    rowsCell = arrayfun(@(p) A(p,:), 1:P, 'UniformOutput', false);

    % SPMD
    ranks_root = []; elapsed_all = [];
    spmd
        activeP = P;

        if labindex <= activeP
            A_local = rowsCell{labindex};
        else
            A_local = int32([]);
        end
        p2p_barrier(900);                   
        t0 = tic;
        ranks0_on_root = my_global_min_loc(A_local, activeP); 
        p2p_barrier(920);                    
        t_elapsed = toc(t0);
        ranks_root  = ranks0_on_root;        
        elapsed_all = t_elapsed;             
    end

    % Collection and check
    result0 = ranks_root{1};
    times   = [elapsed_all{:}];             
    t_sec   = max(times);                    
    is_ok = isequal(result0, truth0);
    fprintf('[%s] P=%d, N=%d, Correct=%d, Time=%.6f s\n', ...
            T.name, P, N, is_ok, t_sec);
    fprintf(fid, '%s,%d,%d,%d,%.6f\n', T.name, P, N, is_ok, t_sec);
    if T.showDetails
        detail(A, result0);
    end
end
fclose(fid);
fprintf('[INFO] Results written to %s\n', Result);

%% Local functions

function ranks0 = my_global_min_loc(A_local, activeP)
    if labindex > activeP
        ranks0 = int32([]);
        return;
    end

    N = numel(A_local);
    minVal  = A_local;                                    
    minRank = int32(labindex-1) * ones(1, N, 'int32');    

    step = 1;
    while step < activeP
        isReceiver = (mod(labindex-1, 2*step) == 0);
        if isReceiver
            partner = labindex + step;
            if partner <= activeP
                vals_in  = labReceive(partner, 100);     
                ranks_in = labReceive(partner, 101);     
                lt = (vals_in <  minVal);
                eq = (vals_in == minVal);
                minVal(lt)  = vals_in(lt);
                minRank(lt) = ranks_in(lt);
                if any(eq)
                    a = minRank(eq); b = ranks_in(eq);
                    minRank(eq) = min(a, b);            
                end
            end
        else
            partner = labindex - step;                    
            labSend(minVal,  partner, 100);
            labSend(minRank, partner, 101);
            break;
        end
        step = step * 2;
    end
    if labindex == 1
        ranks0 = minRank;
    else
        ranks0 = int32([]);
    end
end

function detail(A, ranks0)
    P = size(A,1);
    N = size(A,2);
    fprintf('For P = %d, N = %d:\n', P, N);
    for p = 1:P
        rowItems = strjoin(string(A(p,:)), ', ');
        fprintf('  o Process %d: A%d = {%s}\n', p-1, p-1, rowItems);
    end

    vecStr = strjoin(string(ranks0), ',');
    fprintf('  o Output on root: {%s}, where:\n', vecStr);
    bullet = char(8226);  
    mvals = min(double(A), [], 1);  

    for i = 1:N
        col = A(:,i).';
        items = strjoin(string(col), ', ');
        fprintf('    %s Index %d: min{%s} = %d (Process %d).\n', ...
                bullet, i-1, items, mvals(i), ranks0(i));
    end
end

function p2p_barrier(tagbase)
    P    = numlabs;
    root = 1;
    if labindex ~= root
        labSend(int32(1), root, tagbase + 0);
        labReceive(root,      tagbase + 1);
    else
        for s = 2:P
            labReceive(s,     tagbase + 0);
        end
        for d = 2:P
            labSend(int32(1), d,    tagbase + 1);
        end
    end
end
