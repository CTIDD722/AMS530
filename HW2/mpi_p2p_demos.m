%% mpi_p2p_demos.m
%  Author: Yunpeng Chu
%  Date:   2025/10/09
%  Note:   make sure the version of MATLAB is above R2016a

clear; clc;
%% Start parallel pool
Max_worker = 8; % Please adjust according to your device
cl = parcluster('local');
want = min(Max_worker, cl.NumWorkers);
p = gcp('nocreate');
if isempty(p)
    p = parpool(cl, want); 
elseif p.NumWorkers ~= want
    delete(p); p = parpool(cl, want); 
end
fprintf('\n[INFO] Pool ready: %d workers (local max %d)\n\n', p.NumWorkers, cl.NumWorkers);
root = 1;     
kary = 3;     
chunkS = 8;   
%% MPI_BCAST
fprintf('=== i) MPI_Bcast: Basic (binomial) & Extended (chunked) ===\n');

% Basic
msg_basic = int32([10 20 30 40]);
spmd
    if labindex == root, x = msg_basic; 
    else, x = int32([]); 
    end
    x = p2p_bcast_knomial(x, root, 2, 201);             
    fprintf('[Bcast-basic] lab %d got [%s]\n', labindex, strjoin(string(x),','));
end
fprintf('\n');

% Extended
msg_long = int32(1:36);
spmd
    if labindex == root, x = msg_long; else, x = int32([]); end
    x = p2p_bcast_knomial_chunked(x, root, kary, chunkS, 211);
    fprintf('[Bcast-EXT knomial+chunk] lab %d len=%d (head=[%s])\n', ...
        labindex, numel(x), strjoin(string(x(1:min(6,end))),','));
end
fprintf('\n');

%% MPI_SCATTER
fprintf('=== ii) MPI_Scatter: Basic (binomial tree) & Extended (k-nomial tree + chunking) ===\n');
P = p.NumWorkers;

% Basic
chunks = arrayfun(@(i) int32([i, i*10]), 1:P, 'UniformOutput', false);
spmd
    data_local = p2p_scatter_knomial(chunks, root, 2, 301);  
    fprintf('[Scatter-basic] lab %d got [%s]\n', labindex, strjoin(string(data_local),','));
end
fprintf('\n');

% Extended
bigChunks = arrayfun(@(i) int32(i*100 + (1:20)), 1:P, 'UniformOutput', false);
spmd
    data_local = p2p_scatter_knomial_chunked(bigChunks, root, kary, chunkS, 311);
    head = data_local(1:min(6,end));
    fprintf('[Scatter-EXT knomial+chunk] lab %d len=%d (head=[%s])\n', ...
        labindex, numel(data_local), strjoin(string(head),','));
end
fprintf('\n');

%% ========== iii) MPI_ALLGATHER ==========
fprintf('=== 3) MPI_Allgather: Basic (recursive doubling) & Extended (ring pipeline) ===\n');

% Basic
spmd
    myblock = int32([labindex, labindex+100]);
    all_cell = p2p_allgather_doubling(myblock, 401);          
    if labindex==1
        fprintf('[Allgather-basic doubling] total blocks=%d, block#1=[%s]\n', ...
            numel(all_cell), strjoin(string(all_cell{1}),','));
    end
end

% Extended
spmd
    myblock = int32([labindex, labindex+100]);
    all_cell = p2p_allgather_ring(myblock, 411);              
    if labindex==1
        fprintf('[Allgather-EXT ring] total blocks=%d, block#1=[%s]\n', ...
            numel(all_cell), strjoin(string(all_cell{1}),','));
    end
end
fprintf('\n');

%% ========== iv) MPI_ALLTOALL ==========
fprintf('=== 4) MPI_Alltoall: Basic (round-robin pairing) & Extended (partner aggregation) ===\n');

% Basic
spmd
    P = numlabs;
    out = cell(1,P);
    for j=1:P, out{j} = int32([labindex*100 + j]); end
    in = p2p_alltoall_roundrobin(out, 501);                 
    fprintf('[Alltoall-basic] lab %d got-from-3=[%s]\n', labindex, strjoin(string(in{3}),','));
end

% Extended
spmd
    P = numlabs;
    out = cell(1,P);
    for j=1:P, out{j} = int32([labindex, j, labindex*10+j]); end
    in = p2p_alltoall_roundrobin(out, 511);                   
    if labindex==1
        fprintf('[Alltoall-EXT aggregated] lab1 <- from lab2 pack=[%s]\n', strjoin(string(in{2}),','));
    end
end
fprintf('\n');

%% ========== v) MPI_REDUCE ==========
fprintf('=== 5) MPI_Reduce: Basic (tree reduction) & Extended (k-nomial + chunking) ===\n');

% Basic
spmd
    v = int32([labindex, 1]);           
    res = p2p_reduce_knomial_sum(v, root, 2, 601);           
    if labindex==root
        fprintf('[Reduce-basic sum] root got [%s]\n', strjoin(string(res),','));
    end
end

% Extended
longv = int32(1:37);
spmd
    v = int32(labindex*1000) + longv;   
    res = p2p_reduce_knomial_sum_chunked(v, root, kary, chunkS, 611);
    if labindex==root
        head = res(1:min(8,end));
        fprintf('[Reduce-EXT knomial+chunk] root len=%d (head=[%s])\n', numel(res), strjoin(string(head),','));
    end
end
fprintf('\n[INFO] Demo finished.\n');

%% Helper functions 

% rank mapping
function rr = rel_rank(rank_abs, root_abs, P)
    rr = mod((rank_abs-1) - (root_abs-1) + P, P);
end
function ra = abs_rank(rank_rel, root_abs, P)
    ra = mod(rank_rel + (root_abs-1), P) + 1;
end

% k-nomial levels
function [H, strides] = knomial_levels(P, k)
    strides = 1;  H = 0;
    while strides(end) < P
        strides(end+1) = strides(end)*k; 
        H = H + 1;
    end
    strides = strides(1:end-1); 
end

%% MPI_Bcast
function x = p2p_bcast_knomial(x, root, k, tagbase)
    P = numlabs;
    [H, strides] = knomial_levels(P, k);
    rr = rel_rank(labindex, root, P);

    % propagate from higher to lower levels
    for h = H:-1:1
        step = strides(h);   
        d = floor(rr/step) - k*floor(rr/(k*step));  
        tag = tagbase + h;
        if d == 0
            for j = 1:k-1
                child_rel = rr + j*step;
                if child_rel < P
                    partner = abs_rank(child_rel, root, P);
                    labSend(x, partner, tag);
                end
            end
        else
            parent_rel = rr - d*step;
            partner = abs_rank(parent_rel, root, P);
            x = labReceive(partner, tag);
        end
    end
end

function x = p2p_bcast_knomial_chunked(x, root, k, S, tagbase)
% first broadcast the length, then broadcast by chunks using knomial
    if labindex == root, N = numel(x); else, N = 0; end
    N = p2p_bcast_knomial(N, root, k, tagbase+0);
    if labindex ~= root, x = zeros(1,N,'like',x); end
    ptr = 1; seg = 0;
    while ptr <= N
        seg = seg + 1;
        hi = min(N, ptr+S-1);
        if labindex == root, blk = x(ptr:hi); else, blk = zeros(1,hi-ptr+1,'like',x); end
        blk = p2p_bcast_knomial(blk, root, k, tagbase+seg);
        x(ptr:hi) = blk; ptr = hi+1;
    end
end

%% MPI_Scatter
function data_local = p2p_scatter_knomial(chunks, root, k, tagbase)
    P = numlabs;
    if labindex ~= root, chunks = cell(1,P); end
    [H, strides] = knomial_levels(P, k);
    rr = rel_rank(labindex, root, P);
    for h = H:-1:1
        step = strides(h);  tag = tagbase + h;
        d = floor(rr/step) - k*floor(rr/(k*step));
        if d == 0
            for j = 1:k-1
                child_rel = rr + j*step;
                if child_rel < P
                    partner = abs_rank(child_rel, root, P);
                    R = child_rel:(child_rel+step-1); R = R(R<P);
                    Rabs = arrayfun(@(t) abs_rank(t, root, P), R);
                    payload.idx = Rabs;
                    payload.val = chunks(Rabs);
                    labSend(payload, partner, tag);
                    for u = Rabs, chunks{u} = []; end
                end
            end
        else
            parent_rel = rr - d*step;
            partner = abs_rank(parent_rel, root, P);
            payload = labReceive(partner, tag);
            for t = 1:numel(payload.idx)
                j = payload.idx(t);
                chunks{j} = payload.val{t};
            end
        end
    end
    data_local = chunks{labindex};
end

function data_local = p2p_scatter_knomial_chunked(bigChunks, root, k, S, tagbase)
    P = numlabs;
    if labindex == root
        chunks = cell(1,P);
        for j=1:P
            v = bigChunks{j}; N = numel(v); parts = {};
            ptr = 1;
            while ptr <= N
                hi = min(N, ptr+S-1);
                parts{end+1} = v(ptr:hi); 
                ptr = hi + 1;
            end
            chunks{j} = parts;
        end
    else
        chunks = cell(1,P);
    end

    [H, strides] = knomial_levels(P, k);
    rr = rel_rank(labindex, root, P);

    for h = H:-1:1
        step = strides(h);  tag = tagbase + h;
        d = floor(rr/step) - k*floor(rr/(k*step));
        if d == 0
            for j = 1:k-1
                child_rel = rr + j*step;
                if child_rel < P
                    partner = abs_rank(child_rel, root, P);
                    R = child_rel:(child_rel+step-1); R = R(R<P);
                    Rabs = arrayfun(@(t) abs_rank(t, root, P), R);
                    payload.idx = Rabs;
                    payload.parts = chunks(Rabs);
                    labSend(payload, partner, tag);
                    for u = Rabs, chunks{u} = []; end
                end
            end
        else
            parent_rel = rr - d*step;
            partner = abs_rank(parent_rel, root, P);
            payload = labReceive(partner, tag);
            for t = 1:numel(payload.idx)
                j = payload.idx(t);
                chunks{j} = payload.parts{t};
            end
        end
    end
    parts = chunks{labindex};
    if isempty(parts), data_local = int32([]); else, data_local = cat(2, parts{:}); end
end

%% MPI_Allgather
function all_cell = p2p_allgather_doubling(myblock, tagbase)
    P = numlabs;
    all_cell = cell(1,P); 
    all_cell{labindex} = myblock;

    s = 1; round = 0;
    while s < P
        round = round + 1;
        tag = tagbase + round;                 
        partner = bitxor(labindex-1, s) + 1;
        % lower rank sends then receives
        if labindex < partner  
            labSend(all_cell, partner, tag);
            recv = labReceive(partner, tag);
        % higher rank receives then sends
        else 
            recv = labReceive(partner, tag);
            labSend(all_cell, partner, tag);
        end
        % merge
        for j = 1:P
            if isempty(all_cell{j}) && ~isempty(recv{j})
                all_cell{j} = recv{j};
            end
        end
        s = s * 2;
    end
end

function all_cell = p2p_allgather_ring(myblock, tagbase)
    P = numlabs;
    all_cell = cell(1,P); all_cell{labindex} = myblock;
    send_idx = labindex;
    for r=1:P-1
        to   = mod(labindex, P) + 1;
        from = mod(labindex-2, P) + 1;
        labSend(struct('id',send_idx,'blk',all_cell{send_idx}), to, tagbase+10*r);
        recv = labReceive(from, tagbase+10*r);
        all_cell{recv.id} = recv.blk;
        send_idx = recv.id;
    end
end

%% MPI_Alltoall
function in = p2p_alltoall_roundrobin(out, tagbase)
    P = numlabs;
    in = cell(1,P);
    in{labindex} = out{labindex}; 
    order = 1:P; 
    % circle method
    for r = 1:P-1
        pos = find(order == labindex, 1, 'first');
        partner = order(P - pos + 1);
        if labindex < partner
            labSend(out{partner}, partner, tagbase + r);
            recv = labReceive(partner,        tagbase + r);
        else
            recv = labReceive(partner,        tagbase + r);
            labSend(out{partner}, partner,    tagbase + r);
        end
        in{partner} = recv;
        order = [order(1), order(end), order(2:end-1)];
    end
end

%% MPI_Reduce
function res = p2p_reduce_knomial_sum(v, root, k, tagbase)
    P = numlabs;
    [H, strides] = knomial_levels(P, k);
    rr = rel_rank(labindex, root, P);
    acc = v;  

    for h = 1:H
        step = strides(h); tag = tagbase + h;
        d = floor(rr/step) - k*floor(rr/(k*step));
        if d ~= 0
            parent_rel = rr - d*step;
            partner = abs_rank(parent_rel, root, P);
            labSend(acc, partner, tag);
            acc = [];   
            break;
        else
            for j = 1:k-1
                child_rel = rr + j*step;
                if child_rel < P
                    partner = abs_rank(child_rel, root, P);
                    recv = labReceive(partner, tag);
                    acc = acc + recv;
                end
            end
        end
    end
    if labindex == root, res = acc; else, res = []; end
end

function res = p2p_reduce_knomial_sum_chunked(v, root, k, S, tagbase)
    if labindex == root, L = numel(v); else, L = 0; end
    L = p2p_bcast_knomial(L, root, k, tagbase+0);
    ptr = 1; seg = 0; parts = {};
    while ptr <= L
        seg = seg + 1;
        hi = min(L, ptr+S-1);
        if labindex == root, blk = zeros(1, hi-ptr+1, 'like', v); else, blk = v(ptr:hi); end
        blk_res = p2p_reduce_knomial_sum(blk, root, k, tagbase+seg);
        if labindex == root, parts{end+1} = blk_res; end 
        ptr = hi + 1;
    end
    if labindex == root, res = cat(2, parts{:}); else, res = []; end
end
