function say_hello(maxWorkers, runs)
    %% Start log
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');             % Time
    logFile = sprintf('say_hello_output_%s.txt', timestamp); % File name
    diary(logFile);                                           
    fprintf('[INFO] Logging started: %s\n', logFile);
    fprintf('[INFO] Program begin at %s\n\n', datestr(now));

    %% Initialization
    if nargin < 1, maxWorkers = 32; end
    if nargin < 2, runs = 2; end
    maxWorkers = max(1, min(32, floor(maxWorkers))); % Set upper limit 
    cl = parcluster('local');                        % Local cluster
    avail = cl.NumWorkers;                          
    P = min(maxWorkers, avail);                      % Actual number used

    % Configure parallel pool
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= P
        if ~isempty(pool), delete(pool); end
        pool = parpool(cl, P);
    end
    fprintf('Local cluster allows up to %d workers; using P = %d workers.\n', avail, P);

    %% Non-deterministic order
    for r = 1:runs
        fprintf('\n[Run %d] Non-deterministic order:\n', r);
        seeds = randi(2^31-1, P, 1, 'uint32'); % Random seed
        parfor i = 1:P
            rng(seeds(i), 'twister');        
            pause(0.05 + 0.50 * rand()); % Set disturbance    
            fprintf('Hello from Processor %d\n', i-1);   
        end
    end

    %% Deterministic order: SPMD + barrier
    for r = 1:runs
        fprintf('\n[Run %d] Deterministic order via SPMD + barrier:\n', r);
        spmd
            for k = 1:spmdSize
                spmdBarrier;           
                if spmdIndex == k
                    fprintf('Hello from Processor %d\n', spmdIndex-1);
                end
            end
            spmdBarrier;                       
        end
    end

    %% Deterministic order: gather and client-side print
    fprintf('\nDeterministic order via gather and client-side print:\n');
    spmd
        msg = sprintf('Hello from Processor %d', labindex-1);
    end
    for k = 1:P
        fprintf('%s\n', msg{k});
    end

    %% Check
    expected = compose('Hello from Processor %d', 0:P-1);
    ok = true;
    for k = 1:P
        if ~strcmp(msg{k}, expected{k}), ok = false; break; end
    end
    fprintf('\nValidation: %s\n', tern(ok, 'PASS', 'FAIL'));

    %% End log
    fprintf('\n[INFO] Program finished at %s\n', datestr(now));
    fprintf('[INFO] Logging stopped.\n');
    diary off;                                              

    % delete(gcp('nocreate')); % Automatically close parallel pool
end

% Ternary utility function
function out = tern(cond, a, b)
    if cond, out = a; else, out = b; end
end
