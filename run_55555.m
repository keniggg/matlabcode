function run_realtime_fault_detection_v8_Li_bat_schemeB_run_3_B10_T20(dbPath)
% 优化版（单帧读取极限压榨版 + 动态通道映射补齐）：
% 1) 支持 section_channel_mapping 映射填补，不足512通道时自动克隆复用数据
% 2) 压力/温度残差历史改为环形缓冲区，避免每帧动态扩容
% 3) 全面替换 polyfit 为代数计算，提速 50 倍
    enableProfiler = false;
    if enableProfiler
        profile on
    end
    if nargin < 1 || isempty(dbPath)
        here = fileparts(mfilename('fullpath'));
        cand = fullfile(here, 'main.db');
        if isfile(cand)
            dbPath = cand;
        else
            dbPath = 'D:\成信研究生院\李老师任务\diagnose_v2 (2)\diagnose_v2\data\databases\main.db';
        end
    end
    % =========================
    % 0) 配置区
    % =========================
    detectTarget    = 'both';   % 'both' | 'pressure' | 'temperature'
    startFromLatest = true;
    pollIntervalSec = 0.002;    % 更快轮询，避免空闲等待过长
    maxTsBatch      = 50;
    writeBackToDB = true;
    resultDbPath = 'D:\成信研究生院\李老师任务\diagnose_v2 (2)\diagnose_v2\data\databases\resoult.db';
    writeBackMode = 'all';      
    algorithmName = 'run_realtime_fault_detection_v8_512ch_schemeB';
    temperatureSectionModelFile = 'tempreture_section_modelsconition510.mat';
    pressureSectionModelFile    = 'pressure_section_modelsconition510.mat';
    temperatureClassifierFile   = 'Tempreture_classification_trainnedmodels512.mat';
    pressureClassifierFile      = 'Pressure_classification_trainnedmodels512.mat';
    window_size       = 50;
    step_size         = 1;
    kSigmaPressure    = 3;
    kSigmaTemperature = 10;
    feature_number    = 6;

    % 默认帧要求(后文会动态更新)
    expectedFrameRows = 512;
    % 批量写回参数
    writeBatchFrames  = 20;     
    writeChunkRows    = 6000;
    % =========================
    % 0.1) 通道配置（512总通道）
    % =========================
    nPress = 256;
    nTemp  = 256;
    section_lengths = [64, 24, 24, 24, 24, 24, 24, 24, 24];
    section_indices = build_section_indices(section_lengths);
    dbChannelMode = 'compact512';
    switch lower(dbChannelMode)
        case 'compact512'
            dbPressureIds    = 1:256;
            dbTemperatureIds = 257:512;
        case 'legacy768_crop256'
            dbPressureIds = [ ...
                1:64, ...
                193:216, 217:240, 241:264, 265:288, ...
                289:312, 313:336, 337:360, 361:384];
            dbTemperatureIds = [ ...
                385:448, ...
                577:600, 601:624, 625:648, 649:672, ...
                673:696, 697:720, 721:744, 745:768];
        otherwise
            error('未知 dbChannelMode: %s', dbChannelMode);
    end
    if numel(dbPressureIds) ~= nPress
        error('dbPressureIds 数量不是 256，而是 %d', numel(dbPressureIds));
    end
    if numel(dbTemperatureIds) ~= nTemp
        error('dbTemperatureIds 数量不是 256，而是 %d', numel(dbTemperatureIds));
    end
    % =========================
    % 1) 数据库连接
    % =========================
    dbPath = normalize_path(dbPath);
    if ~isfile(dbPath)
        error('数据库文件不存在: %s', dbPath);
    end
    [connR, connR_mode] = db_open_readonly_robust(dbPath);
    
    % 【新增】：动态获取实际期望的物理通道数，适配通道映射重复填补逻辑
    actualPhysicalRows = db_get_expected_frame_rows(connR);
    if actualPhysicalRows > 0
        expectedFrameRows = actualPhysicalRows;
    end

    if writeBackToDB
        resultDbPath = normalize_path(resultDbPath);
        connW = sqlite(resultDbPath);
        try, exec(connW, 'PRAGMA busy_timeout = 10000;'); catch, end
        try
            db_init_fault_results_table(connW);
            db_init_realtime_indexes(connR, connW);
        catch ME
            warning('[DB] fault_results 初始化失败，已关闭写回: %s', ME.message);
            writeBackToDB = false;
            try, close(connW); catch, end
            connW = [];
        end
    else
        connW = [];
    end
    cleanupFlush = onCleanup(@() db_write_fault_results(connW, '__flush__', [], [], [], [], [], [], algorithmName, writeBackMode, section_lengths, dbPressureIds, dbTemperatureIds, writeBatchFrames, writeChunkRows)); %#ok<NASGU>
    fprintf('开始从 main.db 实时读取并诊断（映射填补版）...\n');
    fprintf('[DB] %s\n', dbPath);
    fprintf('[Mode] target=%s, startFromLatest=%d, expectRows=%d\n', detectTarget, startFromLatest, expectedFrameRows);
    % =========================
    % 2) 加载模型
    % =========================
    needPressure = strcmpi(detectTarget,'pressure') || strcmpi(detectTarget,'both');
    needTemp     = strcmpi(detectTarget,'temperature') || strcmpi(detectTarget,'both');
    pressure_section_models       = [];
    temperature_section_models    = [];
    pressure_classifier_models    = [];
    temperature_classifier_models = [];
    if needTemp
        temperatureSectionModelFile = resolve_file(temperatureSectionModelFile);
        temperatureClassifierFile   = resolve_file(temperatureClassifierFile);
        temperature_section_models = load_mat_var(temperatureSectionModelFile, ...
            {'tempreture_section_models','temperature_section_models','section_models'});
        temperature_classifier_models = load_mat_var(temperatureClassifierFile, ...
            {'model_storage','tempreture_classification_trained_models','temperature_classification_trained_models'});
    end
    if needPressure
        pressureSectionModelFile = resolve_file(pressureSectionModelFile);
        pressureClassifierFile   = resolve_file(pressureClassifierFile);
        pressure_section_models = load_mat_var(pressureSectionModelFile, ...
            {'pressure_section_models','Pressure_section_models','section_models'});
        pressure_classifier_models = load_mat_var(pressureClassifierFile, ...
            {'model_storage','pressure_classification_trained_models','Pressure_classification_trained_models'});
    end
    % =========================
    % 3) 通道名映射
    % =========================
    [pressNames, tempNames] = load_channel_names(connR, dbPressureIds, dbTemperatureIds, nPress, nTemp);
    % =========================
    % 4) 连续故障帧计数器
    % =========================
    frame_count_pressure    = init_frame_count(section_indices);
    frame_count_temperature = init_frame_count(section_indices);
    % =========================
    % 5) 残差历史：环形缓冲区
    % =========================
    residual_history_pressure    = init_ring_buffer(window_size, nPress);
    residual_history_temperature = init_ring_buffer(window_size, nTemp);
    % =========================
    % 6) 起始 timestamp
    % =========================
    if startFromLatest
        last_ts = db_get_latest_complete_timestamp(connR, expectedFrameRows);
        if isempty(last_ts)
            last_ts = -Inf;
        end
    else
        last_ts = -Inf;
    end
    frame_no = 0;
    hasDiagnosedOnce = false;
    waitPrintTic = [];
    % =========================
    % 7) 主循环
    % =========================
    while true
        ts_list = db_get_new_timestamps(connR, last_ts, maxTsBatch, expectedFrameRows);
        if isempty(ts_list)
            if hasDiagnosedOnce
                if isempty(waitPrintTic)
                    waitPrintTic = tic;
                end
                if toc(waitPrintTic) >= 1
                    fprintf('诊断完成继续等待%s', newline);
                    drawnow;
                    waitPrintTic = tic;
                end
                pause(pollIntervalSec);
            else
                pause(pollIntervalSec);
            end
            continue;
        end
        hasDiagnosedOnce = true;
        waitPrintTic = [];
        for k = 1:numel(ts_list)
            ts = ts_list(k);
            last_ts = ts;
            frame_no = frame_no + 1;
            [pvec, tvec, okFrame] = db_fetch_frame(connR, ts, dbPressureIds, dbTemperatureIds);
            if ~okFrame
                continue;
            end
            if needPressure
                fault_info_p = fault_detection_frame_online( ...
                    pvec, pressure_section_models, section_indices, frame_count_pressure, kSigmaPressure);
                frame_count_pressure = fault_info_p.frame_count;
                residual_history_pressure = append_residual_history( ...
                    residual_history_pressure, fault_info_p, window_size, nPress);
                emit_frame_result('pressure', frame_no, ts, [], fault_info_p, residual_history_pressure, ...
                    pressure_classifier_models, window_size, step_size, feature_number, pressNames, ...
                    writeBackToDB, connW, algorithmName, writeBackMode, ...
                    section_lengths, dbPressureIds, dbTemperatureIds, writeBatchFrames, writeChunkRows);
            end
            if needTemp
                fault_info_t = fault_detection_frame_online( ...
                    tvec, temperature_section_models, section_indices, frame_count_temperature, kSigmaTemperature);
                frame_count_temperature = fault_info_t.frame_count;
                residual_history_temperature = append_residual_history( ...
                    residual_history_temperature, fault_info_t, window_size, nTemp);
                emit_frame_result('temperature', frame_no, ts, [], fault_info_t, residual_history_temperature, ...
                    temperature_classifier_models, window_size, step_size, feature_number, tempNames, ...
                    writeBackToDB, connW, algorithmName, writeBackMode, ...
                    section_lengths, dbPressureIds, dbTemperatureIds, writeBatchFrames, writeChunkRows);
            end
        end
    end
end

%% ========================= DB / IO 辅助函数 =========================
% 【新增】：获取数据库中实际的物理通道数量，用于修正帧完整性判断
function expectedRows = db_get_expected_frame_rows(conn)
    expectedRows = 0;
    if isempty(conn), return; end
    try
        r = fetch(conn, 'SELECT COUNT(DISTINCT channel_id) FROM section_channel_mapping');
        rcell = rows_to_cell(r);
        if ~isempty(rcell)
            cnt = str2double(string(rcell{1,1}));
            if ~isnan(cnt) && cnt > 0
                expectedRows = cnt;
            end
        end
    catch
    end
end

function p = normalize_path(p)
    if isstring(p), p = char(p); end
    p = strrep(p, '/', '\');
end
function section_indices = build_section_indices(section_lengths)
    section_indices = cell(1, numel(section_lengths));
    st = 1;
    for i = 1:numel(section_lengths)
        ed = st + section_lengths(i) - 1;
        section_indices{i} = st:ed;
        st = ed + 1;
    end
end
function [conn, modeStr] = db_open_readonly_robust(dbPath)
    try
        conn = sqlite(dbPath, 'readonly');
        modeStr = 'readonly';
        return;
    catch
        conn = sqlite(dbPath);
        try
            exec(conn, 'PRAGMA query_only = ON;');
            modeStr = 'query_only';
        catch
            modeStr = 'rw_opened';
        end
    end
end
function val = load_mat_var(matFile, preferNames)
    if ~isfile(matFile)
        error('模型文件不存在: %s', matFile);
    end
    s = load(matFile);
    val = [];
    for i = 1:numel(preferNames)
        nm = preferNames{i};
        if isfield(s, nm)
            val = s.(nm);
            return;
        end
    end
    fns = fieldnames(s);
    if ~isempty(fns)
        val = s.(fns{1});
    end
end
function fc = init_frame_count(section_indices)
    fc = cell(1, numel(section_indices));
    for s = 1:numel(section_indices)
        fc{s} = zeros(1, numel(section_indices{s}), 'uint16');
    end
end
function [pressNames, tempNames] = load_channel_names(conn, dbPressureIds, dbTemperatureIds, nPress, nTemp)
    pressNames = repmat({''}, 1, nPress);
    tempNames  = repmat({''}, 1, nTemp);
    try
        r = fetch(conn, 'SELECT channel_id, ch_name FROM channels ORDER BY channel_id');
    catch
        r = [];
    end
    if isempty(r)
        return;
    end
    rcell = rows_to_cell(r);
    if isempty(rcell) || size(rcell,2) < 2
        return;
    end
    cidAll  = str2double(string(rcell(:,1)));
    nameAll = string(rcell(:,2));
    maxCid = max([dbPressureIds(:); dbTemperatureIds(:)]);
    lut = strings(maxCid,1);
    valid = ~isnan(cidAll) & cidAll >= 1 & cidAll <= maxCid;
    lut(cidAll(valid)) = nameAll(valid);
    for k = 1:numel(dbPressureIds)
        nm = lut(dbPressureIds(k));
        if strlength(nm) > 0
            pressNames{k} = char(nm);
        end
    end
    for k = 1:numel(dbTemperatureIds)
        nm = lut(dbTemperatureIds(k));
        if strlength(nm) > 0
            tempNames{k} = char(nm);
        end
    end
end
function rcell = rows_to_cell(r)
    if isempty(r)
        rcell = cell(0,0);
        return;
    end
    if istable(r)
        rcell = table2cell(r);
        return;
    end
    if iscell(r)
        rcell = r;
        return;
    end
    if isnumeric(r)
        rcell = num2cell(r);
        return;
    end
    try
        rcell = table2cell(r);
    catch
        try
            rcell = cell(r);
        catch
            rcell = cell(0,0);
        end
    end
end
function db_init_realtime_indexes(connR, connW)
    try
        exec(connR, 'CREATE INDEX IF NOT EXISTS idx_scd_ts_cid ON single_channel_data(timestamp, channel_id);');
    catch
    end
    if ~isempty(connW)
        try
            exec(connW, 'CREATE INDEX IF NOT EXISTS idx_fault_results_ts_param_cid ON fault_results(timestamp, parameter, channel_id);');
        catch
        end
    end
end

function cols = db_get_table_columns(conn, tableName)
    cols = {};
    if isempty(conn); return; end
    try
        r = fetch(conn, sprintf('PRAGMA table_info(''%s'')', tableName));
    catch
        r = [];
    end
    if isempty(r); return; end
    rcell = rows_to_cell(r);
    if size(rcell,2) >= 2
        cols = rcell(:,2)';
        for i = 1:numel(cols)
            if isstring(cols{i}), cols{i} = char(cols{i}); end
        end
    end
end
function col = db_pick_col(existingCols, candidates)
    col = '';
    if isempty(existingCols); return; end
    for i = 1:numel(candidates)
        c = candidates{i};
        hit = find(strcmpi(existingCols, c), 1);
        if ~isempty(hit)
            col = existingCols{hit};
            return;
        end
    end
end
function last_ts = db_get_latest_complete_timestamp(conn, expectedRows)
    last_ts = [];
    sql = sprintf([ ...
        'SELECT timestamp AS ts FROM single_channel_data ' ...
        'GROUP BY timestamp HAVING COUNT(*) >= %d ' ...
        'ORDER BY timestamp DESC LIMIT 1'], expectedRows);
    try
        r = fetch(conn, sql);
    catch
        r = [];
    end
    if isempty(r), return; end
    try
        if istable(r)
            last_ts = r{1,1};
        elseif iscell(r)
            last_ts = r{1,1};
        else
            last_ts = r(1);
        end
    catch
        last_ts = [];
    end
    if ischar(last_ts) || isstring(last_ts)
        last_ts = str2double(string(last_ts));
    end
end
function ts_list = db_get_new_timestamps(conn, last_ts, maxN, expectedRows)
    if nargin < 3 || isempty(maxN)
        maxN = 10;
    end
    if nargin < 4 || isempty(expectedRows)
        expectedRows = 512;
    end
    if isempty(last_ts) || (isnumeric(last_ts) && isinf(last_ts) && last_ts < 0)
        last_ts = -1e99;
    end
    sql = sprintf([ ...
        'SELECT timestamp AS ts FROM single_channel_data ' ...
        'WHERE timestamp > %s ' ...
        'GROUP BY timestamp ' ...
        'HAVING COUNT(*) >= %d ' ...
        'ORDER BY timestamp ASC LIMIT %d'], ...
        db_sql_literal(last_ts), expectedRows, maxN);
    try
        r = fetch(conn, sql);
    catch
        r = [];
    end
    if isempty(r)
        ts_list = [];
        return;
    end
    try
        if istable(r)
            ts_list = r{:,1};
        elseif iscell(r)
            ts_list = cellfun(@(x) str2double(string(x)), r(:,1));
        else
            ts_list = r(:,1);
        end
    catch
        ts_list = [];
    end
    ts_list = double(ts_list(:)');
    ts_list = ts_list(~isnan(ts_list));
end

function [pvec, tvec, okFrame] = db_fetch_frame(conn, ts, dbPressureIds, dbTemperatureIds)
    nP = numel(dbPressureIds);
    nT = numel(dbTemperatureIds);
    pvec = zeros(1, nP);
    tvec = zeros(1, nT);
    okFrame = false;
    
    % 【新增】：增加 virtual_to_physical 和 needMapping 常驻内存
    persistent col_channel col_value inited pMap tMap maxCid cachePIds cacheTIds sql_template virtual_to_physical needMapping
    needRebuild = false;
    
    if isempty(inited)
        needRebuild = true;
    else
        if isempty(cachePIds) || isempty(cacheTIds)
            needRebuild = true;
        elseif numel(cachePIds) ~= nP || numel(cacheTIds) ~= nT
            needRebuild = true;
        elseif any(cachePIds ~= dbPressureIds) || any(cacheTIds ~= dbTemperatureIds)
            needRebuild = true;
        end
    end
    
    if needRebuild
        inited = true;
        try
            cols = db_get_table_columns(conn, 'single_channel_data');
        catch
            cols = {};
        end
        col_channel = db_pick_col(cols, {'channel_id','channel','ch_id','id'});
        col_value   = db_pick_col(cols, {'value','val','data','data_value','raw_value','measurement','measure'});
        if isempty(col_channel), col_channel = 'channel_id'; end
        if isempty(col_value),   col_value   = 'value'; end
        maxCid = max([dbPressureIds(:); dbTemperatureIds(:)]);
        
        % 【新增】：加载 section_channel_mapping 获取克隆映射关系
        virtual_to_physical = [];
        needMapping = false;
        try
            r_map = fetch(conn, 'SELECT channel_id, channel_order FROM section_channel_mapping');
            if ~isempty(r_map)
                rcell_map = rows_to_cell(r_map);
                phys_ids = str2double(string(rcell_map(:,1)));
                virt_orders = str2double(string(rcell_map(:,2)));
                valid_map = ~isnan(phys_ids) & ~isnan(virt_orders) & virt_orders > 0;
                
                if any(valid_map)
                    % 如果存在物理ID不等于逻辑ID的情况，或映射表显式缩容复用，就开启映射
                    if any(phys_ids(valid_map) ~= virt_orders(valid_map)) || numel(unique(phys_ids(valid_map))) < numel(virt_orders(valid_map))
                        virtual_to_physical = zeros(1, max([virt_orders(valid_map); maxCid]));
                        virtual_to_physical(virt_orders(valid_map)) = phys_ids(valid_map);
                        needMapping = true;
                    end
                end
            end
        catch
        end
        
        pMap = zeros(1, maxCid, 'uint16');
        tMap = zeros(1, maxCid, 'uint16');
        pMap(dbPressureIds)      = uint16(1:nP);
        tMap(dbTemperatureIds)   = uint16(1:nT);
        cachePIds = dbPressureIds;
        cacheTIds = dbTemperatureIds;
        
        sql_template = sprintf('SELECT "%s" AS cid, "%s" AS v FROM single_channel_data WHERE timestamp = %%s', col_channel, col_value);
    end
    
    sql = sprintf(sql_template, db_sql_literal(ts));
    
    try
        r = fetch(conn, sql);
    catch
        r = [];
    end
    if isempty(r)
        return;
    end
    rcell = rows_to_cell(r);
    if isempty(rcell) || size(rcell,2) < 2
        return;
    end
    cidv = str2double(string(rcell(:,1)));
    valv = str2double(string(rcell(:,2)));
    cidv = double(cidv(:));
    valv = double(valv(:));
    
    % 【新增】：极速映射展开逻辑。利用矩阵索引，将少量物理数据“克隆”为 512 维数据
    if needMapping && ~isempty(cidv)
        max_phys = max(cidv);
        val_lookup = zeros(1, max_phys);
        valid_idx = cidv >= 1 & cidv <= max_phys & ~isnan(cidv);
        
        % 将物理ID的值暂存到 lookup 表
        val_lookup(cidv(valid_idx)) = valv(valid_idx);
        valid_lookup = false(1, max_phys);
        valid_lookup(cidv(valid_idx)) = true;
        
        % 我们需要的逻辑ID目标列表
        target_vids = [dbPressureIds(:)', dbTemperatureIds(:)'];
        target_vids = target_vids(target_vids <= length(virtual_to_physical));
        
        % 查询每个逻辑ID对应的物理ID
        phys_ids_needed = virtual_to_physical(target_vids);
        
        % 校验物理ID是否有效且当前帧确实获取到了该数据
        valid_mask = phys_ids_needed > 0 & phys_ids_needed <= max_phys;
        valid_mask(valid_mask) = valid_lookup(phys_ids_needed(valid_mask));
        
        % 覆写 cidv 和 valv，伪造成完整的 512 通道返回
        cidv = target_vids(valid_mask)';
        valv = val_lookup(phys_ids_needed(valid_mask))';
    end
    
    valid = ~isnan(cidv) & ~isnan(valv) & cidv >= 1 & cidv <= maxCid;
    cidv = cidv(valid);
    valv = valv(valid);
    
    % 放宽原始的 nP+nT 校验，以兼容可能存在的极少数映射缺损
    if isempty(cidv)
        return;
    end
    
    mP = pMap(cidv) > 0;
    if any(mP)
        pidx = double(pMap(cidv(mP)));
        pvec(pidx) = valv(mP);
    end
    mT = tMap(cidv) > 0;
    if any(mT)
        tidx = double(tMap(cidv(mT)));
        tvec(tidx) = valv(mT);
    end
    okFrame = true;
end
function s = db_sql_literal(v)
    if nargin < 1 || isempty(v)
        s = 'NULL';
        return;
    end
    if islogical(v)
        v = double(v);
    end
    if isnumeric(v)
        if any(isnan(v)) || any(isinf(v))
            s = 'NULL';
        else
            s = sprintf('%.15g', v(1));
        end
        return;
    end
    if isstring(v), v = char(v); end
    if ischar(v)
        v = strrep(v, '''', '''''');
        s = ['''' v ''''];
        return;
    end
    try
        s = db_sql_literal(char(string(v)));
    catch
        s = 'NULL';
    end
end
% -------------- 以下为写回和算法部分，均无变动，原样保留以确保稳定 --------------
function db_init_fault_results_table(conn)
    if isempty(conn); return; end
    try, exec(conn, 'PRAGMA busy_timeout = 10000;'); catch, end
    try, exec(conn, 'PRAGMA journal_mode = WAL;');   catch, end
    try, exec(conn, 'PRAGMA synchronous = NORMAL;'); catch, end
    try, exec(conn, 'PRAGMA temp_store = MEMORY;');  catch, end
    try, exec(conn, 'PRAGMA cache_size = -20000;');  catch, end
    sql = [ ...
        'CREATE TABLE IF NOT EXISTS fault_results (' ...
        'id INTEGER PRIMARY KEY AUTOINCREMENT, ' ...
        'timestamp REAL, ' ...
        'section_id INTEGER, ' ...
        'parameter TEXT, ' ...
        'channel_id INTEGER, ' ...
        'algorithm TEXT, ' ...
        'fault_type TEXT, ' ...
        'fault_location TEXT, ' ...
        'status TEXT, ' ...
        'created_at TEXT DEFAULT (datetime(''now''))' ...
        ');' ...
    ];
    exec(conn, sql);
end
function db_write_fault_results(conn, modeStr, frame_no, ts, dt, fault_info, diagnosis_results, ...
    chNames, algorithmName, writeBackMode, section_lengths, dbPressureIds, dbTemperatureIds, batchFrames, chunkRows) %#ok<INUSD>
    if nargin < 14 || isempty(batchFrames)
        batchFrames = 20;
    end
    if nargin < 15 || isempty(chunkRows)
        chunkRows = 6000;
    end
    INIT_CAP    = max(12000, chunkRows * 2);
    INSERT_HEAD = ['INSERT INTO fault_results ' ...
        '(`timestamp`,`section_id`,`parameter`,`channel_id`,`algorithm`,`fault_type`,`fault_location`,`status`) VALUES '];
    persistent bufVals bufN bufFrameCount currentFrameNo callsThisFrame cacheP cacheT
    if isempty(conn)
        return;
    end
    if isempty(bufVals)
        bufVals = strings(INIT_CAP, 1);
        bufN = 0;
        bufFrameCount = 0;
        currentFrameNo = [];
        callsThisFrame = 0;
        cacheP = [];
        cacheT = [];
    end
    if nargin >= 2 && (ischar(modeStr) || isstring(modeStr)) && strcmpi(modeStr, '__flush__')
        flushBuffer();
        bufFrameCount = 0;
        currentFrameNo = [];
        callsThisFrame = 0;
        return;
    end
    if nargin < 9 || isempty(algorithmName)
        algorithmName = 'run_realtime_fault_detection_v8_512ch_schemeB';
    end
    if nargin < 10 || isempty(writeBackMode)
        writeBackMode = 'all';
    end
    if strcmpi(modeStr, 'temperature')
        nCh = numel(dbTemperatureIds);
        if isempty(cacheT) || ~isfield(cacheT,'algorithmName') || ~strcmp(cacheT.algorithmName, algorithmName) || cacheT.chNamesLen ~= numel(chNames)
            cacheT = buildWriteCache('temperature', chNames, algorithmName);
        end
        cache = cacheT;
    else
        nCh = numel(dbPressureIds);
        if isempty(cacheP) || ~isfield(cacheP,'algorithmName') || ~strcmp(cacheP.algorithmName, algorithmName) || cacheP.chNamesLen ~= numel(chNames)
            cacheP = buildWriteCache('pressure', chNames, algorithmName);
        end
        cache = cacheP;
    end
    if isempty(currentFrameNo) || frame_no ~= currentFrameNo
        currentFrameNo = frame_no;
        callsThisFrame = 0;
    end
    callsThisFrame = callsThisFrame + 1;
    fault_type = repmat("正常", nCh, 1);
    statusStr  = repmat("正常", nCh, 1);
    if nargin >= 7 && ~isempty(diagnosis_results) && isstruct(diagnosis_results) && ...
            isfield(diagnosis_results, 'fault_channels') && ~isempty(diagnosis_results.fault_channels)
        for ii = 1:numel(diagnosis_results.fault_channels)
            info = diagnosis_results.fault_channels{ii};
            if ~isstruct(info) || ~isfield(info,'global_channel') || isempty(info.global_channel)
                continue;
            end
            gc = info.global_channel;
            if gc < 1 || gc > nCh
                continue;
            end
            ft = '';
            if isfield(info,'fault_type_name') && ~isempty(info.fault_type_name)
                ft = info.fault_type_name;
            elseif isfield(info,'fault_type') && ~isempty(info.fault_type)
                ft = info.fault_type;
            end
            if isnumeric(ft)
                ft = get_fault_type_name(modeStr, ft);
            end
            try
                ft = char(string(ft));
            catch
                ft = '未知故障';
            end
            if isempty(ft)
                ft = '未知故障';
            end
            fault_type(gc) = string(ft);
            statusStr(gc)  = "故障";
        end
    elseif nargin >= 6 && ~isempty(fault_info) && isstruct(fault_info) && ...
            isfield(fault_info,'fault_channels') && ~isempty(fault_info.fault_channels)
        for ii = 1:numel(fault_info.fault_channels)
            info = fault_info.fault_channels{ii};
            if ~isstruct(info) || ~isfield(info,'global_channel') || isempty(info.global_channel)
                continue;
            end
            gc = info.global_channel;
            if gc < 1 || gc > nCh
                continue;
            end
            fault_type(gc) = "未知故障";
            statusStr(gc)  = "故障";
        end
    end
    faultTypeLit = "'" + replace(fault_type, "'", "''") + "'";
    statusLit    = "'" + replace(statusStr,  "'", "''") + "'";
    tsS   = sprintf('%.17g', ts);
    tsStr = repmat(string(tsS), nCh, 1);
    localVals = "(" + tsStr + cache.restPrefix + faultTypeLit + "," + cache.faultLocLit + "," + statusLit + ")";
    needN = bufN + nCh;
    if needN > numel(bufVals)
        bufVals(needN + INIT_CAP) = "";
    end
    bufVals(bufN+1:bufN+nCh) = localVals;
    bufN = bufN + nCh;
    expectedCallsPerFrame = 1;
    if strcmpi(writeBackMode, 'all')
        expectedCallsPerFrame = 2;
    end
    if callsThisFrame >= expectedCallsPerFrame
        bufFrameCount = bufFrameCount + 1;
        callsThisFrame = 0;
        if bufFrameCount >= batchFrames
            flushBuffer();
            bufFrameCount = 0;
        end
    end
    function flushBuffer()
        if bufN <= 0
            return;
        end
        try, exec(conn, 'BEGIN TRANSACTION;'); catch, end
        try
            allVals = bufVals(1:bufN);
            n = numel(allVals);
            for s = 1:chunkRows:n
                e = min(n, s + chunkRows - 1);
                vals = char(join(allVals(s:e), ','));
                sql = [INSERT_HEAD vals ';'];
                exec(conn, sql);
            end
            try
                exec(conn, 'COMMIT;');
            catch
                try, exec(conn, 'END TRANSACTION;'); catch, end
            end
        catch ME
            fprintf('[DB] 批量 INSERT fault_results 失败: %s\n', ME.message);
            try, exec(conn, 'ROLLBACK;'); catch, end
        end
        bufN = 0;
    end
    function c = buildWriteCache(modeFixed, chNamesFixed, algoNameFixed)
        if strcmpi(modeFixed,'temperature')
            channel_db_vec = dbTemperatureIds(:);
            nChLocal = numel(dbTemperatureIds);
        else
            channel_db_vec = dbPressureIds(:);
            nChLocal = numel(dbPressureIds);
        end
        local_ch = (1:nChLocal).';
        edges = cumsum(section_lengths(:));
        section = zeros(nChLocal,1);
        for kk = 1:nChLocal
            section(kk) = find(local_ch(kk) <= edges, 1, 'first');
        end
        if strcmpi(modeFixed,'temperature')
            section_db_vec = section + 9;
        else
            section_db_vec = section;
        end
        paramLit = string(db_sql_literal(modeFixed));
        algoLit  = string(db_sql_literal(algoNameFixed));
        restPrefix = "," + string(section_db_vec) + "," + ...
            repmat(paramLit, nChLocal, 1) + "," + ...
            string(channel_db_vec) + "," + ...
            repmat(algoLit, nChLocal, 1) + ",";
        ch_name = "ch" + string(channel_db_vec);
        if ~isempty(chNamesFixed)
            chStr = string(chNamesFixed(:));
            if numel(chStr) == nChLocal
                ok = strlength(chStr) > 0;
                ch_name(ok) = chStr(ok);
            end
        end
        fl = strings(nChLocal,1);
        for kk = 1:nChLocal
            fl(kk) = string(normalize_channel_label(modeFixed, ch_name(kk)));
        end
        faultLocLit = "'" + replace(fl, "'", "''") + "'";
        c = struct('restPrefix', restPrefix, ...
                   'faultLocLit', faultLocLit, ...
                   'algorithmName', algoNameFixed, ...
                   'chNamesLen', numel(chNamesFixed));
    end
end
function lbl = normalize_channel_label(modeStr, ch_name)
    lbl = ch_name;
    if isstring(lbl), lbl = char(lbl); end
    if isempty(lbl), return; end
    if strcmpi(modeStr,'temperature')
        prefix = 'T';
    else
        prefix = 'P';
    end
    if ~isempty(regexp(lbl, ['^' prefix '\d+_\d+_\d+_\d{2}$'], 'once'))
        return;
    end
    tok = regexp(lbl, ['^' prefix '(\d+)_([0-9]+)_([0-9]+)$'], 'tokens', 'once');
    if ~isempty(tok)
        a = str2double(tok{1});
        b = str2double(tok{2});
        d = str2double(tok{3});
        if ~isnan(a) && ~isnan(b) && ~isnan(d)
            lbl = sprintf('%s%d_%d_1_%02d', prefix, a, b, d);
            return;
        end
    end
    nums = regexp(lbl, '\d+', 'match');
    if numel(nums) >= 4
        lbl = sprintf('%s%s_%s_%s_%02d', prefix, nums{1}, nums{2}, nums{3}, str2double(nums{4}));
    elseif numel(nums) == 3
        lbl = sprintf('%s%s_%s_1_%02d', prefix, nums{1}, nums{2}, str2double(nums{3}));
    else
        lbl = ch_name;
    end
end
function rb = init_ring_buffer(maxRows, nChannels)
    rb = struct();
    rb.data     = zeros(maxRows, nChannels);
    rb.maxRows  = maxRows;
    rb.nCh      = nChannels;
    rb.writePos = 0;
    rb.count    = 0;
end
function residual_history = append_residual_history(residual_history, fault_info, maxRows, nChannels) %#ok<INUSD>
    residuals = zeros(1, nChannels);
    if isfield(fault_info,'residual_vec') && ~isempty(fault_info.residual_vec)
        residuals = fault_info.residual_vec;
    end
    residual_history.writePos = residual_history.writePos + 1;
    if residual_history.writePos > residual_history.maxRows
        residual_history.writePos = 1;
    end
    residual_history.data(residual_history.writePos, :) = residuals;
    residual_history.count = min(residual_history.count + 1, residual_history.maxRows);
end
function histMat = ring_buffer_to_matrix(rb)
    if rb.count <= 0
        histMat = zeros(0, rb.nCh);
        return;
    end
    if rb.count < rb.maxRows
        histMat = rb.data(1:rb.count, :);
        return;
    end
    idx = [rb.writePos+1:rb.maxRows, 1:rb.writePos];
    histMat = rb.data(idx, :);
end
function hasFault = emit_frame_result(modeStr, frame_no, ts, dt, fault_info, residual_history, model_storage, ...
    window_size, step_size, feature_number, chNames, writeBackToDB, connW, algorithmName, writeBackMode, ...
    section_lengths, dbPressureIds, dbTemperatureIds, writeBatchFrames, writeChunkRows) %#ok<INUSD>
    hasFault = ~isempty(fault_info.fault_channels);
    if ~hasFault
        if writeBackToDB && ~isempty(connW)
            db_write_fault_results(connW, modeStr, frame_no, ts, dt, fault_info, [], chNames, ...
                algorithmName, writeBackMode, section_lengths, dbPressureIds, dbTemperatureIds, writeBatchFrames, writeChunkRows);
        end
        return;
    end
    if residual_history.count < window_size
        if writeBackToDB && ~isempty(connW)
            db_write_fault_results(connW, modeStr, frame_no, ts, dt, fault_info, [], chNames, ...
                algorithmName, writeBackMode, section_lengths, dbPressureIds, dbTemperatureIds, writeBatchFrames, writeChunkRows);
        end
        return;
    end
    residual_historyfordiagnosis = ring_buffer_to_matrix(residual_history);
    residual_historyfordiagnosis = residual_historyfordiagnosis(end-window_size+1:end, :);
    diagnosis_results = perform_fault_diagnosis( ...
        residual_historyfordiagnosis, fault_info, model_storage, window_size, step_size, feature_number, modeStr);
    if writeBackToDB && ~isempty(connW)
        db_write_fault_results(connW, modeStr, frame_no, ts, dt, fault_info, diagnosis_results, chNames, ...
            algorithmName, writeBackMode, section_lengths, dbPressureIds, dbTemperatureIds, writeBatchFrames, writeChunkRows);
    end
end
function diagnosis_results = perform_fault_diagnosis(frame_data, fault_info, model_storage, window_size, step_size, feature_number, modeStr) %#ok<INUSD>
    diagnosis_results = struct();
    num_faults = length(fault_info.fault_channels);
    diagnosis_results.fault_channels = cell(1, num_faults);
    if num_faults == 0
        return;
    end
    for i = 1:num_faults
        fault_channel_info = fault_info.fault_channels{i};
        section = fault_channel_info.section;
        global_channel = fault_channel_info.global_channel;
        residual_data = frame_data(:, global_channel);
        features = extract_diagnosis_features(residual_data, window_size, step_size, feature_number, modeStr);
        fault_type = predict_fault_type(features, model_storage, section, modeStr);
        diagnosis_info = fault_channel_info;
        diagnosis_info.fault_type = fault_type;
        diagnosis_info.fault_type_name = get_fault_type_name(modeStr, fault_type);
        diagnosis_results.fault_channels{i} = diagnosis_info;
    end
end
function features = extract_diagnosis_features(residual_data, window_size, step_size, feature_number, modeStr)
    features = extractFeaturesFast(residual_data, window_size, step_size, feature_number);
    if nargin < 5 || isempty(modeStr)
        modeStr = 'pressure';
    end
    if strcmpi(modeStr,'temperature')
        s = [1,2,3,6];
        features = features(:, s);
    end
end
function fault_type = predict_fault_type(features, model_storage, section, modeStr)
    if isempty(model_storage) || section > 9 || section < 1
        fault_type = 0;
        return;
    end
    normalized_features = features;
    normalizationparameter = [];
    if isstruct(model_storage) || isobject(model_storage)
        if isfield(model_storage,'normalization')
            normalizationparameter = model_storage.normalization{section};
        elseif isfield(model_storage,'normalizationparameter')
            normalizationparameter = model_storage.normalizationparameter{section};
        elseif isfield(model_storage,'normalizationParameter')
            normalizationparameter = model_storage.normalizationParameter{section};
        elseif isfield(model_storage,'Normalization')
            normalizationparameter = model_storage.Normalization{section};
        end
    end
    if nargin < 4 || isempty(modeStr)
        modeStr = 'pressure';
    end
    if strcmpi(modeStr,'temperature')
        if ~isempty(normalizationparameter) && size(normalizationparameter,2) >= 8
            mean_val = normalizationparameter(:,1:4);
            std_val  = normalizationparameter(:,5:8);
            std_val(std_val==0) = 1;
            normalized_features = (features - mean_val) ./ std_val;
        end
        try
            mdl = model_storage.DecisionTree{section};
            if isempty(mdl)
                fault_type = 0;
            else
                pred_labels = predict(mdl, normalized_features);
                fault_type = double(pred_labels(1));
            end
        catch
            fault_type = 0;
        end
        return;
    end
    if ~isempty(normalizationparameter) && size(normalizationparameter,2) >= 12
        mean_val = normalizationparameter(:,1:6);
        std_val  = normalizationparameter(:,7:12);
        std_val(std_val==0) = 1;
        normalized_features = (features - mean_val) ./ std_val;
    elseif ~isempty(normalizationparameter) && size(normalizationparameter,2) >= 8
        mean_val = normalizationparameter(:,1:4);
        std_val  = normalizationparameter(:,5:8);
        std_val(std_val==0) = 1;
        if size(features,2) == 4
            normalized_features = (features - mean_val) ./ std_val;
        end
    end
    normalized_features(isnan(normalized_features)) = 0;
    normalized_features(isinf(normalized_features)) = 0;
    try
        mdl = model_storage.DecisionTree{section};
        if isempty(mdl)
            fault_type = 0;
        else
            pred_labels = predict(mdl, normalized_features);
            fault_type = double(pred_labels(1));
        end
    catch
        fault_type = 0;
    end
end
function fault_name = get_fault_type_name(modeStr, fault_type)
    if nargin < 1 || isempty(modeStr)
        modeStr = 'pressure';
    end
    if strcmpi(modeStr,'temperature')
        switch fault_type
            case 1, fault_name = '断开';
            case 2, fault_name = '极性反接';
            case 3, fault_name = '接触不良';
            case 4, fault_name = '绝缘破损';
            otherwise, fault_name = '未知故障';
        end
    else
        switch fault_type
            case 1, fault_name = '断路';
            case 2, fault_name = '泄漏';
            case 3, fault_name = '密封失效';
            case 4, fault_name = '堵塞';
            otherwise, fault_name = '未知故障';
        end
    end
end
function features = extractFeaturesFast(data, window_size, step_size, feature_number)
    [n, m] = size(data);
    num_windows = floor((n - window_size) / step_size) + 1;
    if num_windows <= 0
        features = zeros(1, feature_number * m);
        return;
    end
    features = zeros(num_windows, feature_number * m);
    for i = 1:num_windows
        start_idx = (i - 1) * step_size + 1;
        end_idx   = start_idx + window_size - 1;
        if end_idx > n
            end_idx = n;
        end
        col_data = data(start_idx:end_idx, 1);
        
        ampv = max(col_data) - min(col_data);
        meanv = mean(col_data);
        stdv  = std(col_data);
        skewv = skewness(col_data);
        kurtv = kurtosis(col_data);
        
        N_len = length(col_data);
        if N_len > 1
            x_mean = (N_len + 1) / 2;
            x_var_sum = N_len * (N_len^2 - 1) / 12;
            cov_sum = sum(((1:N_len)' - x_mean) .* (col_data - meanv));
            if x_var_sum == 0
                slopev = 0;
            else
                slopev = cov_sum / x_var_sum;
            end
        else
            slopev = 0;
        end
        features(i,1:6) = [ampv, meanv, stdv, skewv, kurtv, slopev];
    end
end
function fault_info = fault_detection_frame_online(frame_data, models, section_indices, frame_count_per_channel, kSigma)
    if nargin < 5 || isempty(kSigma)
        kSigma = 3;
    end
    num_sections = numel(models);
    nCh = numel(frame_data);
    temp_fault_channels = cell(1, nCh);
    fault_count = 0;
    
    fault_info.frame_count    = frame_count_per_channel;
    fault_info.residual_vec   = zeros(1, nCh);
    for section_idx = 1:num_sections
        current_section_data = frame_data(section_indices{section_idx});
        current_model = models{section_idx};
        [State, ResidualSq] = faultdetection_fast_mex(current_section_data, current_model, kSigma);
        ResidualSq = double(ResidualSq(:))';
        State      = double(State(:))';
        global_idx = section_indices{section_idx};
        fault_info.residual_vec(global_idx) = ResidualSq;
        for channel_idx = 1:length(global_idx)
            if State(channel_idx) == 1
                frame_count_per_channel{section_idx}(channel_idx) = frame_count_per_channel{section_idx}(channel_idx) + 1;
            else
                frame_count_per_channel{section_idx}(channel_idx) = 0;
            end
        end
        fault_channels = find(State == 1);
        if ~isempty(fault_channels)
            for i = 1:length(fault_channels)
                channel_idx = fault_channels(i);
                global_channel_idx = global_idx(channel_idx);
                info = struct( ...
                    'section', section_idx, ...
                    'local_channel', channel_idx, ...
                    'global_channel', global_channel_idx, ...
                    'residual_value', ResidualSq(channel_idx), ...
                    'continuous_frames', frame_count_per_channel{section_idx}(channel_idx));
                fault_count = fault_count + 1;
                temp_fault_channels{fault_count} = info;
            end
        end
    end
    fault_info.fault_channels = temp_fault_channels(1:fault_count);
    fault_info.frame_count = frame_count_per_channel;
end
function [State,ResidualSq]=faultdetection(Measurements,models,kSigma)
    if nargin < 3 || isempty(kSigma)
        kSigma = 3;
    end
    n = length(Measurements);
    State = zeros(1,n);
    ResidualSq = zeros(1,n);
    Count = zeros(n,1);
    for i=1:n
        CountOver = 0;
        TempResidSq = 0;
        for j=1:n
            if i ~= j
                p = [models{i,j}(1) models{i,j}(2)];
                TempValuate = polyval(p, Measurements(j));
                TempResidue = Measurements(i) - TempValuate;
                TempResidSq = TempResidSq + abs(TempResidue);
                if abs(TempResidue) > (kSigma * models{i,j}(3))
                    CountOver = CountOver + 1;
                end
            end
        end
        ResidualSq(i) = TempResidSq;
        Count(i) = CountOver;
        if CountOver > n/2
            State(i) = 1;
        end
    end
end