-- Lost Objects Detection Service - Database Initialization
-- PostgreSQL database schema and initial data

-- =====================================================
-- EXTENSIONS
-- =====================================================

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable PostGIS for spatial data (optional)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- Enable pg_stat_statements for query performance monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- =====================================================
-- ENUMS
-- =====================================================

-- Object status enumeration
CREATE TYPE object_status AS ENUM (
    'normal',
    'suspect', 
    'lost',
    'resolved'
);

-- Alert level enumeration
CREATE TYPE alert_level AS ENUM (
    'LOW',
    'MEDIUM',
    'HIGH',
    'CRITICAL'
);

-- Job status enumeration
CREATE TYPE job_status AS ENUM (
    'pending',
    'queued',
    'processing',
    'completed',
    'failed',
    'cancelled'
);

-- Processing type enumeration
CREATE TYPE processing_type AS ENUM (
    'image',
    'video',
    'batch',
    'stream'
);

-- =====================================================
-- MAIN TABLES
-- =====================================================

-- Models table
CREATE TABLE models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(256),
    file_size BIGINT,
    description TEXT,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Locations table
CREATE TABLE locations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    location_type VARCHAR(100), -- airport, train_station, office, etc.
    coordinates POINT, -- spatial coordinates if available
    configuration JSONB, -- location-specific settings
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Processing jobs table
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id VARCHAR(255) NOT NULL UNIQUE,
    job_type processing_type NOT NULL,
    status job_status DEFAULT 'pending',
    progress DECIMAL(5,4) DEFAULT 0.0, -- 0.0 to 1.0
    model_id UUID REFERENCES models(id),
    location_id UUID REFERENCES locations(id),
    
    -- Input information
    input_file_name VARCHAR(500),
    input_file_size BIGINT,
    input_file_hash VARCHAR(256),
    
    -- Processing parameters
    processing_params JSONB,
    
    -- Results
    results JSONB,
    results_file_path TEXT,
    
    -- Metrics
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    processing_time_seconds DECIMAL(10,3),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Error handling
    error_message TEXT,
    error_details JSONB,
    
    -- Cleanup
    cleanup_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_progress CHECK (progress >= 0.0 AND progress <= 1.0)
);

-- Detected objects table (for tracking temporal objects)
CREATE TABLE detected_objects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    object_id VARCHAR(255) NOT NULL, -- unique tracking ID
    
    -- Classification
    class_name VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    
    -- Location
    location_id UUID REFERENCES locations(id),
    bounding_box JSONB NOT NULL, -- {x1, y1, x2, y2}
    
    -- Temporal tracking
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_movement TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    status object_status DEFAULT 'normal',
    alert_level alert_level DEFAULT 'LOW',
    
    -- Measurements
    stationary_duration_seconds INTEGER DEFAULT 0,
    nearest_person_distance DECIMAL(8,2), -- in meters
    
    -- Metadata
    metadata JSONB,
    
    -- Processing info
    model_id UUID REFERENCES models(id),
    job_id UUID REFERENCES processing_jobs(id),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_confidence CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT valid_distance CHECK (nearest_person_distance >= 0.0)
);

-- Object tracking history table
CREATE TABLE object_tracking_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    object_id VARCHAR(255) NOT NULL,
    detected_object_id UUID REFERENCES detected_objects(id),
    
    -- Previous state
    previous_status object_status,
    new_status object_status NOT NULL,
    
    -- Location change
    previous_location JSONB,
    new_location JSONB,
    
    -- Measurements
    confidence DECIMAL(5,4),
    distance_moved DECIMAL(8,2),
    time_since_last_update INTEGER, -- seconds
    
    -- Context
    trigger_reason VARCHAR(255), -- movement, timeout, manual, etc.
    additional_data JSONB,
    
    -- Timestamps
    tracked_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_confidence_history CHECK (confidence >= 0.0 AND confidence <= 1.0)
);

-- Detection results table (for individual detections)
CREATE TABLE detection_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Reference to processing job
    job_id UUID REFERENCES processing_jobs(id),
    
    -- Frame/image information
    frame_number INTEGER,
    timestamp_in_media DECIMAL(10,3), -- seconds from start
    
    -- Detection data
    detections JSONB NOT NULL, -- array of detection objects
    lost_objects JSONB, -- array of lost object alerts
    suspect_objects JSONB, -- array of suspect objects
    
    -- Performance
    processing_time_ms INTEGER,
    model_used VARCHAR(255),
    
    -- Timestamps
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API usage statistics
CREATE TABLE api_usage_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Request information
    endpoint VARCHAR(500) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    
    -- Client information
    client_ip INET,
    user_agent TEXT,
    api_key_hash VARCHAR(256), -- hashed API key for tracking
    
    -- Performance metrics
    response_time_ms INTEGER,
    request_size_bytes BIGINT,
    response_size_bytes BIGINT,
    
    -- Processing information
    model_used VARCHAR(255),
    processing_time_ms INTEGER,
    objects_detected INTEGER DEFAULT 0,
    
    -- Timestamps
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Additional data
    metadata JSONB
);

-- System performance metrics
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Metric information
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(50),
    
    -- Context
    component VARCHAR(255), -- api, model, database, etc.
    model_name VARCHAR(255),
    location_name VARCHAR(255),
    
    -- Aggregation
    aggregation_period VARCHAR(50), -- minute, hour, day
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    
    -- Additional data
    tags JSONB,
    metadata JSONB,
    
    -- Timestamp
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alerts and notifications
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Alert information
    alert_type VARCHAR(100) NOT NULL, -- lost_object, system_error, performance
    severity alert_level NOT NULL,
    title VARCHAR(500) NOT NULL,
    message TEXT NOT NULL,
    
    -- Context
    object_id VARCHAR(255), -- for object-related alerts
    job_id UUID REFERENCES processing_jobs(id),
    location_id UUID REFERENCES locations(id),
    
    -- Status
    status VARCHAR(50) DEFAULT 'active', -- active, acknowledged, resolved
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by VARCHAR(255),
    
    -- Notification
    notified BOOLEAN DEFAULT false,
    notification_attempts INTEGER DEFAULT 0,
    last_notification_attempt TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    alert_data JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- INDEXES
-- =====================================================

-- Models indexes
CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_type ON models(model_type);
CREATE INDEX idx_models_active ON models(is_active);

-- Locations indexes
CREATE INDEX idx_locations_name ON locations(name);
CREATE INDEX idx_locations_type ON locations(location_type);
CREATE INDEX idx_locations_active ON locations(is_active);

-- Processing jobs indexes
CREATE INDEX idx_jobs_job_id ON processing_jobs(job_id);
CREATE INDEX idx_jobs_type_status ON processing_jobs(job_type, status);
CREATE INDEX idx_jobs_created_at ON processing_jobs(created_at);
CREATE INDEX idx_jobs_model_id ON processing_jobs(model_id);
CREATE INDEX idx_jobs_location_id ON processing_jobs(location_id);

-- Detected objects indexes
CREATE INDEX idx_objects_object_id ON detected_objects(object_id);
CREATE INDEX idx_objects_status ON detected_objects(status);
CREATE INDEX idx_objects_class_name ON detected_objects(class_name);
CREATE INDEX idx_objects_location_id ON detected_objects(location_id);
CREATE INDEX idx_objects_first_seen ON detected_objects(first_seen);
CREATE INDEX idx_objects_last_seen ON detected_objects(last_seen);
CREATE INDEX idx_objects_alert_level ON detected_objects(alert_level);

-- Object tracking history indexes
CREATE INDEX idx_tracking_object_id ON object_tracking_history(object_id);
CREATE INDEX idx_tracking_detected_object_id ON object_tracking_history(detected_object_id);
CREATE INDEX idx_tracking_tracked_at ON object_tracking_history(tracked_at);
CREATE INDEX idx_tracking_status_change ON object_tracking_history(previous_status, new_status);

-- Detection results indexes
CREATE INDEX idx_results_job_id ON detection_results(job_id);
CREATE INDEX idx_results_processed_at ON detection_results(processed_at);

-- API usage stats indexes
CREATE INDEX idx_api_stats_endpoint ON api_usage_stats(endpoint);
CREATE INDEX idx_api_stats_requested_at ON api_usage_stats(requested_at);
CREATE INDEX idx_api_stats_client_ip ON api_usage_stats(client_ip);
CREATE INDEX idx_api_stats_status_code ON api_usage_stats(status_code);

-- Performance metrics indexes
CREATE INDEX idx_perf_metrics_name ON performance_metrics(metric_name);
CREATE INDEX idx_perf_metrics_component ON performance_metrics(component);
CREATE INDEX idx_perf_metrics_recorded_at ON performance_metrics(recorded_at);
CREATE INDEX idx_perf_metrics_period ON performance_metrics(period_start, period_end);

-- Alerts indexes
CREATE INDEX idx_alerts_type ON alerts(alert_type);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_status ON alerts(status);
CREATE INDEX idx_alerts_created_at ON alerts(created_at);
CREATE INDEX idx_alerts_object_id ON alerts(object_id);

-- =====================================================
-- FUNCTIONS AND TRIGGERS
-- =====================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updating updated_at
CREATE TRIGGER update_models_updated_at 
    BEFORE UPDATE ON models 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_locations_updated_at 
    BEFORE UPDATE ON locations 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detected_objects_updated_at 
    BEFORE UPDATE ON detected_objects 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alerts_updated_at 
    BEFORE UPDATE ON alerts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically create tracking history
CREATE OR REPLACE FUNCTION create_object_tracking_history()
RETURNS TRIGGER AS $$
BEGIN
    -- Only create history entry if status changed
    IF OLD.status IS DISTINCT FROM NEW.status THEN
        INSERT INTO object_tracking_history (
            object_id,
            detected_object_id,
            previous_status,
            new_status,
            previous_location,
            new_location,
            confidence,
            trigger_reason,
            tracked_at
        ) VALUES (
            NEW.object_id,
            NEW.id,
            OLD.status,
            NEW.status,
            OLD.bounding_box,
            NEW.bounding_box,
            NEW.confidence,
            'status_change',
            CURRENT_TIMESTAMP
        );
    END IF;
    
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for tracking history
CREATE TRIGGER create_tracking_history_trigger
    AFTER UPDATE ON detected_objects
    FOR EACH ROW EXECUTE FUNCTION create_object_tracking_history();

-- Function to cleanup old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete old completed jobs (older than 30 days)
    DELETE FROM processing_jobs 
    WHERE status IN ('completed', 'failed', 'cancelled')
    AND completed_at < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    -- Delete old API usage stats (older than 90 days)
    DELETE FROM api_usage_stats 
    WHERE requested_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Delete old performance metrics (older than 30 days for detailed, 1 year for daily aggregates)
    DELETE FROM performance_metrics 
    WHERE recorded_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
    AND aggregation_period IN ('minute', 'hour');
    
    DELETE FROM performance_metrics 
    WHERE recorded_at < CURRENT_TIMESTAMP - INTERVAL '1 year'
    AND aggregation_period = 'day';
    
    -- Delete old resolved alerts (older than 90 days)
    DELETE FROM alerts 
    WHERE status = 'resolved'
    AND resolved_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    -- Delete old tracking history (older than 60 days)
    DELETE FROM object_tracking_history 
    WHERE tracked_at < CURRENT_TIMESTAMP - INTERVAL '60 days';
    
END;
$$ language 'plpgsql';

-- =====================================================
-- VIEWS
-- =====================================================

-- View for active lost objects
CREATE VIEW active_lost_objects AS
SELECT 
    do.*,
    l.name as location_name,
    m.name as model_name,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - do.first_seen)) as total_duration_seconds
FROM detected_objects do
LEFT JOIN locations l ON do.location_id = l.id
LEFT JOIN models m ON do.model_id = m.id
WHERE do.status IN ('suspect', 'lost')
AND do.resolved_at IS NULL;

-- View for performance summary
CREATE VIEW performance_summary AS
SELECT 
    DATE_TRUNC('day', requested_at) as date,
    COUNT(*) as total_requests,
    COUNT(*) FILTER (WHERE status_code = 200) as successful_requests,
    AVG(response_time_ms) as avg_response_time_ms,
    SUM(objects_detected) as total_objects_detected,
    COUNT(DISTINCT client_ip) as unique_clients
FROM api_usage_stats
WHERE requested_at >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', requested_at)
ORDER BY date DESC;

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Insert default location
INSERT INTO locations (name, description, location_type, configuration) VALUES
('default', 'Default location for unspecified detections', 'general', '{"suspect_threshold": 30, "lost_threshold": 300}'),
('airport_terminal_1', 'Airport Terminal 1 - Departure Area', 'airport', '{"suspect_threshold": 60, "lost_threshold": 600}'),
('train_station_main', 'Main Train Station - Central Hall', 'train_station', '{"suspect_threshold": 45, "lost_threshold": 450}'),
('office_building_lobby', 'Office Building Main Lobby', 'office', '{"suspect_threshold": 120, "lost_threshold": 1800}');

-- Insert default model entries (these will be updated when models are loaded)
INSERT INTO models (name, version, model_type, file_path, description, is_active) VALUES
('stable_model_epoch_30', '1.0', 'production', '/app/storage/models/stable_model_epoch_30.pth', 'Stable production model with 28 object classes', true),
('fast_stream_model', '1.0', 'streaming', '/app/storage/models/fast_stream_model.pth', 'Optimized model for real-time streaming', true),
('mobile_model', '1.0', 'mobile', '/app/storage/models/mobile_model.pth', 'Lightweight model for edge deployment', false);

-- =====================================================
-- GRANTS AND PERMISSIONS
-- =====================================================

-- Create application user (if not exists)
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'lost_objects_app') THEN
      CREATE ROLE lost_objects_app LOGIN PASSWORD 'change_this_password';
   END IF;
END
$$;

-- Grant permissions to application user
GRANT CONNECT ON DATABASE lost_objects TO lost_objects_app;
GRANT USAGE ON SCHEMA public TO lost_objects_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO lost_objects_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO lost_objects_app;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO lost_objects_app;

-- Default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO lost_objects_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO lost_objects_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT EXECUTE ON FUNCTIONS TO lost_objects_app;

-- =====================================================
-- MAINTENANCE
-- =====================================================

-- Enable auto-vacuum
ALTER TABLE processing_jobs SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE detected_objects SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE api_usage_stats SET (autovacuum_vacuum_scale_factor = 0.05);
ALTER TABLE performance_metrics SET (autovacuum_vacuum_scale_factor = 0.05);

-- Create indexes concurrently (for production)
-- These can be run after initial setup to avoid blocking
-- CREATE INDEX CONCURRENTLY idx_objects_composite ON detected_objects(status, alert_level, location_id);
-- CREATE INDEX CONCURRENTLY idx_api_stats_composite ON api_usage_stats(requested_at, endpoint, status_code);

COMMIT;