-- Create Schemas
CREATE SCHEMA IF NOT EXISTS spotify;
CREATE SCHEMA IF NOT EXISTS energy;
CREATE SCHEMA IF NOT EXISTS fraud;

-- ==========================================
-- Schema: spotify
-- ==========================================
CREATE TABLE IF NOT EXISTS spotify.tracks_clean (
    track_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    artist VARCHAR,
