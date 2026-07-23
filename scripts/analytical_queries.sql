-- ====================================================================
-- Day 6: Analytical Queries for Spotify Data
-- These queries showcase SQL depth and will be used in the Streamlit dashboard
-- ====================================================================

-- 1. Top 10 Most Popular Tracks Overall
SELECT 
    name, 
    artist, 
    popularity, 
    decade 
FROM spotify.tracks_clean
ORDER BY popularity DESC
LIMIT 10;

-- 2. Average Track Characteristics (Vibe) by Decade
-- Demonstrates aggregations and type casting
SELECT 
    decade,
    ROUND(AVG(danceability)::numeric, 3) as avg_danceability,
    ROUND(AVG(energy)::numeric, 3) as avg_energy,
    ROUND(AVG(popularity)::numeric, 2) as avg_popularity,
    COUNT(*) as total_tracks
FROM spotify.tracks_clean
GROUP BY decade
ORDER BY decade ASC;

-- 3. Most Energetic Artists (Minimum of 5 tracks)
-- Demonstrates GROUP BY with HAVING clause filtering
SELECT 
    artist,
    COUNT(*) as track_count,
    ROUND(AVG(energy)::numeric, 3) as avg_energy
FROM spotify.tracks_clean
GROUP BY artist
HAVING COUNT(*) >= 5
ORDER BY avg_energy DESC
LIMIT 15;

-- 4. Popularity Distribution by Danceability Buckets (0 to 1 scale)
-- Demonstrates the use of WIDTH_BUCKET for histogram-like analysis
SELECT 
    WIDTH_BUCKET(danceability, 0, 1, 10) as danceability_bucket,
    ROUND(AVG(popularity)::numeric, 2) as avg_popularity,
    COUNT(*) as track_count
FROM spotify.tracks_clean
GROUP BY WIDTH_BUCKET(danceability, 0, 1, 10)
ORDER BY danceability_bucket;
