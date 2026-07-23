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
