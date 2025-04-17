-- =================================================================
-- EXTENSION SETUP
-- =================================================================
-- Enable the pgvector extension for vector similarity search capabilities
create extension if not exists vector;

-- =================================================================
-- TABLE DEFINITION
-- =================================================================
-- Create the site_pages table to store processed website content chunks
create table site_pages (
  id bigserial primary key,                     -- Unique identifier for each chunk
  url varchar not null,                         -- URL of the page where this content was extracted
  chunk_number integer not null,                -- Sequential number for chunks from the same page
  title varchar not null,                       -- Title of the page
  summary varchar not null,                     -- Concise summary of the chunk content
  content text not null,                        -- Complete text content of the chunk
  metadata jsonb not null default '{}'::jsonb,  -- Flexible JSON metadata for additional properties
  embedding vector(1536),                       -- Vector embedding for semantic search (OpenAI dimensions)
  created_at timestamp with time zone default timezone('utc'::text, now()) not null, -- Creation timestamp
  
  -- Prevent duplicate chunks for the same URL
  unique(url, chunk_number)
);

-- =================================================================
-- INDEXES
-- =================================================================
-- Vector similarity search index using Inverted File with Flat compression
create index on site_pages using ivfflat (embedding vector_cosine_ops);

-- JSON metadata index for efficient filtering on metadata fields
create index idx_site_pages_metadata on site_pages using gin (metadata);

-- =================================================================
-- SEARCH FUNCTION
-- =================================================================
-- Function to perform semantic search on site_pages using vector similarity
create function match_site_pages (
  query_embedding vector(1536),     -- Input: The embedding vector to search against
  match_count int default 10,       -- Input: Number of results to return (default 10)
  filter jsonb DEFAULT '{}'::jsonb  -- Input: Optional JSON filter to apply to metadata
) returns table (                   -- Returns a table with the following columns:
  id bigint,                        -- ID of the matching record
  url varchar,                      -- URL where the content is located
  chunk_number integer,             -- Sequence number of the chunk within the page
  title varchar,                    -- Title of the page
  summary varchar,                  -- Summary of the chunk content
  content text,                     -- Full text content of the chunk
  metadata jsonb,                   -- Additional metadata associated with the chunk
  similarity float                  -- Calculated similarity score (higher = more relevant)
)
language plpgsql                    -- Using PL/pgSQL procedural language
as $$
#variable_conflict use_column       -- Resolve naming conflicts by using column names
begin
  return query
  select
  id,
  url,
  chunk_number,
  title,
  summary,
  content,
  metadata,
  1 - (site_pages.embedding <=> query_embedding) as similarity  -- Convert distance to similarity score
  from site_pages
  where metadata @> filter         -- Filter records where metadata contains the filter JSON
  order by site_pages.embedding <=> query_embedding  -- Order by cosine distance (smaller = more similar)
  limit match_count;               -- Return only the requested number of matches
end;
$$;

-- =================================================================
-- UTILITY FUNCTIONS
-- =================================================================
-- Function to retrieve all unique source values from the metadata
create function get_distinct_sources()
returns SETOF text -- Returns a set (multiple rows) of text values
language sql -- The function body is a simple SQL query
STABLE -- Indicates the function cannot modify the database and returns the same results for the same arguments within a single scan
as $$
  -- Select the distinct values for the key 'source' from the metadata column
  -- The ->> operator extracts the JSON object field as text
  select distinct metadata ->> 'source'
  from site_pages
  order by 1; -- Order the results alphabetically
$$;

-- =================================================================
-- SECURITY SETTINGS (Supabase specific)
-- =================================================================
-- Enable row level security for fine-grained access control
alter table site_pages enable row level security;

-- Create a policy that allows anyone to read the site_pages data
-- This is suitable for public-facing documentation/content
create policy "Allow public read access"
  on site_pages
  for select
  to public
  using (true);