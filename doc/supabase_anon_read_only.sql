-- ====================================================================
-- ONE-TIME SECURITY FIX: make the public anon key read-only
-- --------------------------------------------------------------------
-- Changes PERMISSIONS ONLY. It does not drop, re-create, or modify any
-- table or row; all recommendation history stays as it is.
--
-- Why: the anon key is designed to be public (it is meant to be embedded
-- in web/mobile apps), yet the original schema let it insert, update and
-- delete every row. After this script only the secret service_role key can
-- write; anon and signed-in users can still read.
--
-- BEFORE running: switch the pipeline host to the service_role key
-- (SUPABASE_KEY in its .env) and confirm the sanity check reports
-- role 'service_role'. Uploads and settlement made with the anon key are
-- rejected once this script has run.
--
-- Run once in the Supabase SQL Editor:
--   https://supabase.com/dashboard/project/<your-project-id>/sql/new
-- The final query prints the resulting permissions. To undo, run the
-- ROLLBACK block at the bottom.
-- ====================================================================

BEGIN;

-- 1. Table privileges: anon/authenticated read-only, service_role full access.
REVOKE ALL ON public.qlib_daily_top10_recommend, public.qlib_weekly_top10_recommend
    FROM anon, authenticated;
GRANT SELECT ON public.qlib_daily_top10_recommend, public.qlib_weekly_top10_recommend
    TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON public.qlib_daily_top10_recommend, public.qlib_weekly_top10_recommend
    TO service_role;

-- 2. Row Level Security: replace the allow-everything policies with read-only ones.
--    (service_role bypasses RLS, so it needs no policy.)
DROP POLICY IF EXISTS "anon_qlib_daily_policy" ON public.qlib_daily_top10_recommend;
DROP POLICY IF EXISTS "qlib_daily_read" ON public.qlib_daily_top10_recommend;
CREATE POLICY "qlib_daily_read" ON public.qlib_daily_top10_recommend
    FOR SELECT TO anon, authenticated USING (true);

DROP POLICY IF EXISTS "anon_qlib_weekly_policy" ON public.qlib_weekly_top10_recommend;
DROP POLICY IF EXISTS "qlib_weekly_read" ON public.qlib_weekly_top10_recommend;
CREATE POLICY "qlib_weekly_read" ON public.qlib_weekly_top10_recommend
    FOR SELECT TO anon, authenticated USING (true);

COMMIT;

-- 3. Verify. Expected: anon / authenticated -> only can_select = true;
--    service_role -> all four true.
SELECT t.tbl AS table_name,
       r.role,
       has_table_privilege(r.role, t.tbl, 'SELECT') AS can_select,
       has_table_privilege(r.role, t.tbl, 'INSERT') AS can_insert,
       has_table_privilege(r.role, t.tbl, 'UPDATE') AS can_update,
       has_table_privilege(r.role, t.tbl, 'DELETE') AS can_delete
FROM (VALUES ('public.qlib_daily_top10_recommend'),
             ('public.qlib_weekly_top10_recommend')) AS t(tbl)
CROSS JOIN (VALUES ('anon'), ('authenticated'), ('service_role')) AS r(role)
ORDER BY t.tbl, r.role;

-- --------------------------------------------------------------------
-- ROLLBACK (restores the old behaviour where the anon key can write):
-- --------------------------------------------------------------------
-- BEGIN;
-- GRANT SELECT, INSERT, UPDATE, DELETE
--     ON public.qlib_daily_top10_recommend, public.qlib_weekly_top10_recommend
--     TO anon, authenticated;
-- DROP POLICY IF EXISTS "qlib_daily_read" ON public.qlib_daily_top10_recommend;
-- CREATE POLICY "anon_qlib_daily_policy" ON public.qlib_daily_top10_recommend
--     FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
-- DROP POLICY IF EXISTS "qlib_weekly_read" ON public.qlib_weekly_top10_recommend;
-- CREATE POLICY "anon_qlib_weekly_policy" ON public.qlib_weekly_top10_recommend
--     FOR ALL TO anon, authenticated USING (true) WITH CHECK (true);
-- COMMIT;
