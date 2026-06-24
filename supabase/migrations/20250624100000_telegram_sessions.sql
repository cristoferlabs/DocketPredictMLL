-- Telegram sessions (reemplaza WhatsApp como canal principal)

CREATE TABLE IF NOT EXISTS public.telegram_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_hash   TEXT NOT NULL UNIQUE,
    last_intent TEXT,
    context     JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.telegram_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES public.telegram_sessions(id) ON DELETE CASCADE,
    direction       public.message_direction NOT NULL,
    content         TEXT NOT NULL,
    telegram_msg_id TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_telegram_messages_session ON public.telegram_messages (session_id, created_at DESC);

ALTER TABLE public.telegram_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.telegram_messages ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS telegram_sessions_deny ON public.telegram_sessions;
DROP POLICY IF EXISTS telegram_messages_deny ON public.telegram_messages;
CREATE POLICY telegram_sessions_deny ON public.telegram_sessions FOR ALL USING (false);
CREATE POLICY telegram_messages_deny ON public.telegram_messages FOR ALL USING (false);

CREATE TRIGGER telegram_sessions_updated_at BEFORE UPDATE ON public.telegram_sessions
    FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();
