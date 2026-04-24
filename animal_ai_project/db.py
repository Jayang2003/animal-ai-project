import aiosqlite
import json
from datetime import datetime

DB_PATH = "chat_history.db"

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          TEXT PRIMARY KEY,
                username    TEXT NOT NULL,
                label       TEXT,
                created_at  TEXT,
                updated_at  TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                username    TEXT NOT NULL,
                question    TEXT,
                answer      TEXT,
                created_at  TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        await db.commit()

# ── SESSIONS ───────────────────────────────────────────────────────────────

async def save_session(session_id: str, username: str, label: str):
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO sessions (id, username, label, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                label      = excluded.label,
                updated_at = excluded.updated_at
        """, (session_id, username, label, now, now))
        await db.commit()

async def rename_session(session_id: str, new_name: str):
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            UPDATE sessions SET label = ?, updated_at = ? WHERE id = ?
        """, (new_name, now, session_id))
        await db.commit()

async def delete_session(session_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions  WHERE id = ?",         (session_id,))
        await db.commit()

async def get_sessions(username: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT s.id, s.label, s.updated_at,
                   COUNT(m.id) as message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            WHERE s.username = ?
            GROUP BY s.id
            ORDER BY s.updated_at DESC
        """, (username,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

# ── MESSAGES ───────────────────────────────────────────────────────────────

async def save_message(session_id: str, username: str, question: str, answer: str):
    now = datetime.utcnow().isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO messages (session_id, username, question, answer, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, username, question, answer, now))
        # Update session updated_at
        await db.execute("""
            UPDATE sessions SET updated_at = ? WHERE id = ?
        """, (now, session_id))
        await db.commit()

async def get_messages(session_id: str):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT question, answer, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (session_id,))
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]