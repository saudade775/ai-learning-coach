import sqlite3
import os


DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "data.db"
)


def get_connection():
    return sqlite3.connect(DB_PATH)


# ============================================================
# 初始化数据库
# ============================================================

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # 用户档案
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            level TEXT,
            direction TEXT,
            time_per_day TEXT,
            preference TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # API 配置
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT,
            api_key TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 聊天记录
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 文件记录
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # 记忆
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# ============================================================
# 用户档案
# ============================================================

def save_profile(name, level, direction, time_per_day, preference):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM profile")
    cursor.execute(
        """
        INSERT INTO profile (name, level, direction, time_per_day, preference)
        VALUES (?, ?, ?, ?, ?)
        """,
        (name, level, direction, time_per_day, preference)
    )
    conn.commit()
    conn.close()


def load_profile():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name, level, direction, time_per_day, preference FROM profile LIMIT 1"
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "name": row[0] or "",
            "level": row[1] or "",
            "direction": row[2] or "",
            "time_per_day": row[3] or "",
            "preference": row[4] or ""
        }
    return {"name": "", "level": "", "direction": "", "time_per_day": "", "preference": ""}


def has_profile():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM profile LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row is not None


def clear_profile():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM profile")
    conn.commit()
    conn.close()


# ============================================================
# API 配置
# ============================================================

def save_api_config(provider, api_key, model):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM api_config")
    cursor.execute(
        "INSERT INTO api_config (provider, api_key, model) VALUES (?, ?, ?)",
        (provider, api_key, model)
    )
    conn.commit()
    conn.close()


def load_api_config():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT provider, api_key, model FROM api_config LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"provider": row[0], "api_key": row[1], "model": row[2]}
    return {"provider": "", "api_key": "", "model": ""}


# ============================================================
# 聊天
# ============================================================

def save_message(role, content):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (role, content) VALUES (?, ?)",
        (role, content)
    )
    conn.commit()
    conn.close()


def load_messages():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def clear_messages():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages")
    conn.commit()
    conn.close()


# ============================================================
# 文件
# ============================================================

def save_file(filename):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO files(filename) VALUES(?)", (filename,))
    conn.commit()
    conn.close()


def load_files():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM files ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def clear_files():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM files")
    conn.commit()
    conn.close()


# ============================================================
# 记忆
# ============================================================

def save_memory(content):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO memories(content) VALUES(?)", (content,))
    conn.commit()
    conn.close()


def load_memories(limit=5):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT content FROM memories ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [r[0] for r in rows]


def clear_questionnaire_memory():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM memories WHERE content LIKE ?",
        ("用户问卷信息：%",)
    )
    conn.commit()
    conn.close()
# ============================================================
# 试用次数（持久化，删文件/聊天不会重置）
# ============================================================

def get_trial_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            used_count INTEGER DEFAULT 0
        )
    """)
    cursor.execute("SELECT used_count FROM trial LIMIT 1")
    row = cursor.fetchone()
    if not row:
        cursor.execute("INSERT INTO trial (used_count) VALUES (0)")
        conn.commit()
        used = 0
    else:
        used = row[0]
    conn.close()
    return used


def increase_trial_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE trial SET used_count = used_count + 1")
    conn.commit()
    conn.close()


def reset_trial_count():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE trial SET used_count = 0")
    conn.commit()
    conn.close()