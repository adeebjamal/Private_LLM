import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import query_constants

logger = logging.getLogger(__name__)

def get_db_connection():
    """Create and return a database connection."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    try:
        conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def init_db():
    """Create tables if they don't exist."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Table 1: Conversations
        cursor.execute(query_constants.CREATE_CONVERSATIONS_TABLE)
        
        # Table 2: Messages
        cursor.execute(query_constants.CREATE_MESSAGES_TABLE)
        
        conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def create_conversation(title: str):
    """Insert new conversation, return it."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            query_constants.INSERT_CONVERSATION,
            (title,)
        )
        new_conv = cursor.fetchone()
        conn.commit()
        return dict(new_conv)
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def get_all_conversations():
    """Return all conversations with message count."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query_constants.GET_ALL_CONVERSATIONS)
        conversations = cursor.fetchall()
        return [dict(conv) for conv in conversations]
    except Exception as e:
        logger.error(f"Error getting all conversations: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_conversation(id: int):
    """Return single conversation by ID."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query_constants.GET_CONVERSATION_BY_ID, (id,))
        conv = cursor.fetchone()
        return dict(conv) if conv else None
    except Exception as e:
        logger.error(f"Error getting conversation by id: {e}")
        raise
    finally:
        if conn:
            conn.close()

def save_message(conv_id: int, query: str, resp: str):
    """Insert one Q&A row into messages."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            query_constants.INSERT_MESSAGE,
            (conv_id, query, resp)
        )
        new_msg = cursor.fetchone()
        conn.commit()
        return dict(new_msg)
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def get_messages(conv_id: int, limit: int = 10):
    """Return last N messages as LLM history format."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            query_constants.GET_MESSAGES_FOR_LLM,
            (conv_id, limit)
        )
        # Fetching returns descending order (latest first), we need chronological for LLM prompt
        rows = cursor.fetchall()
        rows.reverse()
        
        history = []
        for row in rows:
            history.append({"role": "user", "content": row["user_query"]})
            history.append({"role": "assistant", "content": row["response"]})
            
        return history
    except Exception as e:
        logger.error(f"Error getting messages for LLM context: {e}")
        raise
    finally:
        if conn:
            conn.close()

def get_messages_paginated(conv_id: int, start_row: int, end_row: int):
    """Return paginated messages."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Calculate pagination
        offset = start_row - 1
        limit = end_row - start_row + 1
        
        # Get total messages count
        cursor.execute(query_constants.COUNT_MESSAGES, (conv_id,))
        total_messages = cursor.fetchone()["total"]
        
        # Get paginated messages
        cursor.execute(
            query_constants.GET_PAGINATED_MESSAGES,
            (conv_id, offset, limit)
        )
        messages = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_messages": total_messages,
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error getting paginated messages: {e}")
        raise
    finally:
        if conn:
            conn.close()

def rename_conversation(conv_id: int, new_name: str):
    """Update a conversation's title."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(query_constants.RENAME_CONVERSATION, (new_name, conv_id))
        updated = cursor.fetchone()
        conn.commit()
        return dict(updated) if updated else None
    except Exception as e:
        logger.error(f"Error renaming conversation: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def delete_conversation(conv_id: int):
    """Delete a conversation and all its messages."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Delete messages first (also handled by CASCADE, but explicit is safer)
        cursor.execute(query_constants.DELETE_MESSAGES_BY_CONVERSATION, (conv_id,))
        # Delete the conversation itself
        cursor.execute(query_constants.DELETE_CONVERSATION, (conv_id,))
        deleted = cursor.fetchone()
        conn.commit()
        return dict(deleted) if deleted else None
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
