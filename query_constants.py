CREATE_CONVERSATIONS_TABLE = """
            CREATE TABLE IF NOT EXISTS conversations (
                id         SERIAL PRIMARY KEY,
                title      VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """

CREATE_MESSAGES_TABLE = """
            CREATE TABLE IF NOT EXISTS messages (
                id              SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                user_query      TEXT NOT NULL,
                response        TEXT NOT NULL,
                created_at      TIMESTAMP DEFAULT NOW()
            );
        """

INSERT_CONVERSATION = "INSERT INTO conversations (title) VALUES (%s) RETURNING id, title, created_at;"

GET_ALL_CONVERSATIONS = """
            SELECT c.id, c.title, c.created_at, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id, c.title, c.created_at
            ORDER BY c.created_at DESC;
        """

GET_CONVERSATION_BY_ID = "SELECT id, title, created_at FROM conversations WHERE id = %s;"

INSERT_MESSAGE = """
            INSERT INTO messages (conversation_id, user_query, response) 
            VALUES (%s, %s, %s) 
            RETURNING id, conversation_id, user_query, response, created_at;
            """

GET_MESSAGES_FOR_LLM = """
            SELECT user_query, response 
            FROM messages 
            WHERE conversation_id = %s 
            ORDER BY created_at DESC 
            LIMIT %s;
            """

COUNT_MESSAGES = "SELECT COUNT(*) as total FROM messages WHERE conversation_id = %s;"

GET_PAGINATED_MESSAGES = """
            SELECT id, user_query, response, created_at 
            FROM messages 
            WHERE conversation_id = %s 
            ORDER BY created_at ASC 
            OFFSET %s LIMIT %s;
            """
