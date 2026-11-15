import aiomysql
import json
import os
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

pool = None
async def init_db_pool():
    global pool
    pool = await aiomysql.create_pool(
        host = os.getenv('DB_HOST','127.0.0.1'),
        port = int(os.getenv('DB_PORT',3307)),
        user = 'root',
        password = os.getenv('DB_PASSWORD'),
        db = 'rag_db',
        charset = 'utf8mb4',
        cursorclass = aiomysql.DictCursor,
        minsize = 1,
        maxsize = 10,
        autocommit = False
    )

async def save_message_to_mysql(thread_id:str,role:str,content:str,contexts:list=None):
    if not pool:
        return
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                await cursor.execute('INSERT IGNORE INTO chat_threads (id,user_id,title) VALUES (%s,%s,%s)',(thread_id,'anonymous',content[:20]))
                contexts_json = json.dumps(contexts,ensure_ascii=False) if contexts else None
                await cursor.execute('INSERT INTO chat_messages (thread_id,role,content,contexts) VALUES (%s,%s,%s,%s)',
                                     (thread_id, role, content, contexts_json))
                await conn.commit()
            except Exception as e:
                print(f"MySQL写入失败:{e}")
                await conn.rollback()

async def load_history_from_mysql(thread_id:str)->list:
    if not pool:
        return []
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                await cursor.execute(
                    'SELECT role,content,contexts FROM chat_messages WHERE thread_id = %s ORDER BY id ASC',
                    (thread_id,))
                rows = await cursor.fetchall()
                for row in rows:
                    if row.get('contexts'):
                        row['contexts'] = json.loads(row['contexts'])
                return rows
            except Exception as e:
                print(f'MySQL读取失败:{e}')
                return []




