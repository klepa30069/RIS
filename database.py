# database.py (переименуйте database_postgres.py в database.py)
import os
import pandas as pd
from datetime import date

# Получаем URL базы данных из переменных окружения Railway
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    # Если есть DATABASE_URL, используем PostgreSQL
    try:
        import psycopg
        from psycopg.rows import dict_row

        print("Используется PostgreSQL с psycopg3")
        USE_POSTGRES = True
    except ImportError:
        print("psycopg3 не установлен, проверьте requirements.txt")
        USE_POSTGRES = False
else:
    # Для локальной разработки используем SQLite
    print("DATABASE_URL не найден, используем SQLite для локальной разработки")
    USE_POSTGRES = False
    import sqlite3


def get_connection():
    """Получить соединение с базой данных"""
    if not USE_POSTGRES:
        import sqlite3
        return sqlite3.connect('students.db')

    try:
        # Используем psycopg3 для PostgreSQL
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        return conn
    except Exception as e:
        print(f"Ошибка подключения к PostgreSQL: {e}")
        # Fallback на SQLite для совместимости
        import sqlite3
        return sqlite3.connect('students.db')


def init_db():
    """Инициализация базы данных и создание таблиц"""
    conn = get_connection()

    if not USE_POSTGRES:
        # SQLite
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age REAL NOT NULL,
                diagnosis TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                date DATE NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students (id),
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
        ''')
    else:
        # PostgreSQL
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                age REAL NOT NULL,
                diagnosis TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignments (
                id SERIAL PRIMARY KEY,
                student_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                date DATE NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students (id) ON DELETE CASCADE,
                FOREIGN KEY (game_id) REFERENCES games (id)
            )
        ''')

    conn.commit()
    cursor.close()
    conn.close()
    print("База данных инициализирована")


# ... остальные функции остаются аналогичными, но с проверкой USE_POSTGRES ...

def get_all_students():
    """Получить всех учеников"""
    conn = get_connection()

    if not USE_POSTGRES:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis FROM students ORDER BY name")
        students = cursor.fetchall()
        cursor.close()
        conn.close()
        return students
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis FROM students ORDER BY name")
        students = cursor.fetchall()
        cursor.close()
        conn.close()
        # Преобразуем словари в кортежи для совместимости
        return [(s['id'], s['name'], s['age'], s['diagnosis']) for s in students]


def get_all_games():
    """Получить все уникальные игры из базы данных"""
    conn = get_connection()

    if not USE_POSTGRES:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM games ORDER BY name")
        games = [row[0] for row in cursor.fetchall()]
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM games ORDER BY name")
        games = [row['name'] for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return games


def execute_query(query, params=None, fetch=False):
    """Универсальная функция выполнения запроса"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        if params:
            query = query.replace('%s', '?')
            cursor.execute(query, params)
        else:
            cursor.execute(query)

        if fetch:
            result = cursor.fetchall()
        else:
            result = cursor.lastrowid if query.strip().upper().startswith('INSERT') else None

        if not fetch and query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
            conn.commit()

        cursor.close()
        conn.close()

        if fetch:
            # Преобразуем строки SQLite в словари для совместимости
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in result]
            return result
        return result
    else:
        # PostgreSQL
        cursor = conn.cursor()

        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if fetch:
                result = cursor.fetchall()
                # psycopg3 возвращает список словарей напрямую
                # psycopg2 через RealDictCursor тоже
                return result
            else:
                # Для INSERT с RETURNING
                if query.strip().upper().startswith('INSERT') and 'RETURNING' in query.upper():
                    result = cursor.fetchone()
                    conn.commit()
                    cursor.close()
                    conn.close()
                    return result['id'] if result else None

                conn.commit()
                cursor.close()
                conn.close()
                return cursor.rowcount

        except Exception as e:
            conn.rollback()
            cursor.close()
            conn.close()
            raise e


# Упрощенные версии функций
def load_data_from_csv_to_db():
    """Загружает данные из CSV в базу данных, если таблица students пуста."""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        count = cursor.fetchone()[0]
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM students")
        count = cursor.fetchone()['count']

    if count > 0:
        print("Данные уже загружены в базу данных.")
        cursor.close()
        conn.close()
        return

    print("Загружаю данные из CSV в базу данных...")

    try:
        csv_path = 'Data.csv'
        if not os.path.exists(csv_path):
            print(f"Файл {csv_path} не найден. Ищем в других местах...")
            # Пробуем найти файл
            for file in os.listdir('.'):
                if file.lower().endswith('.csv'):
                    csv_path = file
                    print(f"Найден CSV файл: {csv_path}")
                    break

        df = pd.read_csv(csv_path, sep=';', encoding='cp1251')
        print(f"CSV файл успешно загружен. Количество строк: {len(df)}")

        for index, row in df.iterrows():
            name = row['ФИО']
            age_str = str(row['Возраст']).replace(',', '.')
            try:
                age = float(age_str)
            except ValueError:
                print(f"Предупреждение (строка {index + 1}): Невозможно преобразовать возраст для {name}. Пропускаю.")
                continue
            diagnosis = row['диагноз']

            # Вставляем ученика
            if isinstance(conn, sqlite3.Connection):
                cursor.execute(
                    "INSERT INTO students (name, age, diagnosis) VALUES (?, ?, ?)",
                    (name, age, diagnosis)
                )
                student_id = cursor.lastrowid
            else:
                cursor.execute(
                    "INSERT INTO students (name, age, diagnosis) VALUES (%s, %s, %s) RETURNING id",
                    (name, age, diagnosis)
                )
                student_id = cursor.fetchone()['id']

            # Получаем игры для ученика
            games_str = row['игры']
            if pd.notna(games_str) and games_str.strip() != '':
                games_list = [g.strip() for g in games_str.split(',')]
                for game_name in games_list:
                    if game_name:
                        # Вставляем игру (если не существует)
                        if isinstance(conn, sqlite3.Connection):
                            cursor.execute("INSERT OR IGNORE INTO games (name) VALUES (?)", (game_name,))
                            cursor.execute("SELECT id FROM games WHERE name = ?", (game_name,))
                            game_result = cursor.fetchone()
                            if game_result:
                                game_id = game_result[0]
                        else:
                            # Для PostgreSQL
                            cursor.execute(
                                "INSERT INTO games (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                                (game_name,)
                            )
                            cursor.execute("SELECT id FROM games WHERE name = %s", (game_name,))
                            game_result = cursor.fetchone()
                            game_id = game_result['id'] if game_result else None

                        if game_id:
                            current_date = date.today().strftime('%Y-%m-%d')
                            cursor.execute(
                                "INSERT INTO assignments (student_id, game_id, date) VALUES (?, ?, ?)",
                                (student_id, game_id, current_date)
                            )

        conn.commit()
        print("Данные успешно загружены из CSV в базу данных.")

    except Exception as e:
        print(f"Ошибка при загрузке данных из CSV: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def add_student(name, age, diagnosis):
    """Добавить нового ученика"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (name, age, diagnosis) VALUES (?, ?, ?)",
            (name, age, diagnosis)
        )
        student_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        return student_id
    else:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (name, age, diagnosis) VALUES (%s, %s, %s) RETURNING id",
            (name, age, diagnosis)
        )
        student_id = cursor.fetchone()['id']
        conn.commit()
        cursor.close()
        conn.close()
        return student_id


def get_student(student_id):
    """Получить информацию об ученике по ID"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        cursor.close()
        conn.close()
        return student
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis FROM students WHERE id = %s", (student_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return (result['id'], result['name'], result['age'], result['diagnosis'])
        return None


def get_student_games(student_id):
    """Получить список игр, назначенных ученику"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("""
            SELECT g.name, a.date
            FROM assignments a
            JOIN games g ON a.game_id = g.id
            WHERE a.student_id = ?
            ORDER BY a.date DESC
        """, (student_id,))
        games = cursor.fetchall()
        cursor.close()
        conn.close()
        return games
    else:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT g.name, a.date
            FROM assignments a
            JOIN games g ON a.game_id = g.id
            WHERE a.student_id = %s
            ORDER BY a.date DESC
        """, (student_id,))
        games = cursor.fetchall()
        cursor.close()
        conn.close()
        # Преобразуем в список кортежей
        return [(g['name'], g['date']) for g in games]


def add_game(game_name):
    """Добавить игру в базу данных (если не существует)"""
    conn = get_connection()

    try:
        if isinstance(conn, sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO games (name) VALUES (?)", (game_name,))
            cursor.execute("SELECT id FROM games WHERE name = ?", (game_name,))
            game_result = cursor.fetchone()
            if game_result:
                game_id = game_result[0]
            else:
                raise Exception(f"Не удалось получить ID для игры {game_name}")
        else:
            cursor = conn.cursor()
            # Для PostgreSQL
            cursor.execute(
                "INSERT INTO games (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
                (game_name,)
            )
            game_result = cursor.fetchone()

            if game_result:
                game_id = game_result['id']
            else:
                # Игра уже существует
                cursor.execute("SELECT id FROM games WHERE name = %s", (game_name,))
                game_result = cursor.fetchone()
                if game_result:
                    game_id = game_result['id']
                else:
                    raise Exception(f"Не удалось получить ID для игры {game_name}")

        conn.commit()
        return game_id

    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        raise e
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()


def add_assignment(student_id, game_id, date_str):
    """Назначить задание ученику"""
    query = "INSERT INTO assignments (student_id, game_id, date) VALUES (%s, %s, %s)"
    execute_query(query, (student_id, game_id, date_str))


def get_student_data_for_model(student_id):
    """Получить данные ученика для использования в модели кластеризации"""
    # Получаем данные ученика
    student_query = "SELECT name, age, diagnosis FROM students WHERE id = %s"
    student_result = execute_query(student_query, (student_id,), fetch=True)

    if not student_result:
        return None

    student_data = student_result[0]

    # Получаем игры ученика
    games_query = """
        SELECT g.name
        FROM assignments a
        JOIN games g ON a.game_id = g.id
        WHERE a.student_id = %s
    """
    games_result = execute_query(games_query, (student_id,), fetch=True)

    played_games = []
    for game in games_result:
        if isinstance(game, dict):
            played_games.append(game['name'])
        elif isinstance(game, tuple):
            played_games.append(game[0])
        else:
            played_games.append(game)

    return {
        'name': student_data['name'] if isinstance(student_data, dict) else student_data[0],
        'age': student_data['age'] if isinstance(student_data, dict) else student_data[1],
        'diagnosis': student_data['diagnosis'] if isinstance(student_data, dict) else student_data[2],
        'played_games': played_games
    }


def update_student(student_id, name, age, diagnosis):
    """Обновить данные ученика"""
    query = "UPDATE students SET name = %s, age = %s, diagnosis = %s WHERE id = %s"
    execute_query(query, (name, age, diagnosis, student_id))


def delete_student(student_id):
    """Удалить ученика и все его назначения"""
    query = "DELETE FROM students WHERE id = %s"
    execute_query(query, (student_id,))


# Импортируем sqlite3 для локальной разработки
import sqlite3
