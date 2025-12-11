# database_postgres.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from datetime import date

# Добавляем импорт sqlite3 для локальной разработки
import sqlite3

# Получаем URL базы данных из переменных окружения Railway
DATABASE_URL = os.environ.get('DATABASE_URL')


def get_connection():
    """Получить соединение с PostgreSQL"""
    if not DATABASE_URL:
        # Для локальной разработки можно использовать SQLite
        import sqlite3
        return sqlite3.connect('students.db')

    # Railway предоставляет DATABASE_URL в формате postgresql://...
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn


def init_db():
    """Инициализация базы данных и создание таблиц"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        # Код для SQLite (локальная разработка)
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
        # Код для PostgreSQL (продакшн)
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
        # Загружаем CSV из локального файла
        csv_path = 'Data.csv'
        df = pd.read_csv(csv_path, sep=';', encoding='cp1251')
        print(f"CSV файл успешно загружен. Количество строк: {len(df)}")

        # Загружаем учеников
        for index, row in df.iterrows():
            name = row['ФИО']
            age_str = str(row['Возраст']).replace(',', '.')
            try:
                age = float(age_str)
            except ValueError:
                print(
                    f"Предупреждение (строка {index + 1}): Невозможно преобразовать возраст '{row['Возраст']}' для {name} в float. Пропускаю.")
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
                            cursor.execute(
                                "INSERT INTO games (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
                                (game_name,)
                            )
                            game_result = cursor.fetchone()
                            if game_result:
                                game_id = game_result['id']
                            else:
                                cursor.execute("SELECT id FROM games WHERE name = %s", (game_name,))
                                game_result = cursor.fetchone()
                                game_id = game_result['id'] if game_result else None

                        if game_id:
                            # Вставляем назначение
                            current_date = date.today().strftime('%Y-%m-%d')
                            if isinstance(conn, sqlite3.Connection):
                                cursor.execute(
                                    "INSERT INTO assignments (student_id, game_id, date) VALUES (?, ?, ?)",
                                    (student_id, game_id, current_date)
                                )
                            else:
                                cursor.execute(
                                    "INSERT INTO assignments (student_id, game_id, date) VALUES (%s, %s, %s)",
                                    (student_id, game_id, current_date)
                                )

        conn.commit()
        print("Данные успешно загружены из CSV в базу данных.")

    except Exception as e:
        print(f"Ошибка при загрузке данных из CSV: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


def get_all_students():
    """Получить всех учеников"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis FROM students ORDER BY name")
        students = cursor.fetchall()
        # SQLite возвращает кортежи
        result = students
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis FROM students ORDER BY name")
        students = cursor.fetchall()
        # Преобразуем RealDictRow в кортеж для совместимости
        result = [(s['id'], s['name'], s['age'], s['diagnosis']) for s in students]

    cursor.close()
    conn.close()
    return result


def add_student(name, age, diagnosis):
    """Добавить нового ученика"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("INSERT INTO students (name, age, diagnosis) VALUES (?, ?, ?)", (name, age, diagnosis))
        student_id = cursor.lastrowid
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
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, diagnosis FROM students WHERE id = %s", (student_id,))
        result = cursor.fetchone()
        student = (result['id'], result['name'], result['age'], result['diagnosis']) if result else None

    cursor.close()
    conn.close()
    return student


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
        # Преобразуем в список кортежей
        games = [(g['name'], g['date']) for g in games]

    cursor.close()
    conn.close()
    return games


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
            # Пытаемся вставить игру
            cursor.execute(
                "INSERT INTO games (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
                (game_name,)
            )
            game_result = cursor.fetchone()

            if game_result:
                game_id = game_result['id']
            else:
                # Игра уже существует, получаем ее ID
                cursor.execute("SELECT id FROM games WHERE name = %s", (game_name,))
                game_result = cursor.fetchone()
                if game_result:
                    game_id = game_result['id']
                else:
                    raise Exception(f"Не удалось получить ID для игры {game_name}")

        conn.commit()
        return game_id

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def get_all_games():
    """Получить все уникальные игры из базы данных"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
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


def add_assignment(student_id, game_id, date_str):
    """Назначить задание ученику"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO assignments (student_id, game_id, date) VALUES (?, ?, ?)",
            (student_id, game_id, date_str)
        )
    else:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO assignments (student_id, game_id, date) VALUES (%s, %s, %s)",
            (student_id, game_id, date_str)
        )

    conn.commit()
    cursor.close()
    conn.close()


def get_student_data_for_model(student_id):
    """Получить данные ученика для использования в модели кластеризации"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute("SELECT name, age, diagnosis FROM students WHERE id = ?", (student_id,))
        student = cursor.fetchone()
        if not student:
            cursor.close()
            conn.close()
            return None

        name, age, diagnosis = student

        cursor.execute("""
            SELECT g.name
            FROM assignments a
            JOIN games g ON a.game_id = g.id
            WHERE a.student_id = ?
        """, (student_id,))
        played_games = [row[0] for row in cursor.fetchall()]
    else:
        cursor = conn.cursor()
        cursor.execute("SELECT name, age, diagnosis FROM students WHERE id = %s", (student_id,))
        result = cursor.fetchone()
        if not result:
            cursor.close()
            conn.close()
            return None

        name = result['name']
        age = result['age']
        diagnosis = result['diagnosis']

        cursor.execute("""
            SELECT g.name
            FROM assignments a
            JOIN games g ON a.game_id = g.id
            WHERE a.student_id = %s
        """, (student_id,))
        played_games = [row['name'] for row in cursor.fetchall()]

    cursor.close()
    conn.close()

    return {
        'name': name,
        'age': age,
        'diagnosis': diagnosis,
        'played_games': played_games
    }


def update_student(student_id, name, age, diagnosis):
    """Обновить данные ученика"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE students SET name = ?, age = ?, diagnosis = ? WHERE id = ?",
            (name, age, diagnosis, student_id)
        )
    else:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE students SET name = %s, age = %s, diagnosis = %s WHERE id = %s",
            (name, age, diagnosis, student_id)
        )

    conn.commit()
    cursor.close()
    conn.close()


def delete_student(student_id):
    """Удалить ученика и все его назначения"""
    conn = get_connection()

    if isinstance(conn, sqlite3.Connection):
        cursor = conn.cursor()
        # В SQLite каскадное удаление должно быть настроено через внешние ключи
        cursor.execute("DELETE FROM assignments WHERE student_id = ?", (student_id,))
        cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    else:
        cursor = conn.cursor()
        # В PostgreSQL ON DELETE CASCADE автоматически удаляет связанные записи
        cursor.execute("DELETE FROM students WHERE id = %s", (student_id,))

    conn.commit()
    cursor.close()
    conn.close()

