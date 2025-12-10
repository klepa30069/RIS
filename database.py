# database.py
import sqlite3
import pandas as pd
import os

DB_PATH = 'students.db'
CSV_PATH = 'Data.csv'

def init_db():
    """Инициализация базы данных и создание таблиц"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Таблица учеников
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age REAL NOT NULL,
            diagnosis TEXT
        )
    ''')

    # Таблица игр
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    ''')

    # Таблица назначений (заданий)
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

    conn.commit()
    conn.close()

def load_data_from_csv_to_db():
    """Загружает данные из CSV в базу данных, если таблица students пуста."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Проверяем, есть ли ученики в базе
    cursor.execute("SELECT COUNT(*) FROM students")
    count = cursor.fetchone()[0]
    if count > 0:
        print("Данные уже загружены в базу данных.")
        conn.close()
        return

    print(f"Загружаю данные из {CSV_PATH} в базу данных...")

    try:
        # Загружаем CSV
        df = pd.read_csv(CSV_PATH, sep=';', encoding='cp1251')
        print(f"CSV файл {CSV_PATH} успешно загружен. Количество строк: {len(df)}")
        print(f"Названия столбцов: {df.columns.tolist()}")
        print(f"Первые 2 строки данных:\n{df.head(2)}")

        # Загружаем учеников
        for index, row in df.iterrows():
            name = row['ФИО']
            age_str = str(row['Возраст']).replace(',', '.')
            try:
                age = float(age_str)
            except ValueError:
                print(f"Предупреждение (строка {index+1}): Невозможно преобразовать возраст '{row['Возраст']}' для {name} в float. Пропускаю.")
                continue
            diagnosis = row['диагноз']

            # Вставляем ученика
            cursor.execute("INSERT INTO students (name, age, diagnosis) VALUES (?, ?, ?)", (name, age, diagnosis))
            student_id = cursor.lastrowid

            # Получаем игры для ученика
            games_str = row['игры']
            if pd.notna(games_str) and games_str.strip() != '':
                games_list = [g.strip() for g in games_str.split(',')]
                for game_name in games_list:
                    if game_name:
                        # Вставляем игру (если не существует)
                        cursor.execute("INSERT OR IGNORE INTO games (name) VALUES (?)", (game_name,))
                        cursor.execute("SELECT id FROM games WHERE name = ?", (game_name,))
                        game_id = cursor.fetchone()
                        if game_id:
                            game_id = game_id[0]
                            # Вставляем назначение
                            from datetime import date
                            current_date = date.today().strftime('%Y-%m-%d')
                            cursor.execute("INSERT INTO assignments (student_id, game_id, date) VALUES (?, ?, ?)",
                                           (student_id, game_id, current_date))
            else:
                print(f"Предупреждение (строка {index+1}): У ученика {name} нет назначенных игр.")

        conn.commit()
        print("Данные успешно загружены из CSV в базу данных.")

    except FileNotFoundError:
        print(f"Файл {CSV_PATH} не найден. Пропускаю загрузку из CSV.")
    except Exception as e:
        print(f"Ошибка при загрузке данных из CSV: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_all_students():
    """Получить всех учеников"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, diagnosis FROM students ORDER BY name")
    students = cursor.fetchall()
    conn.close()
    return students

def add_student(name, age, diagnosis):
    """Добавить нового ученика"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO students (name, age, diagnosis) VALUES (?, ?, ?)", (name, age, diagnosis))
    conn.commit()
    student_id = cursor.lastrowid
    conn.close()
    return student_id

def get_student(student_id):
    """Получить информацию об ученике по ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, diagnosis FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()
    conn.close()
    return student

def get_student_games(student_id):
    """Получить список игр, назначенных ученику"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT g.name, a.date
        FROM assignments a
        JOIN games g ON a.game_id = g.id
        WHERE a.student_id = ?
        ORDER BY a.date DESC
    """, (student_id,))
    games = cursor.fetchall()
    conn.close()
    return games

def add_game(game_name):
    """Добавить игру в базу данных (если не существует)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT OR IGNORE INTO games (name) VALUES (?)", (game_name,))
        conn.commit()
        cursor.execute("SELECT id FROM games WHERE name = ?", (game_name,))
        game_id = cursor.fetchone()
        if game_id:
            game_id = game_id[0]
        else:
            raise Exception(f"Не удалось получить ID для игры {game_name}")
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
    return game_id

def get_all_games():
    """Получить все уникальные игры из базы данных"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM games ORDER BY name")
    games = [row[0] for row in cursor.fetchall()]
    conn.close()
    return games

def add_assignment(student_id, game_id, date_str):
    """Назначить задание ученику"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO assignments (student_id, game_id, date) VALUES (?, ?, ?)",
                   (student_id, game_id, date_str))
    conn.commit()
    conn.close()

def get_student_data_for_model(student_id):
    """Получить данные ученика для использования в модели кластеризации"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, diagnosis FROM students WHERE id = ?", (student_id,))
    student = cursor.fetchone()
    if not student:
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

    conn.close()
    return {
        'name': name,
        'age': age,
        'diagnosis': diagnosis,
        'played_games': played_games
    }

def update_student(student_id, name, age, diagnosis):
    """Обновить данные ученика"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE students SET name = ?, age = ?, diagnosis = ? WHERE id = ?", (name, age, diagnosis, student_id))
    conn.commit()
    conn.close()

# --- НОВАЯ ФУНКЦИЯ ---
def delete_student(student_id):
    """Удалить ученика и все его назначения"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Удаляем назначения, связанные с учеником
    cursor.execute("DELETE FROM assignments WHERE student_id = ?", (student_id,))
    # Удаляем самого ученика
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()
# --- КОНЕЦ НОВОЙ ФУНКЦИИ ---
