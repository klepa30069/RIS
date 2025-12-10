# app.py
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import pickle
import os
from datetime import date # Добавим для даты по умолчанию

app = Flask(__name__)

# --- Импортируем функции работы с базой данных ---
from database import init_db, load_data_from_csv_to_db, get_all_students, add_student, get_student, get_student_games, add_game, get_all_games, add_assignment, get_student_data_for_model

# Путь к модели и скалеру
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
DB_PATH = 'students.db'

# Глобальные переменные для модели
kmeans_model = None
scaler = None
all_games_from_db = [] # Будет заполняться из базы данных
all_diagnoses_from_db = [] # Будет заполняться из базы данных

def load_model_and_data():
    global kmeans_model, scaler, all_games_from_db, all_diagnoses_from_db

    # Инициализируем базу данных
    init_db()

    # Загружаем данные из CSV в базу данных (если база пуста)
    load_data_from_csv_to_db()

    # Загружаем все игры и диагнозы из базы данных
    all_games_from_db = get_all_games()
    # Для диагнозов: собираем все уникальные диагнозы из таблицы студентов
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT diagnosis FROM students WHERE diagnosis IS NOT NULL AND diagnosis != ''")
    diagnoses = cursor.fetchall()
    all_diagnoses_from_db = []
    for diag_tuple in diagnoses:
        diag_str = diag_tuple[0]
        if diag_str:
            # Разбиваем строку диагнозов на список
            diag_list = [d.strip() for d in diag_str.split(',')]
            all_diagnoses_from_db.extend(diag_list)
    all_diagnoses_from_db = list(set(all_diagnoses_from_db)) # Уникальные
    all_diagnoses_from_db.sort()
    conn.close()

    # Загружаем модель и скалер
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            kmeans_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("Модель и данные успешно загружены.")
    else:
        print("Модель или скалер не найдены. Запустите обучение модели.")

# ... (остальная часть app.py остается без изменений, включая train_and_save_model и маршруты) ...

def prepare_patient_features_for_prediction(student_data):
    """
    Подготовка признаков для одного пациента (ученика) для предсказания.
    student_data: словарь с ключами 'age', 'diagnosis', 'played_games'
    """
    # Создаем пустой вектор признаков
    feature_vector = []

    # Признаки по играм (бинарные) - какие игры использует пациент
    for game in all_games_from_db:
        feature_vector.append(1 if game in student_data['played_games'] else 0)

    # Признаки по диагнозам (бинарные) - какие диагнозы у пациента
    current_diagnoses = []
    if student_data['diagnosis']:
        current_diagnoses = [d.strip() for d in student_data['diagnosis'].split(',') if d.strip()]
    for diagnosis in all_diagnoses_from_db:
        feature_vector.append(1 if diagnosis in current_diagnoses else 0)

    # Демографические признаки - возраст как числовой признак
    feature_vector.append(float(student_data['age']))

    return np.array([feature_vector])

def train_and_save_model():
    """
    Функция для обучения модели на основе данных из базы данных.
    Эту функцию нужно вызвать один раз для обучения модели.
    """
    # Получаем всех учеников из базы данных
    students = get_all_students()
    if len(students) == 0:
        print("Нет данных для обучения модели.")
        return

    # Создаем DataFrame для обучения
    data_list = []
    for student in students:
        student_id, name, age, diagnosis = student
        # Получаем игры, которые играет ученик
        played_games = [g[0] for g in get_student_games(student_id)] # g[0] - имя игры
        data_list.append({
            'ФИО': name,
            'Возраст': age,
            'диагноз': diagnosis,
            'игры': ', '.join(played_games) if played_games else ''
        })

    data = pd.DataFrame(data_list)

    # Подготовка признаков (аналогично ячейке 3 из Colab)
    def prepare_patient_features(data):
        data_processed = data.copy()
        data_processed['Возраст_число'] = data_processed['Возраст'].astype(str).str.replace(',', '.').astype(float)
        data_processed['игры_список'] = data_processed['игры'].str.split(', ')

        all_games = set()
        for games_list in data_processed['игры_список']:
            if isinstance(games_list, list):
                all_games.update([str(game).strip() for game in games_list])

        all_diagnoses = set()
        for diagnosis in data_processed['диагноз']:
            if pd.notna(diagnosis):
                diagnoses = str(diagnosis).split(', ')
                all_diagnoses.update([str(d).strip() for d in diagnoses])

        features = []
        patient_ids = []

        for _, row in data_processed.iterrows():
            feature_vector = []
            current_games = row['игры_список'] if isinstance(row['игры_список'], list) else []
            current_games = [str(g).strip() for g in current_games]
            for game in all_games:
                feature_vector.append(1 if str(game) in current_games else 0)

            current_diagnoses = str(row['диагноз']).split(', ') if pd.notna(row['диагноз']) else []
            current_diagnoses = [str(d).strip() for d in current_diagnoses]
            for diagnosis in all_diagnoses:
                feature_vector.append(1 if str(diagnosis) in current_diagnoses else 0)

            feature_vector.append(float(row['Возраст_число']))
            features.append(feature_vector)
            patient_ids.append(str(row['ФИО']))

        feature_names = list(all_games) + list(all_diagnoses) + ['Возраст']
        return np.array(features), feature_names, patient_ids, list(all_games), list(all_diagnoses), data_processed

    X, _, _, all_games_from_func, all_diagnoses_from_func, _ = prepare_patient_features(data)

    # Масштабирование и обучение
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Оптимальное количество кластеров (предположим 3)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Сохраняем модель и скалер
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Модель обучена и сохранена. K = {optimal_k}")

# --- Маршруты API ---

@app.route('/')
def index():
    """Главная страница со списком учеников"""
    students = get_all_students()
    html_template = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Ученики</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .student-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .student-card { background-color: #f0f0f0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .student-name { font-weight: bold; font-size: 1.2em; margin-bottom: 5px; }
        .student-age { margin-bottom: 5px; }
        .student-diagnosis { font-style: italic; }
        .add-button { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; text-decoration: none; }
        .add-button:hover { background-color: #45a049; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Список учеников</h1>
        <a href="/add_student" class="add-button">Добавить ученика</a>
    </div>

    <div class="student-grid">
        {% for student in students %}
            <div class="student-card">
                <div class="student-name">{{ student[1] }}</div>
                <div class="student-age">{{ student[2] }} лет</div>
                <div class="student-diagnosis">Диагнозы: {{ student[3] or 'N/A' }}</div>
                <a href="/student/{{ student[0] }}" style="display: inline-block; margin-top: 10px; text-decoration: none; color: #0066cc;">Перейти к ученику</a>
            </div>
        {% endfor %}
    </div>
</body>
</html>
    '''
    return render_template_string(html_template, students=students)

@app.route('/add_student', methods=['GET', 'POST'])
def add_student_page():
    """Страница для добавления нового ученика"""
    if request.method == 'POST':
        name = request.form.get('name')
        age = float(request.form.get('age'))
        diagnosis = request.form.get('diagnosis', '').strip()
        if not name or not age:
            return "Имя и возраст обязательны!", 400

        student_id = add_student(name, age, diagnosis)
        return redirect(url_for('index'))

    html_template = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Добавить ученика</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        form { max-width: 400px; }
        label { display: block; margin-bottom: 5px; }
        input, textarea { width: 100%; padding: 8px; margin-bottom: 15px; }
        button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .back-link { display: inline-block; margin-top: 10px; text-decoration: none; color: #0066cc; }
    </style>
</head>
<body>
    <h1>Добавить нового ученика</h1>
    <form method="POST">
        <label for="name">Имя:</label>
        <input type="text" id="name" name="name" required>

        <label for="age">Возраст:</label>
        <input type="number" id="age" name="age" step="0.1" required>

        <label for="diagnosis">Диагнозы (через запятую):</label>
        <textarea id="diagnosis" name="diagnosis" placeholder="Например: дислалия, дисграфия"></textarea>

        <button type="submit">Добавить ученика</button>
    </form>
    <a href="/" class="back-link">Назад к списку учеников</a>
</body>
</html>
    '''
    return render_template_string(html_template)

@app.route('/student/<int:student_id>')
def student_page(student_id):
    """Страница ученика с его играми и кнопкой добавить задание"""
    student = get_student(student_id)
    if not student:
        return "Ученик не найден", 404

    games = get_student_games(student_id)
    html_template = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>{{ student[1] }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
        .games-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .game-card { background-color: #f0f0f0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .game-name { font-weight: bold; font-size: 1.2em; margin-bottom: 5px; }
        .game-stats { margin-bottom: 5px; }
        .add-assignment-button { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; text-decoration: none; }
        .add-assignment-button:hover { background-color: #45a049; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ student[1] }} ({{ student[2] }} лет)</h1>
        <div>Диагнозы: {{ student[3] or 'N/A' }}</div>
        <a href="/student/{{ student[0] }}/add_assignment" class="add-assignment-button">Добавить задание</a>
    </div>

    <h2>Назначенные игры</h2>
    {% if games %}
        <div class="games-grid">
            {% for game in games %}
                <div class="game-card">
                    <div class="game-name">{{ game[0] }}</div>
                    <div class="game-stats">Играл: 10 раз</div>
                    <div class="game-stats">Время: 20 минут</div>
                    <!-- Здесь можно добавить дату назначения -->
                    <div class="game-stats">Дата: {{ game[1] }}</div>
                </div>
            {% endfor %}
        </div>
    {% else %}
        <p>У этого ученика пока нет назначенных игр.</p>
    {% endif %}

    <a href="/" style="display: inline-block; margin-top: 20px; text-decoration: none; color: #0066cc;">Назад к списку учеников</a>
</body>
</html>
    '''
    return render_template_string(html_template, student=student, games=games)

# ... (весь предыдущий код app.py остается без изменений) ...

@app.route('/student/<int:student_id>/add_assignment', methods=['GET', 'POST'])
def add_assignment_page(student_id):
    """Страница для добавления задания (выбор даты и рекомендаций)"""
    student = get_student(student_id)
    if not student:
        return "Ученик не найден", 404

    if request.method == 'POST':
        date_str = request.form.get('date')
        selected_games = request.form.getlist('selected_games')
        if not date_str or not selected_games:
            return "Дата и хотя бы одна игра обязательны!", 400

        # Добавляем каждую выбранную игру как задание
        for game_name in selected_games:
            game_id = add_game(game_name) # Добавляет игру в базу, если ее нет
            add_assignment(student_id, game_id, date_str)

        return redirect(url_for('student_page', student_id=student_id))

    # --- ИСПРАВЛЕНИЕ: Логика рекомендаций ---
    recommended_games = []
    if kmeans_model and scaler:
        try:
            student_data = get_student_data_for_model(student_id)
            if not student_data:
                print(f"Ошибка: данные ученика ID {student_id} не найдены для рекомендаций.")
                # Возвращаем все игры, если данных нет
                recommended_games = all_games_from_db
            else:
                # --- ВЫВОД ИНФОРМАЦИИ О КЛАСТЕРИЗАЦИИ ---
                print(f"--- Информация о кластеризации для ученика {student[1]} (ID {student_id}) ---")
                print(f"  Диагноз: {student_data['diagnosis']}")
                print(f"  Возраст: {student_data['age']}")
                print(f"  Играл в: {student_data['played_games']}")

                # Подготовка признаков для ученика
                X_single = prepare_patient_features_for_prediction(student_data)
                print(f"  Вектор признаков (X_single): {X_single[0]}") # Выводим вектор признаков

                X_scaled = scaler.transform(X_single)
                print(f"  Вектор признаков (X_scaled): {X_scaled[0]}") # Выводим масштабированный вектор

                cluster_label = kmeans_model.predict(X_scaled)[0]
                print(f"  Ученик принадлежит кластеру: {cluster_label}")

                centroid = kmeans_model.cluster_centers_[cluster_label]
                print(f"  Центроид кластера {cluster_label}: {centroid}")

                # --- НОВАЯ ЛОГИКА РЕКОМЕНДАЦИЙ НА ОСНОВЕ ДИАГНОЗА ---
                # Получаем диагноз ученика
                diagnosis = student_data['diagnosis']
                if diagnosis:
                    # Разбиваем строку диагнозов на список
                    diag_list = [d.strip() for d in diagnosis.split(',') if d.strip()]
                    # Берем первый диагноз для простоты (можно расширить для нескольких)
                    primary_diagnosis = diag_list[0] if diag_list else None

                    if primary_diagnosis:
                        print(f"  Используем диагноз '{primary_diagnosis}' для фильтрации рекомендаций.")
                        # Получаем всех учеников с этим диагнозом из базы данных
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM students WHERE diagnosis LIKE ?", ('%' + primary_diagnosis + '%',))
                        students_with_diag = cursor.fetchall()
                        conn.close()

                        # Собираем все игры, которые назначались ученикам с этим диагнозом
                        games_for_diag = set()
                        for (stud_id,) in students_with_diag:
                            games = get_student_games(stud_id)
                            for game_name, _ in games:
                                games_for_diag.add(game_name)

                        # Исключаем уже назначенные игры для текущего ученика
                        assigned_games = [g[0] for g in get_student_games(student_id)]
                        recommended_games = [g for g in games_for_diag if g not in assigned_games]

                        # Сортируем по частоте использования (если нужно)
                        # Для простоты, просто берем топ-5
                        recommended_games = list(recommended_games)[:5]

                        print(f"  Рекомендации (на основе диагноза): {recommended_games}")

                    else:
                        print(f"  Диагноз не найден или пуст. Используем логику на основе кластера.")
                        # Используем логику на основе кластера
                        difference_vector = centroid - X_scaled[0]
                        print(f"  Вектор разности (centroid - X_student): {difference_vector}")
                        num_games = len(all_games_from_db)
                        game_feature_indices = list(range(num_games))
                        game_differences = difference_vector[game_feature_indices]
                        print(f"  Разности для игр: {game_differences}")
                        top_game_indices = np.argsort(game_differences)[::-1]
                        print(f"  Индексы игр по убыванию разности: {top_game_indices}")
                        potential_recommended_games = [all_games_from_db[i] for i in top_game_indices if i < len(all_games_from_db)]
                        assigned_games = [g[0] for g in get_student_games(student_id)]
                        recommended_games = [g for g in potential_recommended_games if g not in assigned_games]
                        recommended_games = recommended_games[:5]
                        print(f"  Рекомендации (на основе кластера): {recommended_games}")

                else:
                    print(f"  Диагноз отсутствует. Используем логику на основе кластера.")
                    # Используем логику на основе кластера
                    difference_vector = centroid - X_scaled[0]
                    print(f"  Вектор разности (centroid - X_student): {difference_vector}")
                    num_games = len(all_games_from_db)
                    game_feature_indices = list(range(num_games))
                    game_differences = difference_vector[game_feature_indices]
                    print(f"  Разности для игр: {game_differences}")
                    top_game_indices = np.argsort(game_differences)[::-1]
                    print(f"  Индексы игр по убыванию разности: {top_game_indices}")
                    potential_recommended_games = [all_games_from_db[i] for i in top_game_indices if i < len(all_games_from_db)]
                    assigned_games = [g[0] for g in get_student_games(student_id)]
                    recommended_games = [g for g in potential_recommended_games if g not in assigned_games]
                    recommended_games = recommended_games[:5]
                    print(f"  Рекомендации (на основе кластера): {recommended_games}")

                print(f"--- Конец информации о кластеризации ---")

        except Exception as e:
            print(f"Ошибка при получении рекомендаций: {e}")
            # Если ошибка, вернем все игры
            recommended_games = all_games_from_db

    # Если модель не загружена, также возвращаем все игры
    if not kmeans_model or not scaler:
        recommended_games = all_games_from_db

    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    html_template = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Добавить задание для {{ student[1] }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input, select { width: 100%; padding: 8px; margin-top: 2px; box-sizing: border-box; }
        button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .recommended-games { margin-top: 20px; }
        .game-checkbox { margin: 5px 0; }
        .back-link { display: inline-block; margin-top: 20px; text-decoration: none; color: #0066cc; }
    </style>
</head>
<body>
    <h1>{{ student[1] }} - добавить задание</h1>

    <form method="POST">
        <div class="form-group">
            <label for="date">Выберите дату:</label>
            <input type="date" id="date" name="date" required value="{{ date.today().strftime('%Y-%m-%d') }}">
        </div>

        <div class="form-group">
            <label>Рекомендованные игры (на основе модели, истории ученика и диагноза):</label>
            <div class="recommended-games">
                {% if recommended_games %}
                    {% for game in recommended_games %}
                        <div class="game-checkbox">
                            <input type="checkbox" id="game_{{ loop.index }}" name="selected_games" value="{{ game }}">
                            <label for="game_{{ loop.index }}">{{ game }}</label>
                        </div>
                    {% endfor %}
                {% else %}
                    <p>Нет специфических рекомендаций. Выберите игры вручную из списка ниже.</p>
                {% endif %}
            </div>
        </div>

        <div class="form-group">
            <label>Все доступные игры:</label>
            <select name="selected_games" multiple size="10">
                {% for game in all_games %}
                    <option value="{{ game }}">{{ game }}</option>
                {% endfor %}
            </select>
        </div>

        <button type="submit">Назначить</button>
    </form>

    <a href="/student/{{ student[0] }}" class="back-link">Назад к ученику</a>
</body>
</html>
    '''
    # Передаем date.today() в шаблон для значения по умолчанию
    from datetime import date
    return render_template_string(html_template, student=student, recommended_games=recommended_games, all_games=all_games_from_db, date=date)

# ... (остальной код app.py остается без изменений) ...

if __name__ == '__main__':
    load_model_and_data()
    # Если модель не найдена, раскомментируйте следующую строку для первого запуска
    # train_and_save_model()
    app.run(debug=False)
