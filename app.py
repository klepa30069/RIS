# app.py
import sqlite3
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
from datetime import date

app = Flask(__name__)

# --- Импортируем функции работы с базой данных ---
from database import init_db, load_data_from_csv_to_db, get_all_students, add_student, get_student, get_student_games, \
    add_game, get_all_games, add_assignment, get_student_data_for_model, update_student, delete_student

# Путь к модели и скалеру
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'

# Глобальные переменные для модели
kmeans_model = None
scaler = None
all_games_from_db = []
all_diagnoses_from_db = []


def load_model_and_data():
    global kmeans_model, scaler, all_games_from_db, all_diagnoses_from_db

    # Инициализируем базу данных
    init_db()

    # Загружаем данные из CSV в базу данных (если база пуста)
    load_data_from_csv_to_db()

    # Загружаем все игры и диагнозы из базы данных
    all_games_from_db = get_all_games()

    # Получаем диагнозы через функцию базы данных
    from database import get_connection
    conn = get_connection()

    try:
        if hasattr(conn, 'cursor'):
            # PostgreSQL через psycopg
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT diagnosis FROM students WHERE diagnosis IS NOT NULL AND diagnosis != ''")
            diagnoses = cursor.fetchall()
            cursor.close()
        else:
            # SQLite
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT diagnosis FROM students WHERE diagnosis IS NOT NULL AND diagnosis != ''")
            diagnoses = cursor.fetchall()
            cursor.close()
    finally:
        if hasattr(conn, 'close'):
            conn.close()

    # Обрабатываем диагнозы
    all_diagnoses_from_db = []
    for diag in diagnoses:
        if isinstance(diag, dict):
            diag_str = diag['diagnosis']
        else:
            diag_str = diag[0] if diag else ''

        if diag_str:
            diag_list = [d.strip() for d in diag_str.split(',')]
            all_diagnoses_from_db.extend(diag_list)

    all_diagnoses_from_db = list(set(all_diagnoses_from_db))
    all_diagnoses_from_db.sort()

    # Загружаем модель и скалер
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            kmeans_model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("Модель и данные успешно загружены.")
    else:
        print("Модель или скалер не найдены. Запустите обучение модели.")


def prepare_patient_features_for_prediction(student_data):
    """
    Подготовка признаков для одного пациента (ученика) для предсказания.
    student_data: словарь с ключами 'age', 'diagnosis', 'played_games'
    """
    feature_vector = []

    for game in all_games_from_db:
        feature_vector.append(1 if game in student_data['played_games'] else 0)

    current_diagnoses = []
    if student_data['diagnosis']:
        current_diagnoses = [d.strip() for d in student_data['diagnosis'].split(',') if d.strip()]
    for diagnosis in all_diagnoses_from_db:
        feature_vector.append(1 if diagnosis in current_diagnoses else 0)

    feature_vector.append(float(student_data['age']))

    return np.array([feature_vector])


def train_and_save_model():
    """
    Функция для обучения модели на основе данных из базы данных.
    """
    students = get_all_students()
    if len(students) == 0:
        print("Нет данных для обучения модели.")
        return

    data_list = []
    for student in students:
        student_id, name, age, diagnosis = student
        played_games = [g[0] for g in get_student_games(student_id)]
        data_list.append({
            'ФИО': name,
            'Возраст': age,
            'диагноз': diagnosis,
            'игры': ', '.join(played_games) if played_games else ''
        })

    data = pd.DataFrame(data_list)

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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

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

    global all_diagnoses_from_db
    all_diagnoses = all_diagnoses_from_db
    # --- ПЕРЕДАЕМ КОЛИЧЕСТВО ДИАГНОЗОВ ---
    num_diagnoses = len(all_diagnoses)
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    html_template = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Ученики</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px; }
        .diagnosis-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
        .tag { background-color: #e0e0e0; padding: 5px 10px; border-radius: 15px; cursor: pointer; font-size: 0.9em; border: 1px solid #ccc; }
        .tag:hover { background-color: #d0d0d0; }
        .tag.selected { background-color: #4CAF50; color: white; border-color: #45a049; }
        .student-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .student-card { background-color: #f0f0f0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .student-name { font-weight: bold; font-size: 1.2em; margin-bottom: 5px; }
        .student-age { margin-bottom: 5px; }
        .student-diagnosis { font-style: italic; margin-bottom: 10px; }
        .add-button { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; text-decoration: none; }
        .add-button:hover { background-color: #45a049; }
        .view-button, .edit-button { display: inline-block; margin-top: 10px; text-decoration: none; color: #0066cc; margin-right: 10px; }
        .edit-button { color: #f44336; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Список учеников</h1>
        <a href="/add_student" class="add-button">Добавить ученика</a>
    </div>

    <div class="diagnosis-tags">
        <span>Фильтр по диагнозу: </span>
        {% for diag in all_diagnoses %}
            <span class="tag" data-diagnosis="{{ diag }}">{{ diag }}</span>
        {% endfor %}
        <span id="clear-filters" class="tag" style="background-color: #f44336; color: white; display: none;">Сбросить фильтр</span>
    </div>

    <div class="student-grid">
        {% for student in students %}
            <div class="student-card" data-diagnoses="{{ student[3] or '' }}">
                <div class="student-name">{{ student[1] }}</div>
                <div class="student-diagnosis">Диагнозы: {{ student[3] or 'N/A' }}</div>
                <div class="student-age">{{ student[2] }} лет</div>
                <a href="/student/{{ student[0] }}" class="view-button">Перейти к ученику</a>
                <a href="/edit_student/{{ student[0] }}" class="edit-button">Редактировать</a>
            </div>
        {% endfor %}
    </div>

    <script>
        const selectedDiagnoses = new Set();
        const clearButton = document.getElementById('clear-filters');

        function updateCardVisibility() {
            const cards = document.querySelectorAll('.student-card');
            const tags = document.querySelectorAll('.tag[data-diagnosis]');

            cards.forEach(card => {
                const cardDiagnoses = card.getAttribute('data-diagnoses').toLowerCase();
                let showCard = selectedDiagnoses.size === 0;

                if (selectedDiagnoses.size > 0) {
                    showCard = false;
                    for (const diag of selectedDiagnoses) {
                        if (cardDiagnoses.includes(diag.toLowerCase())) {
                            showCard = true;
                            break;
                        }
                    }
                }

                if (showCard) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });

            tags.forEach(tag => {
                const diag = tag.getAttribute('data-diagnosis');
                if (selectedDiagnoses.has(diag)) {
                    tag.classList.add('selected');
                } else {
                    tag.classList.remove('selected');
                }
            });

            clearButton.style.display = selectedDiagnoses.size > 0 ? 'inline-block' : 'none';
        }

        document.querySelectorAll('.tag[data-diagnosis]').forEach(tag => {
            tag.addEventListener('click', function() {
                const diag = this.getAttribute('data-diagnosis');
                if (selectedDiagnoses.has(diag)) {
                    selectedDiagnoses.delete(diag);
                } else {
                    selectedDiagnoses.add(diag);
                }
                updateCardVisibility();
            });
        });

        clearButton.addEventListener('click', function() {
            selectedDiagnoses.clear();
            updateCardVisibility();
        });

        updateCardVisibility();

        // --- ВЫВОД КОЛИЧЕСТВА ТЕГОВ В КОНСОЛЬ ---
        console.log("Количество найденных тегов (диагнозов): {{ num_diagnoses }}");
        // --- КОНЕЦ ВЫВОДА ---
    </script>
</body>
</html>
    '''
    return render_template_string(html_template, students=students, all_diagnoses=all_diagnoses,
                                  num_diagnoses=num_diagnoses)


@app.route('/add_student', methods=['GET', 'POST'])
def add_student_page():
    """Страница для добавления нового ученика"""
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT diagnosis FROM students WHERE diagnosis IS NOT NULL AND diagnosis != ''")
    diagnoses = cursor.fetchall()
    all_diagnoses = set()
    for diag_tuple in diagnoses:
        diag_str = diag_tuple[0]
        if diag_str:
            diag_list = [d.strip() for d in diag_str.split(',')]
            all_diagnoses.update(diag_list)
    all_diagnoses = sorted(list(all_diagnoses))
    conn.close()

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        diagnosis_list = request.form.getlist('diagnosis')
        if not name or not age:
            return "Имя и возраст обязательны!", 400
        try:
            age = float(age)
            if age < 2.5 or age > 12:
                return "Возраст должен быть от 2.5 до 12 лет!", 400
        except ValueError:
            return "Неверный формат возраста!", 400

        diagnosis = ', '.join(diagnosis_list) if diagnosis_list else ''
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
        input, select { width: 100%; padding: 8px; margin-bottom: 15px; }
        button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .back-link { display: inline-block; margin-top: 10px; text-decoration: none; color: #0066cc; }
    </style>
</head>
<body>
    <h1>Добавить нового ученика</h1>
    <form method="POST">
        <label for="name">Имя (Фамилия Имя):</label>
        <input type="text" id="name" name="name" required>

        <label for="age">Возраст (2.5 - 12):</label>
        <input type="number" id="age" name="age" step="0.5" min="2.5" max="12" required>

        <label for="diagnosis">Диагнозы (можно выбрать несколько):</label>
        <select name="diagnosis" id="diagnosis" multiple size="6">
            {% for diag in all_diagnoses %}
                <option value="{{ diag }}">{{ diag }}</option>
            {% endfor %}
        </select>

        <button type="submit">Добавить ученика</button>
    </form>
    <a href="/" class="back-link">Назад к списку учеников</a>
</body>
</html>
    '''
    return render_template_string(html_template, all_diagnoses=all_diagnoses)


# --- НОВЫЙ МАРШРУТ ДЛЯ УДАЛЕНИЯ ---
@app.route('/delete_student/<int:student_id>', methods=['POST'])
def delete_student_page(student_id):
    """Удалить ученика по ID"""
    student = get_student(student_id)
    if not student:
        return "Ученик не найден", 404

    delete_student(student_id)
    return redirect(url_for('index'))


# --- КОНЕЦ НОВОГО МАРШРУТА ---

@app.route('/student/<int:student_id>')
def student_page(student_id):
    """Страница ученика с его играми и кнопками редактировать/удалить"""
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
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px; }
        .student-info { }
        .student-name { font-weight: bold; font-size: 1.5em; margin-bottom: 5px; }
        .student-diagnosis { font-style: italic; color: #555; margin-bottom: 5px; }
        .student-age { margin-bottom: 10px; }
        .games-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }
        .game-card { background-color: #f0f0f0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .game-name { font-weight: bold; font-size: 1.2em; margin-bottom: 5px; }
        .game-stats { margin-bottom: 5px; }
        .add-assignment-button, .edit-student-button, .delete-student-button { background-color: #4CAF50; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; text-decoration: none; margin-right: 10px; }
        .add-assignment-button:hover, .edit-student-button:hover { background-color: #45a049; }
        .delete-student-button { background-color: #f44336; }
        .delete-student-button:hover { background-color: #da190b; }
        .back-link { display: inline-block; margin-top: 20px; text-decoration: none; color: #0066cc; }
    </style>
</head>
<body>
    <div class="header">
        <div class="student-info">
            <div class="student-name">{{ student[1] }}</div>
            <div class="student-diagnosis">Диагнозы: {{ student[3] or 'N/A' }}</div>
            <div class="student-age">{{ student[2] }} лет</div>
        </div>
        <div>
            <a href="/student/{{ student[0] }}/add_assignment" class="add-assignment-button">Добавить задание</a>
            <a href="/edit_student/{{ student[0] }}" class="edit-student-button">Редактировать</a>
            <!-- Форма для удаления -->
            <form action="/delete_student/{{ student[0] }}" method="POST" style="display: inline;">
                <button type="submit" class="delete-student-button" onclick="return confirm('Вы уверены, что хотите удалить ученика {{ student[1] }}?')">Удалить</button>
            </form>
        </div>
    </div>

    <h2>Назначенные игры</h2>
    {% if games %}
        <div class="games-grid">
            {% for game in games %}
                <div class="game-card">
                    <div class="game-name">{{ game[0] }}</div>
                    <div class="game-stats">Играл: 10 раз</div>
                    <div class="game-stats">Время: 20 минут</div>
                    <div class="game-stats">Дата: {{ game[1] }}</div>
                </div>
            {% endfor %}
        </div>
    {% else %}
        <p>У этого ученика пока нет назначенных игр.</p>
    {% endif %}

    <a href="/" class="back-link">Назад к списку учеников</a>
</body>
</html>
    '''
    return render_template_string(html_template, student=student, games=games)


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

        # --- ИСПРАВЛЕНИЕ: Проверка даты ---
        from datetime import datetime
        try:
            selected_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            today = datetime.now().date()
            if selected_date < today:
                return "Нельзя назначать задание на прошедшую дату!", 400
        except ValueError:
            return "Неверный формат даты!", 400
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        # --- ИСПРАВЛЕНИЕ: Удаление дубликатов ---
        selected_games = list(set(selected_games))
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        # Добавляем каждую выбранную игру как задание
        for game_name in selected_games:
            game_id = add_game(game_name)  # Добавляет игру в базу, если ее нет
            add_assignment(student_id, game_id, date_str)

        return redirect(url_for('student_page', student_id=student_id))

    # ... (остальной код функции add_assignment_page без изменений) ...
    # (включая логику рекомендаций и вывод HTML) ...
    recommended_games = []
    cluster_label = None  # Переменная для хранения метки кластера
    if kmeans_model and scaler:
        try:
            student_data = get_student_data_for_model(student_id)
            if not student_data:
                print(f"Ошибка: данные ученика ID {student_id} не найдены для рекомендаций.")
                recommended_games = all_games_from_db
            else:
                print(f"--- Информация о кластеризации для ученика {student[1]} (ID {student_id}) ---")
                print(f"  Диагноз: {student_data['diagnosis']}")
                print(f"  Возраст: {student_data['age']}")
                print(f"  Играл в: {student_data['played_games']}")

                X_single = prepare_patient_features_for_prediction(student_data)
                print(f"  Вектор признаков (X_single): {X_single[0]}")

                X_scaled = scaler.transform(X_single)
                print(f"  Вектор признаков (X_scaled): {X_scaled[0]}")

                cluster_label = kmeans_model.predict(X_scaled)[0]
                print(f"  Ученик принадлежит кластеру: {cluster_label}")

                centroid = kmeans_model.cluster_centers_[cluster_label]
                print(f"  Центроид кластера {cluster_label}: {centroid}")

                diagnosis = student_data['diagnosis']
                if diagnosis:
                    diag_list = [d.strip() for d in diagnosis.split(',') if d.strip()]
                    primary_diagnosis = diag_list[0] if diag_list else None

                    if primary_diagnosis:
                        print(f"  Используем диагноз '{primary_diagnosis}' для фильтрации рекомендаций.")
                        conn = sqlite3.connect('students.db')
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM students WHERE diagnosis LIKE ?",
                                       ('%' + primary_diagnosis + '%',))
                        students_with_diag = cursor.fetchall()
                        conn.close()

                        games_for_diag = set()
                        for (stud_id,) in students_with_diag:
                            games = get_student_games(stud_id)
                            for game_name, _ in games:
                                games_for_diag.add(game_name)

                        assigned_games = [g[0] for g in get_student_games(student_id)]
                        recommended_games = [g for g in games_for_diag if g not in assigned_games]
                        recommended_games = list(recommended_games)[:5]
                        print(f"  Рекомендации (на основе диагноза): {recommended_games}")

                    else:
                        print(f"  Диагноз не найден или пуст. Используем логику на основе кластера.")
                        difference_vector = centroid - X_scaled[0]
                        print(f"  Вектор разности (centroid - X_student): {difference_vector}")
                        num_games = len(all_games_from_db)
                        game_feature_indices = list(range(num_games))
                        game_differences = difference_vector[game_feature_indices]
                        print(f"  Разности для игр: {game_differences}")
                        top_game_indices = np.argsort(game_differences)[::-1]
                        print(f"  Индексы игр по убыванию разности: {top_game_indices}")
                        potential_recommended_games = [all_games_from_db[i] for i in top_game_indices if
                                                       i < len(all_games_from_db)]
                        assigned_games = [g[0] for g in get_student_games(student_id)]
                        recommended_games = [g for g in potential_recommended_games if g not in assigned_games]
                        recommended_games = recommended_games[:5]
                        print(f"  Рекомендации (на основе кластера): {recommended_games}")

                else:
                    print(f"  Диагноз отсутствует. Используем логику на основе кластера.")
                    difference_vector = centroid - X_scaled[0]
                    print(f"  Вектор разности (centroid - X_student): {difference_vector}")
                    num_games = len(all_games_from_db)
                    game_feature_indices = list(range(num_games))
                    game_differences = difference_vector[game_feature_indices]
                    print(f"  Разности для игр: {game_differences}")
                    top_game_indices = np.argsort(game_differences)[::-1]
                    print(f"  Индексы игр по убыванию разности: {top_game_indices}")
                    potential_recommended_games = [all_games_from_db[i] for i in top_game_indices if
                                                   i < len(all_games_from_db)]
                    assigned_games = [g[0] for g in get_student_games(student_id)]
                    recommended_games = [g for g in potential_recommended_games if g not in assigned_games]
                    recommended_games = recommended_games[:5]
                    print(f"  Рекомендации (на основе кластера): {recommended_games}")

                print(f"--- Конец информации о кластеризации ---")

        except Exception as e:
            print(f"Ошибка при получении рекомендаций: {e}")
            recommended_games = all_games_from_db

    if not kmeans_model or not scaler:
        recommended_games = all_games_from_db

    # --- ПЕРЕДАЕМ КОЛИЧЕСТВО ВСЕХ ИГР ---
    num_all_games = len(all_games_from_db)
    # --- КОНЕЦ ИЗМЕНЕНИЯ ---

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
        /* Исправление выравнивания и расположения */
        LABEL.game-label-checkbox
        {
          margin-left: 2em;
          display: block;
          position: relative;
          margin-top: -1.4em;  /* make this margin match whatever your line-height is */
          line-height: 1.4em;  /* can be set here, or elsewehere */
        }
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
                            <label for="game_{{ loop.index }}" class="game-label-checkbox">{{ game }}</label>
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

    <!-- Скрипт для вывода информации в консоль браузера -->
    <script>
        // Выводим информацию о кластере в консоль браузера
        console.log("Ученик принадлежит кластеру: {{ cluster_label }}");
        // --- ВЫВОД КОЛИЧЕСТВА ИГР В КОНСОЛЬ ---
        console.log("Количество найденных игр: {{ num_all_games }}");
        // --- КОНЕЦ ВЫВОДА ---
    </script>
</body>
</html>
    '''
    from datetime import date
    return render_template_string(html_template, student=student, recommended_games=recommended_games,
                                  all_games=all_games_from_db, date=date, cluster_label=cluster_label,
                                  num_all_games=num_all_games)


@app.route('/edit_student/<int:student_id>', methods=['GET', 'POST'])
def edit_student_page(student_id):
    """Страница для редактирования ученика"""
    student = get_student(student_id)
    if not student:
        return "Ученик не найден", 404

    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT diagnosis FROM students WHERE diagnosis IS NOT NULL AND diagnosis != ''")
    diagnoses = cursor.fetchall()
    all_diagnoses = set()
    for diag_tuple in diagnoses:
        diag_str = diag_tuple[0]
        if diag_str:
            diag_list = [d.strip() for d in diag_str.split(',')]
            all_diagnoses.update(diag_list)
    all_diagnoses = sorted(list(all_diagnoses))
    conn.close()

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        diagnosis_list = request.form.getlist('diagnosis')
        if not name or not age:
            return "Имя и возраст обязательны!", 400
        try:
            age = float(age)
            if age < 2.5 or age > 12:
                return "Возраст должен быть от 2.5 до 12 лет!", 400
        except ValueError:
            return "Неверный формат возраста!", 400

        diagnosis = ', '.join(diagnosis_list) if diagnosis_list else ''
        update_student(student_id, name, age, diagnosis)
        return redirect(url_for('index'))

    current_diagnoses = []
    if student[3]:
        current_diagnoses = [d.strip() for d in student[3].split(',')]

    html_template = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Редактировать ученика</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        form { max-width: 400px; }
        label { display: block; margin-bottom: 5px; }
        input, select { width: 100%; padding: 8px; margin-bottom: 15px; }
        button { padding: 10px 15px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        .back-link { display: inline-block; margin-top: 10px; text-decoration: none; color: #0066cc; }
    </style>
</head>
<body>
    <h1>Редактировать ученика: {{ student[1] }}</h1>
    <form method="POST">
        <label for="name">Имя (Фамилия Имя):</label>
        <input type="text" id="name" name="name" value="{{ student[1] }}" required>

        <label for="age">Возраст (2.5 - 12):</label>
        <input type="number" id="age" name="age" value="{{ student[2] }}" step="0.5" min="2.5" max="12" required>

        <label for="diagnosis">Диагнозы (можно выбрать несколько):</label>
        <select name="diagnosis" id="diagnosis" multiple size="6">
            {% for diag in all_diagnoses %}
                <option value="{{ diag }}" {% if diag in current_diagnoses %}selected{% endif %}>{{ diag }}</option>
            {% endfor %}
        </select>

        <button type="submit">Сохранить изменения</button>
    </form>
    <a href="/" class="back-link">Назад к списку учеников</a>
</body>
</html>
    '''
    return render_template_string(html_template, student=student, all_diagnoses=all_diagnoses,
                                  current_diagnoses=current_diagnoses)


@app.route('/check_db')
def check_db():
    """Проверка подключения к базе данных"""
    from database import get_connection, get_all_students, get_all_games

    try:
        conn = get_connection()
        students_count = len(get_all_students())
        games_count = len(get_all_games())

        result = {
            "database_url_exists": bool(os.environ.get('DATABASE_URL')),
            "students_count": students_count,
            "games_count": games_count,
            "all_games": get_all_games()[:5]  # первые 5 игр
        }

        if hasattr(conn, 'close'):
            conn.close()

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_model_and_data()
    # train_and_save_model()
    app.run(debug=True)
