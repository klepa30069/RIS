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
from datetime import date  # Импортируем date для форматирования

app = Flask(__name__)

# --- Импортируем функции работы с базой данных ---
from database import init_db, load_data_from_csv_to_db, get_all_students, add_student, get_student, get_student_games, \
    add_game, get_all_games, add_assignment, get_student_data_for_model, update_student, \
    delete_student  # Импортируем новую функцию

# Путь к модели и скалеру
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
DB_PATH = 'students.db'

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
    conn = sqlite3.connect('students.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT diagnosis FROM students WHERE diagnosis IS NOT NULL AND diagnosis != ''")
    diagnoses = cursor.fetchall()
    all_diagnoses_from_db = []
    for diag_tuple in diagnoses:
        diag_str = diag_tuple[0]
        if diag_str:
            diag_list = [d.strip() for d in diag_str.split(',')]
            all_diagnoses_from_db.extend(diag_list)
    all_diagnoses_from_db = list(set(all_diagnoses_from_db))
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
    Теперь находит оптимальное количество кластеров и выводит информацию о них.
    """
    students = get_all_students()
    if len(students) < 3: # KMeans требует минимум 2 точки, но для кластеризации лучше больше
        print("Недостаточно данных для обучения модели (нужно минимум 3 ученика).")
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
                all_games.update([str(game).strip() for game in games_list if str(game).strip()])

        all_diagnoses = set()
        for diagnosis in data_processed['диагноз']:
            if pd.notna(diagnosis):
                diagnoses = str(diagnosis).split(', ')
                all_diagnoses.update([str(d).strip() for d in diagnoses if str(d).strip()])

        features = []
        patient_ids = []

        for _, row in data_processed.iterrows():
            feature_vector = []
            current_games = row['игры_список'] if isinstance(row['игры_список'], list) else []
            current_games = [str(g).strip() for g in current_games]
            for game in all_games:
                feature_vector.append(1 if str(game) in current_games else 0)

            current_diagnoses = []
            if pd.notna(row['диагноз']):
                 current_diagnoses = [str(d).strip() for d in str(row['диагноз']).split(', ') if str(d).strip()]
            for diagnosis in all_diagnoses:
                feature_vector.append(1 if str(diagnosis) in current_diagnoses else 0)

            feature_vector.append(float(row['Возраст_число']))
            features.append(feature_vector)
            patient_ids.append(str(row['ФИО']))

        feature_names = list(all_games) + list(all_diagnoses) + ['Возраст']
        return np.array(features), feature_names, patient_ids

    X, _, patient_ids = prepare_patient_features(data)

    # --- НОВАЯ ЛОГИКА: Поиск оптимального k ---
    def find_optimal_clusters_patients(X, max_k=10):
        """
        Находит оптимальное количество кластеров для пациентов
        """
        # Масштабируем данные для лучшей кластеризации
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Убедимся, что max_k не превышает количество точек данных
        max_k = min(max_k, len(X), len(X) - 1) # n_clusters должен быть < n_samples
        if max_k < 2:
             print("Недостаточно данных для кластеризации (max_k < 2).")
             return pd.DataFrame(), StandardScaler() # Возвращаем пустой DF и пустой scaler

        k_range = range(2, max_k + 1)
        results = []

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_scaled)

                # Вычисляем метрики
                if len(set(labels)) > 1:  # Проверяем, что есть хотя бы 2 кластера
                    silhouette = silhouette_score(X_scaled, labels)
                    db_index = davies_bouldin_score(X_scaled, labels)
                    ch_index = calinski_harabasz_score(X_scaled, labels)
                    inertia = kmeans.inertia_

                    # Вычисляем средний размер кластера
                    cluster_sizes = np.bincount(labels)
                    avg_cluster_size = np.mean(cluster_sizes)
                    cluster_size_std = np.std(cluster_sizes)

                else:
                    silhouette = db_index = ch_index = inertia = avg_cluster_size = cluster_size_std = np.nan

                results.append({
                    'k': k,
                    'silhouette': silhouette,
                    'davies_bouldin': db_index,
                    'calinski_harabasz': ch_index,
                    'inertia': inertia,
                    'avg_cluster_size': avg_cluster_size,
                    'cluster_size_std': cluster_size_std
                })
            except Exception as e:
                print(f"Ошибка при k={k}: {e}")
                results.append({
                    'k': k,
                    'silhouette': np.nan,
                    'davies_bouldin': np.nan,
                    'calinski_harabasz': np.nan,
                    'inertia': np.nan,
                    'avg_cluster_size': np.nan,
                    'cluster_size_std': np.nan
                })

        return pd.DataFrame(results), scaler

    def select_optimal_k_patients(k_results, min_cluster_size=3):
        """
        Автоматический выбор оптимального k для пациентов с учетом размера кластеров
        """
        # Убираем строки с NaN значениями
        df = k_results.dropna().copy()

        if len(df) == 0:
            print("Нет валидных результатов для выбора k, использую значение по умолчанию (3)")
            return 3, df

        # Нормализуем метрики
        # Проверим, что у нас есть вариативность в данных перед нормализацией
        if df['silhouette'].max() == df['silhouette'].min():
            df['silhouette_norm'] = 0
        else:
            df['silhouette_norm'] = (df['silhouette'] - df['silhouette'].min()) / (df['silhouette'].max() - df['silhouette'].min())

        if df['calinski_harabasz'].max() == df['calinski_harabasz'].min():
            df['calinski_norm'] = 0
        else:
            df['calinski_norm'] = (df['calinski_harabasz'] - df['calinski_harabasz'].min()) / (df['calinski_harabasz'].max() - df['calinski_harabasz'].min())

        if df['davies_bouldin'].max() == df['davies_bouldin'].min():
            df['davies_norm'] = 1
        else:
            df['davies_norm'] = 1 - (df['davies_bouldin'] - df['davies_bouldin'].min()) / (df['davies_bouldin'].max() - df['davies_bouldin'].min())

        if df['inertia'].max() == df['inertia'].min():
            df['inertia_norm'] = 1
        else:
            df['inertia_norm'] = 1 - (df['inertia'] - df['inertia'].min()) / (df['inertia'].max() - df['inertia'].min())

        # Штрафуем кластеры с малым средним размером
        df['size_penalty'] = (df['avg_cluster_size'] - min_cluster_size) / (df['avg_cluster_size'].max() - min_cluster_size)
        df['size_penalty'] = df['size_penalty'].clip(lower=0)  # Не штрафуем если размер больше минимального

        # Композитная оценка с учетом размера кластеров
        df['composite_score'] = (df['silhouette_norm'] + df['calinski_norm'] + df['davies_norm'] + df['inertia_norm'] + df['size_penalty']) / 5

        optimal_k = df.loc[df['composite_score'].idxmax(), 'k']

        return int(optimal_k), df

    # Выполняем поиск
    k_results, temp_scaler = find_optimal_clusters_patients(X)
    if k_results.empty:
         print("Невозможно определить оптимальное количество кластеров.")
         return
    print("Результаты поиска оптимального k для пациентов:")
    print(k_results[['k', 'silhouette', 'davies_bouldin', 'calinski_harabasz', 'avg_cluster_size']].round(4))

    optimal_k, k_scores = select_optimal_k_patients(k_results)
    print(f"Оптимальное количество кластеров для пациентов: {optimal_k}")

    # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

    # Теперь обучаем финальную модель с найденным optimal_k
    # Используем scaler, полученный из функции поиска оптимального k
    X_scaled = temp_scaler.transform(X)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    final_labels = kmeans.fit_predict(X_scaled) # Получаем метки кластеров

    # --- ВЫВОД ИНФОРМАЦИИ О КЛАСТЕРАХ ---
    print("\n--- ИНФОРМАЦИЯ О КЛАСТЕРАХ ---")
    for cluster_id in range(optimal_k):
        cluster_mask = (final_labels == cluster_id)
        cluster_students_info = data[cluster_mask] # Фильтруем DataFrame по метке кластера
        cluster_students_count = len(cluster_students_info)

        print(f"\nКЛАСТЕР {cluster_id}:")
        print(f"  Количество детей: {cluster_students_count}")

        if cluster_students_count > 0:
            diagnoses_in_cluster = cluster_students_info['диагноз'].dropna().tolist()
            ages_in_cluster = cluster_students_info['Возраст'].tolist()
            games_played_in_cluster = cluster_students_info['игры'].dropna().tolist()

            # Подсчет уникальных заболеваний
            all_diagnoses = []
            for diag_str in diagnoses_in_cluster:
                if diag_str:
                    all_diagnoses.extend([d.strip() for d in diag_str.split(',')])
            unique_diagnoses = set(all_diagnoses)
            print(f"  Уникальные диагнозы: {', '.join(sorted(unique_diagnoses)) if unique_diagnoses else 'N/A'}")

            # Средний возраст
            avg_age = sum(ages_in_cluster) / len(ages_in_cluster) if ages_in_cluster else 0
            print(f"  Средний возраст: {avg_age:.2f}")

            # Подсчет уникальных игр
            all_games = []
            for games_str in games_played_in_cluster:
                if games_str:
                    all_games.extend([g.strip() for g in games_str.split(',')])
            unique_games = set(all_games)
            print(f"  Уникальные игры: {', '.join(sorted(unique_games)) if unique_games else 'N/A'}")

            # Вывод списка детей в кластере
            print(f"  Дети в кластере:")
            for _, row in cluster_students_info.iterrows():
                print(f"    - {row['ФИО']}: {row['Возраст']} лет, диагноз: {row['диагноз'] or 'N/A'}, играл: {row['игры'] or 'N/A'}")

        else:
            print("  (пустой кластер)")

    print("\n--- КОНЕЦ ИНФОРМАЦИИ О КЛАСТЕРАХ ---")
    # --- КОНЕЦ ВЫВОДА ---

    # Сохраняем модель и скалер
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(temp_scaler, f) # Сохраняем scaler из поиска k

    print(f"Модель обучена и сохранена. K = {optimal_k}")
    # Обновляем глобальные переменные модели после обучения
    global kmeans_model, scaler
    kmeans_model = kmeans
    scaler = temp_scaler



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
    conn = sqlite3.connect(DB_PATH)
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
                        conn = sqlite3.connect(DB_PATH)
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

    conn = sqlite3.connect(DB_PATH)
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


if __name__ == '__main__':
    load_model_and_data()
    train_and_save_model()
    app.run(debug=True)
