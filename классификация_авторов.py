import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

os.makedirs('результаты/изображения', exist_ok=True)

print("="*60)
print("ПРОЕКТ: КЛАССИФИКАЦИЯ АВТОРОВ - WriterExtnd")
print("="*60)

# ЭТАП 1: ЗАГРУЗКА ДАННЫХ АВТОРОВ

print("\n1. Загрузка данных авторов...")

authors_files = {
    'Bradbury': 'Bradbury.txt',
    'Genri': 'Genri.txt',
    'Simak': 'Simak.txt',
    'Bulgakov': 'Bulgakov.txt',
    'Strugatskie': 'Strugatskie.txt',
    'Fry': 'Fry.txt'
}

def clean_text(text):
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_chunks(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    if len(words) < 100:
        return [text]
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i+chunk_size]
        if len(chunk_words) >= 100:
            chunks.append(' '.join(chunk_words))
    return chunks

train_texts = []
train_labels = []
chunk_counts = {}

for author, filename in authors_files.items():
    filepath = os.path.join('данные', 'авторы', filename)
    if not os.path.exists(filepath):
        print(f"  Файл не найден: {filepath}")
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        full_text = f.read()
    full_text = clean_text(full_text)
    chunks = split_into_chunks(full_text, chunk_size=400, overlap=50)
    train_texts.extend(chunks)
    train_labels.extend([author] * len(chunks))
    chunk_counts[author] = len(chunks)
    print(f"  {author}: {len(chunks)} фрагментов")

print(f"  Всего: {len(train_texts)} фрагментов")

# ЭТАП 2: ЗАГРУЗКА ТАЙНЫХ ТЕКСТОВ

print("\n2. Загрузка тайных текстов...")

test_files = sorted(glob.glob('данные/секретные_тексты/author*.txt'))

if len(test_files) == 0:
    print("  ОШИБКА: Тайные тексты не найдены")
    exit()

test_texts = []
test_names = []

for filepath in test_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        text = clean_text(text)
        test_texts.append(text)
        test_names.append(os.path.basename(filepath))

print(f"  Загружено {len(test_texts)} тайных текстов")

# ЭТАП 3: ВЕКТОРИЗАЦИЯ TF-IDF

print("\n3. Векторизация TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)

X_train = vectorizer.fit_transform(train_texts)
y_train = train_labels
X_test = vectorizer.transform(test_texts)

print(f"  Размерность: {X_train.shape[1]} признаков")
print(f"  Матрица обучения: {X_train.shape[0]} x {X_train.shape[1]}")
print(f"  Матрица теста: {X_test.shape[0]} x {X_test.shape[1]}")

# ЭТАП 4: ОБУЧЕНИЕ МОДЕЛЕЙ

print("\n4. Обучение моделей...")

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
model_lr.fit(X_train_split, y_train_split)
acc_lr = accuracy_score(y_val, model_lr.predict(X_val))
print(f"  Logistic Regression: {acc_lr:.3f}")

model_svm = SVC(kernel='linear', C=1.0, random_state=42, class_weight='balanced')
model_svm.fit(X_train_split, y_train_split)
acc_svm = accuracy_score(y_val, model_svm.predict(X_val))
print(f"  SVM Линейный: {acc_svm:.3f}")

model_rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, class_weight='balanced')
model_rf.fit(X_train_split, y_train_split)
acc_rf = accuracy_score(y_val, model_rf.predict(X_val))
print(f"  Random Forest: {acc_rf:.3f}")

# ЭТАП 5: АНСАМБЛЬ МОДЕЛЕЙ

print("\n5. Создание ансамбля моделей...")

voting_clf = VotingClassifier(
    estimators=[('lr', model_lr), ('svm', model_svm), ('rf', model_rf)],
    voting='hard'
)

voting_clf.fit(X_train_split, y_train_split)
acc_vote = accuracy_score(y_val, voting_clf.predict(X_val))
print(f"  Voting Classifier: {acc_vote:.3f}")

cv_scores = cross_val_score(voting_clf, X_train, y_train, cv=5)
print(f"  Перекрёстная проверка: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ЭТАП 6: ПРЕДСКАЗАНИЕ ТАЙНЫХ ТЕКСТОВ

print("\n6. Предсказание тайных текстов...")

voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)

# ЭТАП 7: СОЗДАНИЕ ФАЙЛА предсказания.csv

print("\n7. Создание файла предсказания.csv...")

submission = pd.DataFrame({
    'filename': test_names,
    'author': predictions
})

def extract_number(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

submission['sort_key'] = submission['filename'].apply(extract_number)
submission = submission.sort_values('sort_key').drop('sort_key', axis=1)

submission.to_csv('результаты/предсказания.csv', index=False)

print("предсказания.csv создан")
print(submission.to_string(index=False))

# ЭТАП 8: СОЗДАНИЕ ГРАФИКОВ

print("\n8. Создание графиков...")

# График 1: Распределение фрагментов
plt.figure(figsize=(10, 6))
authors = list(chunk_counts.keys())
counts = list(chunk_counts.values())
bars = plt.bar(authors, counts, color='steelblue')
plt.title('Распределение фрагментов по авторам', fontsize=14)
plt.xlabel('Автор')
plt.ylabel('Количество фрагментов')
plt.xticks(rotation=45)
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(count), ha='center', va='bottom')
plt.tight_layout()
plt.savefig('результаты/изображения/graphique1_distribution.png', dpi=150)
plt.close()
print("  graphique1_distribution.png")

# График 2: Матрица ошибок
cm = confusion_matrix(y_val, voting_clf.predict(X_val))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=voting_clf.classes_, yticklabels=voting_clf.classes_)
plt.title('Матрица ошибок', fontsize=14)
plt.xlabel('Предсказание')
plt.ylabel('Правильный ответ')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('результаты/изображения/graphique2_confusion.png', dpi=150)
plt.close()
print("  graphique2_confusion.png")

# График 3: Сравнение моделей
plt.figure(figsize=(10, 6))
accuracies = [acc_lr, acc_svm, acc_rf, acc_vote]
model_names = ['Logistic Regression', 'SVM', 'Random Forest', 'Voting Classifier']
bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.title('Точность моделей', fontsize=14)
plt.xlabel('Модель')
plt.ylabel('Точность')
plt.ylim(0, 1)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('результаты/изображения/graphique3_performances.png', dpi=150)
plt.close()
print("  graphique3_performances.png")

# График 4: Перекрёстная проверка
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), cv_scores, 'o-', linewidth=2, markersize=8, color='green')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Среднее: {cv_scores.mean():.3f}')
plt.xlabel('Блок')
plt.ylabel('Точность')
plt.title('Перекрёстная проверка (5 блоков)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('результаты/изображения/graphique4_crossvalidation.png', dpi=150)
plt.close()
print("  graphique4_crossvalidation.png")

# ИТОГИ

print("\n" + "="*60)
print("ИТОГИ")
print("="*60)
print(f"  Авторов: {len(authors_files)}")
print(f"  Фрагментов: {len(train_texts)}")
print(f"  Тайных текстов: {len(test_texts)}")
print(f"  Voting Classifier: {acc_vote:.3f} ({acc_vote*100:.1f}%)")
print(f"  Перекрёстная проверка: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

print("\nРаспределение предсказаний:")
for author, count in submission['author'].value_counts().items():
    print(f"  {author}: {count}")

print("\n" + "="*60)
print("ГОТОВО")
print("="*60)
print("\nСоздано:")
print("  - результаты/предсказания.csv")
print("  - 4 графика в папке результаты/изображения/")
print("="*60)