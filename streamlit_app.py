import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Настраиваем саму страницу
st.set_page_config(page_title="Страховой скоринг")

# Функцию загрузки кэшируем, чтобы приложение не тормозило
@st.cache_resource
def load_model_and_preprocessor():
    # Загружаем файлы, которые мы скачали из Colab
    model = joblib.load('insurance_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

# Пробуем загрузить модель, если файлов нет — выведем понятную ошибку
try:
    model, preprocessor = load_model_and_preprocessor()
except Exception as e:
    st.error(f"Не удалось загрузить файлы модели. Ошибка: {e}")
    st.stop()

st.title(" Прогноз спроса на турстраховку")
st.write("Инструмент для оценки вероятности покупки страхового полиса клиентом.")

# --- Блок ввода данных ---
st.subheader("Анкета клиента")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Возраст клиента", 18, 100, 28)
    income = st.number_input("Годовой доход (INR)", 100000, 2000000, 800000, step=50000)
    family = st.slider("Человек в семье", 1, 10, 4)

with c2:
    emp = st.selectbox("Сфера занятости", 
                       ["Private Sector/Self Employed", "Government Sector"],
                       format_func=lambda x: "Частный сектор / ИП" if "Private" in x else "Госслужба")
    
    grad = st.radio("Высшее образование", ["Yes", "No"], 
                    format_func=lambda x: "Да" if x == "Yes" else "Нет")
    
    chronic = st.checkbox("Есть хронические заболевания")

# Эти два параметра — самые важные по результатам моего анализа
st.markdown("---")
flyer = st.selectbox("Клиент часто летает самолетами?", ["Yes", "No"], 
                     format_func=lambda x: "Да" if x == "Yes" else "Нет")
abroad = st.selectbox("Клиент бывал за границей ранее?", ["Yes", "No"],
                      format_func=lambda x: "Да" if x == "Yes" else "Нет")

# --- Кнопка расчета ---
if st.button("Проанализировать"):
    # Собираем данные в DataFrame (важно соблюдать названия колонок из обучения!)
    input_data = pd.DataFrame([{
        "Age": age,
        "Employment Type": emp,
        "GraduateOrNot": grad,
        "AnnualIncome": income,
        "FamilyMembers": family,
        "ChronicDiseases": 1 if chronic else 0,
        "FrequentFlyer": flyer,
        "EverTravelledAbroad": abroad
    }])

    # Прогоняем через пайплайн и делаем прогноз
    processed = preprocessor.transform(input_data)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]

    # Вывод результата
    st.markdown("### Результат анализа")
    res_col, prob_col = st.columns(2)
    
    with res_col:
        if prediction == 1:
            st.success("РЕКОМЕНДАЦИЯ: Предложить расширенный пакет")
        else:
            st.warning("РЕКОМЕНДАЦИЯ: Стандартное предложение")

    with prob_col:
        st.metric("Вероятность покупки", f"{probability*100:.1f}%")
        st.progress(int(probability * 100))

# Маленькое пояснение 
with st.expander("Заметки разработчика"):
    st.write("""
        * Модель: Gradient Boosting Classifier.
        * Обучена на датасете Travel Insurance (2000 записей).
        * Основные веса: Доход и наличие поездок за границу.
    """)