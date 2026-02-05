import streamlit as st
import requests
import os
import streamlit.components.v1 as components

API_URL = os.getenv("API_URL", "http://api:8000")
st.set_page_config(layout="wide", page_title="BPMN AI Assistant", page_icon="🤖")

st.markdown("""
<style>
    table.custom-table { width: 100% !important; border-collapse: collapse !important; color: #ffffff !important; background-color: #262730 !important; }
    table.custom-table th { background-color: #4F4F4F !important; color: white !important; padding: 12px !important; text-align: left !important; border: 1px solid #5e5e5e !important; }
    table.custom-table td { padding: 10px !important; border: 1px solid #5e5e5e !important; vertical-align: top !important; }
    table.custom-table tr:nth-child(even) { background-color: #363636 !important; }
    [data-testid="stSidebar"] { background-color: #1E1E1E; }
    .stButton>button { width: 100%; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

def render_mermaid_local(mermaid_code):
    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script> 
        </head>
        <body style="background-color: #0e1117; color: white;">
            <div class="mermaid">
                {mermaid_code}
            </div>
            <script>
                mermaid.initialize({{ startOnLoad: true, theme: 'dark', securityLevel: 'loose' }});
            </script>
        </body>
        </html>
    """
    components.html(html_content, height=600, scrolling=True)


def page_image_to_text():
    st.header("Распознавание Блок-схемы")
    st.caption("Загрузите картинку BPMN диаграммы, чтобы получить текстовое описание шагов.")

    left, mid, right = st.columns([5, 0.2, 5], gap="medium")

    with left:
        st.subheader("1. Загрузка файла")
        img_file = st.file_uploader("Выберите изображение в png или jpg формате", type=["png", "jpg", "jpeg"])
        process_btn = st.button("Отправить", type="primary", key="btn_img_text")

        if img_file:
            st.image(img_file, caption=f"Файл: {img_file.name}", use_container_width=True)

    with right:
        st.subheader("2. Результат анализа")
        if img_file and process_btn:
            with st.spinner("Анализ изображения..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (img_file.name, img_file.getvalue(), img_file.type)},
                        timeout=120
                    )

                    response.raise_for_status()
                    result = response.json()

                    results_list = result.get('results', [])

                    if results_list:
                        roles_exist = any(
                            item.get('role') and str(item.get('role')).strip()
                            for item in results_list
                        )

                        rows_html = ""
                        for i, item in enumerate(results_list, 1):
                            raw_text = item.get('text')
                            safe_text = str(raw_text) if raw_text else ""
                            text_display = safe_text.replace('\n', '<br>')

                            row_cells = f"<td>{i}</td><td>{text_display}</td>"

                            if roles_exist:
                                raw_role = item.get('role')
                                safe_role = str(raw_role) if raw_role else ""
                                role_display = safe_role.replace('\n', ' ')
                                row_cells += f"<td>{role_display}</td>"

                            rows_html += f"<tr>{row_cells}</tr>"

                        if roles_exist:
                            headers = '<th style="width: 10%;">№</th><th style="width: 50%;">Действие</th><th style="width: 40%;">Роль</th>'
                        else:
                            headers = '<th style="width: 10%;">№</th><th style="width: 90%;">Действие</th>'

                        exec_time = result.get('execution_time', 0)
                        st.success(f"Обработка завершена за {exec_time:.2f} сек")

                        table_html = f"""
                        <table class="custom-table">
                            <thead><tr>{headers}</tr></thead>
                            <tbody>{rows_html}</tbody>
                        </table>
                        """
                        st.markdown(table_html, unsafe_allow_html=True)



                    else:
                        st.warning("Текст или структура не найдены на схеме.")

                except requests.exceptions.ConnectionError:
                    st.error("Ошибка: Не удается подключиться к API. Убедитесь, что сервис запущен.")
                except requests.exceptions.Timeout:
                    st.error("Timeout: API слишком долго обрабатывает запрос.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"HTTP ошибка: {e.response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")


def page_text_to_diagram():
    st.header("Генерация Диаграммы из Текста")
    st.caption("Опишите бизнес-процесс, и AI построит диаграмму.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Описание процесса")
        example = st.selectbox("Загрузить пример:",
                               ["-- Пусто --", "Согласование отпуска", "Обработка заказа"])

        default_text = ""
        if example == "Согласование отпуска":
            default_text = "Сотрудник создает заявку на отпуск. Руководитель получает уведомление. Если руководитель одобряет, заявка идет в HR. Если отклоняет, сотрудник получает отказ."
        elif example == "Обработка заказа":
            default_text = "Клиент делает заказ. Менеджер проверяет наличие. Если товар есть, склад отгружает. Иначе отмена заказа."

        text_input = st.text_area("Введите текст здесь:", value=default_text, height=200)
        generate_btn = st.button("✨ Сгенерировать схему", type="primary", key="btn_text_diag")

    with col2:
        st.subheader("Предпросмотр")
        if generate_btn and text_input:
            with st.spinner("AI генерирует Mermaid код..."):
                try:
                    import time
                    time.sleep(0.5)
                    mermaid_code = """
                    flowchart TD
                        Start((Начало)) --> A[Создать заявку]
                        A --> B{Одобрено?}
                        B -- Да --> C[Оформить приказ]
                        B -- Нет --> D[Отказ]
                        C --> End((Конец))
                        D --> End
                    """

                    with st.expander("Исходный код Mermaid"):
                        st.code(mermaid_code, language='mermaid')

                    render_mermaid_local(mermaid_code)

                except Exception as e:
                    st.error(f"Ошибка генерации: {e}")


def main():
    with st.sidebar:
        st.title("Меню")
        page = st.radio(
            "Режим работы:",
            ["Распознавание (Img → Text)", "Генерация (Text → Diagram)"]
        )
        st.divider()
        st.info("Команда PoletiSchool")

    if page == "Распознавание (Img → Text)":
        page_image_to_text()
    else:
        page_text_to_diagram()


if __name__ == "__main__":
    main()
