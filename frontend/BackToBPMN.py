import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b-q4_K_M"

SYSTEM_PROMPT = """
## Goal  
Generate a HIGH QUALITY Mermaid diagram from Markdown content.

## Roles
- "AI Architect": You, the AI, will act as a Senior Software Architect that produces very high-quality Mermaid diagrams.

## Steps

> On first chat, please print in short bullet points those 6 steps we will follow.
1. Ask for the document to convert.
2. Once provided, analyze and write down the plan for the diagram, identify:
  - Components (main elements, logical groups) (in colors)
  - Children and parents elements
  - Directions and hierarchies
  - Relationships (in colors, connections and dependencies)
  - Notes and labels needed for each element if any
3. Ask user: "Do you confirm the plan?" and wait for user confirmation.
4. Generate the 100% valid Mermaid diagram from the plan.
5. Ask user: "Do you want me to review it?" and wait for user confirmation.
6. If the user confirms, review the diagram and suggest improvements :
  - Check syntax
  - DO NOT add any extra elements
  - Look for empty nodes or misplaced elements
  - Ensure styling is correct
  - Upgrade styles if necessary

## Rules  

- Chart type: "stateDiagram-v2".  
- Flow: "left-to-right"
- Use Mermaid v10.8.0 minimum.
- 100% valid Mermaid diagram is required.
- Read the rules under "Mermaid generation rules" section.

### Mermaid generation rules

**Header**:
- Use "---" to define the header's title.

**States and nodes**:
- Define groups, parents and children.
- Fork and join states.
- Use clear and concise names.
- Use "choice" when a condition is needed.
- No standalone nodes.
- No empty nodes.

**Naming**:
- Consistent naming
- Descriptive (no "A", "B"...)
- Surrounded by double quotes.
- Replace ":" with "$" in state names if any.

**Links**:
- Use direction when possible.
- "A -- text --> B" for regular links.
- "A -.-> B" for conditional links.
- "A ==> B" for self-loops.

**Styles**:
- Forms:
  - Circles for states
  - Rectangles for blocks
  - Diamonds for decisions
  - Hexagons for groups
- Max 4 colors in high contrast.


**Miscellaneous**:
- Avoid `linkStyle`.
"""


def generate_mermaid_code(process_description: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Process description:\n{process_description}"
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 2048
        }
    }

    try:
        print(f"Генерирую схему с помощью {MODEL_NAME}...")
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()

        result_text = response.json()['message']['content']

        clean_code = clean_output(result_text)
        return clean_code

    except requests.exceptions.ConnectionError:
        return "Ошибка: Ollama не запущена. Проверьте http://localhost:11434"
    except Exception as e:
        return f"Ошибка: {str(e)}"


def clean_output(text: str) -> str:
    """Убирает лишние символы маркдауна, оставляя только код"""
    pattern = r"```(?:mermaid)?\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# --- ПРИМЕР ЗАПУСКА ---
if __name__ == "__main__":
    input_text = """
    Клиент отправляет заявку на сайте.
    Менеджер получает уведомление и проверяет заявку.
    Если заявка заполнена неверно, менеджер отправляет отказ, и процесс завершается.
    Если заявка верна, менеджер создает договор.
    Система автоматически отправляет договор клиенту на почту.
    """

    mermaid_code = generate_mermaid_code(input_text)

    print("\n--- RESULT MERMAID CODE ---\n")
    print(mermaid_code)
    print("\n---------------------------")
