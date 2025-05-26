# Shakespeare Chatbot Project

# CSCI933

---

## Table of Contents

- [About This Project](#about-this-project)
- [What It Does](#what-it-does)
- [How to Run It](#how-to-run-it)
  - [What You Need](#what-you-need)
  - [Steps to Get Started](#steps-to-get-started)
- [How to Use It](#how-to-use-it)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Ideas](#future-ideas)

---

## About This Project

This project is a lightweight, domain-specific intelligent chatbot capable of engaging in dialogue about Shakespearean literature, including answering factual and interpretive questions about Shakespeare’s plays, generating relevant quotes, and providing summaries of specific scenes. I created it as part of my Assignment to learn about basic natural language processing and how to build interactive programs'.

## What It Does

Here are the main things my chatbot can do:

- Engage in basic multi-turn dialogue about Shakespearean plays.
- Answer factual and thematic questions about characters, events, and scenes.
- Generate short quotes or passages from Shakespeare’s corpus.
- Summarize a selected scene from a specified play in prose.

## How to Run It

This part tells you how to get the chatbot working on your computer.

### What You Need

Make sure you have these installed:

- **Python 3.12**
- **pip**
- **Git**

### Steps to Get Started

Follow these steps:

1.  **Download the project:**

    ```bash
    git clone [https://github.com/id-nynt/2500010_shakespeare_bot.git]
    cd 2500010_shakespeare_bot

    ```

2.  **Install necessary libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the chatbot!**
    ```bash
    python utils/2_chatbot_interface.py
    ```

## How to Use It

- When you run the script, you'll see local web server using Gradio.
- Open your local web server
- Type your question or message and press **Submit**.

## Technologies Used

Here are the main tools and libraries I used:

- **Programming Language:** Python
- **Key Libraries:**
  - HuggingFace
  - LangChain

## Project Structure
```
This is how the files in this project are organized:
2500010_shakespeare_bot/
├── .gitignore # Files/folders Git should ignore (e.g., venv, .env)
├── README.md # Project description, setup instructions, usage
├── requirements.txt # Python dependencies (for pip install -r)
├── scripts/ # Utility scripts (database migration, deployment, data processing, model fine-tuning)
├── data/ # Raw data, processed datasets
├── models/ # gpt2-neo-125m fine-tuned by LoRA
├── retrieval/ # RAG system
└── utils/ # Deploy UI
    ├── 1_rag_inference.py # Deploy by Terminal
    └── 2_chatbot_interface.py # Deploy by Gradio local host
```
## Future Ideas

If I had more time, here's what I'd love to add or improve:

- Improve the correctness.
- Make the chatbot understand more complex sentences.
