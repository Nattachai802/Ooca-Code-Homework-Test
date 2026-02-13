# Support Ticket Agent(Ooca Test)

AI-powered support ticket system using OpenAI function calling and Pydantic structured output

## Prerequisites

- Python 3.11+
- OpenAI API key

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Ooca
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit the `.env` file and add your OpenAI API key:

```
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=your model
```

### 5. Run the agent

```bash
python main.py
```

## Project Structure

```
Ooca/
├── agent.py         
├── models.py      
├── tools.py     
├── vector_store.py  
├── main.py  
├── requirements.txt   
├── prompts/
│   └── system_prompt.txt  
├── data/
│   ├── customers.json     # Customer profiles
│   ├── plan_tiers.json    # Plan tier definitions (Free, Pro, Enterprise)
│   ├── knowledge_base.json # KB articles for semantic search
│   └── sample_tickets.json # Sample support tickets
└── chroma_db/           # ChromaDB persistent storage (auto-generated)
```
