# Proba-DM-Language

**Proba-DM-Language** is a domain-specific language (DSL) system designed for **probabilistic data mining and analysis**. It combines the rigor of **probability theory**, **statistics**, and **data mining** into a user-friendly interface, allowing analysts and researchers to uncover insights using probabilistic queries and models.

Built using [ANTLR](https://www.antlr.org/) for parsing and [Flask](https://flask.palletsprojects.com/) for web integration, the system supports advanced features such as distribution-based variables, statistical queries, and data mining techniques — all within a web or desktop UI.

---

## Features

- **Probabilistic Variables**: Define variables using statistical distributions.
- **Query Language (PQL)**: Express uncertainty and ask probabilistic questions.
- **Data Mining Tools**: Clustering, classification, and association rule mining.
- **Visualization**: Dynamic charts for interpreting results.
- **Syntax Validation**: Built-in parser checks your queries before execution.

---

## Project Structure
```text
Proba-DM-Language/
│
├── app.py                # Flask web application
├── main.py               # Main entry point
├── run.py                # Utility script to generate ANTLR parser
├── constants.py          # Configuration file (ANTLR JAR path, etc.)
├── ProbDataMine.g4       # ANTLR grammar for the PQL language
│
├── components/           # Core modules (execution engine, parser interface)
├── static/               # Frontend static files (CSS, JS)
├── templates/            # HTML templates for web UI
├── CompiledFiles/        # Auto-generated ANTLR parser files
```


<p>&nbsp;</p>


## Installation

### 1. Python & Package Setup

```bash
python.exe -m pip install --upgrade pip
pip install Flask
pip install pandas
pip install numpy
pip install -U matplotlib
pip install scipy
pip install scikit-learn
pip install mlxtend
```

### 2. ANTLR Setup
- Download ANTLR from [ANTLR](https://www.antlr.org/)
- Ensure Java is installed on your system
- Update the ANTLR JAR path in constants.py according to your local library folder: 
```bash
ANTLR_JAR = "C:\\JavaLibrary\\antlr-4.13.2-complete.jar"
```

<p>&nbsp;</p>


## Running the Application
### 1. Generate the ANTLR Parser
Before running the application, generate the parser:
```bash
python run.py gen or py run.py gen
```

### 2. Start the Web Interface
```bash
py main.py
```
Then navigate to: ```http://localhost:5000``` or ```http://***.*.*.*:5000```

