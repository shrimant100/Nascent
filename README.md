# Nascent

Hi, Please refer to the following:

## Project Structure

```
Nascent/
├── data/                  # Data files
│   ├── raw/              # Raw, unprocessed data files
│   └── cleaned/          # Processed and cleaned data files
├── notebooks/            # Jupyter notebooks for analysis and experimentation
│   ├── tasks.ipynb       # Task-specific analysis notebooks
│   └── summary_report.ipynb  # Summary and reporting notebooks
├── reports/              # Generated reports and outputs
│   └── Summary_Report.pdf
├── src/                  # Source code modules
│   └── utils.py          # Utility functions and helper modules
├── venv/                 # Python virtual environment (not tracked in git)
├── requirements.txt      # Python package dependencies
└── README.md            # This file
```

## Setup

### Prerequisites
- Python 3.12+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/shrimant100/Nascent.git
cd Nascent
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

The project uses the following main libraries:
- **pandas**: Data manipulation and analysis
- **langchain**: Framework for building LLM applications
- **langchain-core**: Core LangChain functionality
- **langchain-text-splitters**: Text splitting utilities
- **langsmith**: LangChain monitoring and debugging

## Usage

1. Activate the virtual environment
2. Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab
# or
jupyter notebook
```
3. Open and run notebooks from the `notebooks/` directory

## Data

- Raw data files should be placed in `data/raw/`
- Processed/cleaned data should be saved to `data/cleaned/`

## Development

Source code utilities and helper functions are located in the `src/` directory and can be imported in notebooks or other Python scripts.


## Author

shrimant100

