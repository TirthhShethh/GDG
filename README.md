# GDG
# GDG Task Solutions

This repository contains solutions for two distinct tasks:

📁 Project Structure

```
├── README.md          # Project documentation (this file)
├── task1.py           # Solution for Task 1 (Forest Analysis & IPL Team Selection)
└── task2.py           # Solution for Task 2 (F1 DNF Data Analysis)
```

🎯 Task 1: Multi-Domain Problem Solver

Part A: Forest Zone Extraction
- Function: `extract_zone(forest_map, center, m)`
- Purpose: Extracts an m×m zone from a forest map centered at given coordinates
- Features:
  - Handles boundary cases where extraction zone exceeds map boundaries
  - Calculates total Lal Chandan trees in the extracted zone
  - Displays the extraction zone matrix with visual output

Part B: IPL Dream Team Selection
- Function: `find_best_team(k)`
- Purpose: Selects optimal cricket team of k players based on strengths and weaknesses
- Scoring System:
  - +1 point for each unique strength
  - -1 point for each unique weakness
  - Bonus points for key roles: opener, finisher, death_bowling
  - Penalty for overlapping strengths and weaknesses

Players Available: Kohli, Rahul, Bumrah, Jadeja, Maxwell, Siraj, Shreyas, Chahal, DK, Faf

📊 Task 2: F1 DNF Data Analysis

EDA Script for Formula 1 Did Not Finish (DNF) Data

Feature:
- Data Loading & Cleaning: Automatically handles CSV loading and basic data cleaning
- Missing Value Analysis: Comprehensive missing value report with visualization
- Data Distribution: Numeric column distributions and frequency analysis
- Temporal Analysis: Year-wise record distribution (if date/year columns available)
- Categorical Analysis: Top drivers, teams, and retirement reasons visualization

Output:
- `f1_dnf_cleaned.csv` - Cleaned dataset
- `plots/` directory containing:
  - Missing values visualization
  - Numeric distributions
  - Year-wise analysis
  - Top drivers/teams charts
  - Retirement reason analysis


🚀 Usage

Task 1
```bash
python task1.py
```
- Follow prompts to input team size (k) for IPL team selection
- Forest analysis runs with predefined test case

Task 2
```bash
python task2.py
```
Prerequisite: Place `f1_dnf.csv` in the same directory before running
Output: Comprehensive EDA report with visualizations and cleaned data

🛠 Requirements

- Python 3.6+
- Required libraries for Task 2:
  - pandas
  - numpy
  - matplotlib
  - seaborn

📝 Notes

- Task 1 demonstrates problem-solving across different domains (environmental analysis and sports analytics)
- Task 2 showcases automated data analysis pipeline with comprehensive reporting
- Both tasks include error handling and edge case management