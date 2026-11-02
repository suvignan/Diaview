
# DiaView: Type 1 Diabetes Dashboard

DiaView is an interactive web dashboard that helps people with Type 1 diabetes understand their glucose data. It tracks glucose, carbs and insulin, finds recurring glucose patterns with time-series clustering, and turns the numbers into plain-language insights.

> **Note:** The app currently runs on **synthetic CGM-style data** (5-minute glucose readings over 15 days, with meals, insulin doses, hypoglycemia events and a simulated pump interruption) generated at startup. No real patient data is used.

## Features

**Visualizations**
- **Ambulatory Glucose Profile (AGP):** hourly 10th, 25th, 50th, 75th and 90th percentile bands, the standard view clinicians use to read CGM data.
- **Horizon graphs** for comparing glucose across multiple selected days.
- **Time in range** and treatment (carbs, bolus and basal insulin) summaries.
- **CSV export** of daily statistics and detected patterns.

**Pattern detection (ML)**
- Extracts 3-hour glucose windows after every meal and after every hypoglycemia event (glucose dropping below 70 mg/dL).
- Normalizes each window (mean-variance scaling) and clusters them with **time-series K-Means using Dynamic Time Warping (DTW)** via `tslearn`.
- DTW matches curves by shape even when they are shifted in time (for example, a spike at 30 minutes after one meal and 50 minutes after another), which plain Euclidean distance would treat as different.

**Personalized insights**
- Mean glucose, time in range against a user-set goal range, and hypoglycemia counts for a chosen day or date range.
- Suggestions adjusted for the user's age, weight and activity level.
- A simple linear-regression **trend indicator** that flags whether glucose is drifting toward low or high levels.

**Diary and Learn tabs**
- Day-by-day diary with previous/next navigation.
- Educational content on hypoglycemia, insulin, carb counting, exercise, stress and meal planning.

## Tech stack

Python, Dash, Plotly, pandas, NumPy, scikit-learn, tslearn, Tailwind CSS

## Getting started

```bash
git clone https://github.com/suvignan/Diaview.git
cd Diaview
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:8050 in your browser.

## Project structure

```
Diaview/
├── app.py            # Data generation, ML functions, layout and callbacks
├── requirements.txt
└── README.md
```

## Limitations and next steps

- Uses synthetic data; the next step is supporting CSV uploads from real CGM exports (e.g. Dexcom, FreeStyle Libre).
- The trend estimate is a linear fit over the selected period, so it is an indicator, not a clinical forecast.
- Insulin and carb suggestions are rule-based and for educational purposes only.

## Disclaimer

DiaView is a learning project and is **not a medical device**. It does not provide medical advice. Always consult a healthcare professional before changing insulin doses or diet.
