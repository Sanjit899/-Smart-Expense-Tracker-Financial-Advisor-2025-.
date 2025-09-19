:

💸 Smart Expense Tracker & Financial Advisor (2025)

A production-ready Streamlit app to analyze personal finances, categorize transactions, and provide actionable insights. Designed for real-life bank statements (CSV & PDF), combining rule-based and ML-powered categorization with visualization and reporting features.


🔹 Features
1. Data Upload & Parsing

Supports CSV and PDF bank statements.

Handles messy real-life bank statements with noisy formatting.

Built-in sample demo data for quick testing.

2. Automatic Categorization

Rule-based keyword matching for instant categorization (salary, dining, utilities, shopping, etc.).

ML classifier (Tfidf + Logistic Regression) trained on sample transactions.

Handles unknown or unusual descriptions with fallback category: others.

3. Data Cleaning & Standardization

Standardizes column names: Date, Description, Amount, Type, Category.

Cleans numeric amounts (removes commas, currency symbols).

Converts transaction types to correct sign: Credits positive, Debits negative.

Extracts transaction month automatically.

4. Summary & Insights

Income, Expense, Net displayed as metrics.

Category-wise expense breakdown with interactive bar charts.

Monthly trends via line charts.

Budget insights using the 50/30/20 rule:

Needs (utilities, groceries, rent, healthcare)

Wants (entertainment, shopping, dining, transport)

Savings

Savings tips & warnings based on spending patterns.

5. Downloads

Export categorized transactions as:

CSV

Excel

6. Streamlit Sidebar Features

File uploader (CSV or PDF)

Option to use built-in sample demo data

Interactive real-time analysis

🔹 Sample Categories
Keyword Group	Example Categories
salary, payroll, deposit	salary
rent	rent
electricity, water, bill	utilities
uber, ola, taxi, flight	transport
coffee, dining, pizza	dining
amazon, flipkart, shopping	shopping
netflix, spotify, subscription	entertainment
grocery, supermarket	groceries
pharmacy, medical	healthcare
insurance, premium	insurance
interest, dividend	interest
transfer, upi, neft	transfer
atm, withdrawal	withdrawal
gym, fitness	fitness
🔹 Performance Test Results

Tested on a complex real-life styled PDF bank statement.

✅ Strengths

Successfully extracted 13+ transactions.

Category classification using rules + ML performed accurately.

Handles noisy data: symbols, commas, inconsistent formatting.

Month extraction works perfectly (2025-07, 2025-08).

No crashes during parsing or visualization.

⚠️ Weak Spots / Limitations

Some amounts extracted incorrectly (e.g., Salary: 0.0, ATM withdrawal misparsed).

PDF parsing is sensitive to very messy formatting.

Minor improvements needed in parse_pdf() regex and Debit/Credit handling.


🔹 Technology Stack

Frontend & App: Streamlit

Data Handling: Pandas, NumPy

ML: Scikit-learn (TfidfVectorizer + Logistic Regression)

PDF Parsing: pdfplumber (optional)

Visualization: Streamlit charts, Matplotlib

Python Version: 3.10+

Create a virtual environment:
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

Install dependencies:pip install -r requirements.txt

Run the Streamlit app:streamlit run app.py
Open the app in your browser: http://localhost:8501

🔹 Usage

Upload your CSV or PDF bank statement.

The app auto-categorizes transactions.

Explore:

Summary metrics: Income, Expense, Net

Expense breakdown by category

Monthly trends

Budget insights (50/30/20 rule)

Download the cleaned, categorized data as CSV or Excel.

🔹 Future Improvements

Enhanced PDF parsing for messy statements.

Better amount extraction & type detection.

Custom ML training on user’s historical data.

Interactive dashboards with drill-down features.

Alerts & recommendations via email or SMS.

🔹 License

MIT License ©.
