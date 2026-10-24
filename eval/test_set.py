TEST_SET = [
    # --- Code Generation Queries (Pass@1) ---
    {
        "question": "Write a snippet using empyrical to compute the Sharpe ratio of a pandas Series of daily returns.",
        "ground_truth": "The correct answer uses sharpe_ratio from empyrical, accepts a pandas Series of returns, and mentions the default daily annualization factor of 252."
    },
    {
        "question": "Calculate the maximum drawdown of a return series using empyrical.",
        "ground_truth": "The correct answer imports max_drawdown from empyrical and applies it to a pandas Series or numpy array of returns to return the maximum observed drawdown."
    },
    {
        "question": "How do I compute the annual return of a portfolio using empyrical?",
        "ground_truth": "The correct answer uses annual_return from empyrical, passing the returns series and specifying the appropriate annualization period if necessary."
    },
    {
        "question": "Write code to calculate the Calmar ratio using empyrical.",
        "ground_truth": "The correct answer utilizes calmar_ratio from empyrical, passing a series of returns to measure the relative risk-adjusted return against max drawdown."
    },
    {
        "question": "Show how to calculate the Sortino ratio using empyrical.",
        "ground_truth": "The correct answer uses sortino_ratio from empyrical, passing returns and optionally a target or required rate of return."
    },
    {
        "question": "How can I compute alpha and beta for a strategy against a benchmark using empyrical?",
        "ground_truth": "The correct answer calls alpha_beta or individual alpha and beta functions from empyrical, passing both the strategy returns and benchmark returns series."
    },
    {
        "question": "Write a python snippet to compute excess returns over a risk-free rate using empyrical utilities.",
        "ground_truth": "The correct answer demonstrates how to calculate excess returns or calls excess_sharpe from empyrical with a specified risk-free rate."
    },

    # --- Explanation Queries ---
    {
        "question": "What does the risk_free parameter in sharpe_ratio do?",
        "ground_truth": "The risk_free parameter specifies the constant risk-free rate during the period, which is subtracted from the returns before calculating the Sharpe ratio."
    },
    {
        "question": "What annualization factor does empyrical use by default for daily returns?",
        "ground_truth": "Empyrical uses an annualization factor of 252 for daily returns to scale metrics to an annual basis."
    },
    {
        "question": "What is the difference between sharpe_ratio and excess_sharpe in empyrical?",
        "ground_truth": "Sharpe ratio measures risk-adjusted return over a risk-free rate, whereas excess sharpe calculates the Sharpe ratio of the excess returns over a benchmark or risk-free rate stream."
    },
    {
        "question": "How does annual_volatility compute volatility in empyrical?",
        "ground_truth": "Annual volatility computes the standard deviation of returns scaled by the square root of the annualization factor (such as 252 for daily data)."
    },

    # --- Edge Case / Recall Queries ---
    {
        "question": "What happens when sharpe_ratio is called with a return series containing fewer than 2 data points?",
        "ground_truth": "With insufficient data points (less than 2), standard deviation cannot be reliably calculated, leading to NaN or a division-by-zero handling in empyrical."
    },
    {
        "question": "How does empyrical handle NaN or missing values in a returns series?",
        "ground_truth": "Empyrical functions typically drop or clean NaN values from the returns series before performing statistical calculations."
    },
    {
        "question": "Can empyrical handle monthly and weekly return periods as well as daily?",
        "ground_truth": "Yes, empyrical accepts period parameters like monthly (12) or weekly (52) to adjust the annualization factor appropriately."
    },
    {
        "question": "What is required for the benchmark series when computing alpha and beta?",
        "ground_truth": "The benchmark returns series must be aligned by index and time period with the strategy returns series for accurate regression calculations."
    }
]