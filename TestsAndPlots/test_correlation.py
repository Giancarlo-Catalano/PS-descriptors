import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Load the data
data = pd.read_csv(r'C:\Users\gac8\Downloads\reattempt_worker_properties.csv')


def test_between_columns(col_a: str, col_b: str) -> None:
    # Extract columns A and B
    A = data[col_a]
    B = data[col_b]

    # Calculate the correlation coefficient and p-value
    correlation_coefficient, p_value = stats.pearsonr(A, B)

    # Calculate the R-squared value
    model = sm.OLS(B, sm.add_constant(A)).fit()
    r_squared = model.rsquared

    # Print the results
    print(f"Information about the correlation between {col_a} and {col_b}")
    print(f"\tCorrelation coefficient: {correlation_coefficient}")
    print(f"\tR-squared value: {r_squared}")
    print(f"\tP-value: {p_value}")



pairs_to_test = [["local_freq", "rota_quantity"],
                 ["local_freq", "skill_count"],
                 ["local_freq", "working_days"],
                 ["local_freq", "working_weekends"],
                 ["local_freq", "LIME"],
                 ["global_freq", "rota_quantity"],
                 ["global_freq", "skill_count"],
                 ["global_freq", "working_days"],
                 ["global_freq", "working_weekends"],
                 ["global_freq", "LIME"]]


for first_col, second_col in pairs_to_test:
    test_between_columns(first_col, second_col)
