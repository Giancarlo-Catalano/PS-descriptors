import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

# Load the data
data = pd.read_csv(r'C:\Users\gac8\PycharmProjects\PS-PDF\resources\temp\importance_table.csv')


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



pairs_to_test = [(f"SKILL_{n}", "variable_importance") for n in range(18)]
pairs_to_test.extend([("rota_counts","variable_importance"),
                      ("skill_counts", "variable_importance")])


for first_col, second_col in pairs_to_test:
    test_between_columns(first_col, second_col)
