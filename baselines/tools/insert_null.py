
import pandas as pd
import random


def inject_missing_values(csv_file, output_file, attributes_error_ratio=None, missing_value_in_ori_data='empty',missing_value_representation='empty'):
    """
    Inject missing values. Existing nulls in the source data are first normalized to the given representation.

    Args:
        csv_file (str): input CSV path
        output_file (str): output CSV path
        attributes_error_ratio (dict): mapping from attribute to error ratio (%)
        missing_value_in_ori_data (str): existing null representation in the source data (default "empty")
        missing_value_representation (str): target null representation (default "empty")

    Output:
        The CSV file with injected errors.
    """
    # Load the CSV.
    df = pd.read_csv(csv_file)
    # Cast float values that are whole numbers back to integers for each column.
    for col in df.columns:
        # Is the column a float dtype?
        if pd.api.types.is_float_dtype(df[col]):
            # Cast values that are integer-valued floats back to int.
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) and x == int(x) else x)

        # Finally cast every column to string.
        df[col] = df[col].astype(str)
    # Preprocessing: normalize any existing null (NaN or empty string) to missing_value_representation.
    df = df.fillna(missing_value_representation)
    df.replace('', missing_value_representation, inplace=True)
    df.replace('nan', missing_value_representation, inplace=True)
    df.replace('_nan_', missing_value_representation, inplace=True)
    df.replace('null', missing_value_representation, inplace=True)
    df.replace('__NULL__', missing_value_representation, inplace=True)
    df.replace(missing_value_in_ori_data, missing_value_representation, inplace=True)
    if attributes_error_ratio is None:
        print("No error ratio specified; only normalizing existing nulls without injecting new errors.")
    else:
    # Iterate over attributes and inject nulls.
        for attribute, error_ratio in attributes_error_ratio.items():
            if attribute in df.columns:
                num_rows = len(df)
                num_errors = int(num_rows * error_ratio / 100)
                error_indices = random.sample(range(num_rows), num_errors)

                # Replace the selected rows with the null representation.
                df.loc[error_indices, attribute] = missing_value_representation

    # Save the injected file.
    df.to_csv(output_file, index=False)
    print(f"Injected file saved to: {output_file}")
if __name__ == "__main__":
    # Usage example.
    # Attribute list.
    attributes = [
        "journal_issn",
        "journal_title",
        "jounral_abbreviation",
    ]

    # 2% error ratio per attribute.
    attributes_error_ratio = {attribute: 0 for attribute in attributes}

    # If the clean data has nulls, normalize them in the clean data too.
    inject_missing_values(
        csv_file='../Data/3_beers/clean_index.csv',
        output_file='../Data/3_beers/clean_index.csv',
        attributes_error_ratio=None,
        missing_value_in_ori_data='NaN',
        missing_value_representation='empty'
    )
