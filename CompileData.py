import pandas as pd
import glob
import os
import sys

def combine_csv_files(input_folder, output_file, column_mappings, chunk_size=10000, sentiment_normalization_map=None, remove_duplicates=False):
    """
    Combines multiple CSV files from an input folder into a single output CSV file,
    handling different column names, skipping bad lines, normalizing sentiment labels
    (with optional file-specific maps), and optionally removing duplicate rows
    from the final output.

    Args:
        input_folder (str): Path to the folder containing input CSV files.
        output_file (str): Path to the output combined CSV file.
        column_mappings (dict): A dictionary where keys are patterns matching input
                                filenames and values are dictionaries specifying the
                                'text_col', 'sentiment_col', and optionally a
                                'sentiment_map' for that file pattern.
                                Example: {'file1.csv': {'text_col': 'text', 'sentiment_col': 'label'},
                                          'file2.csv': {'text_col': 'Sentence', 'sentiment_col': 'Sentiment',
                                                        'sentiment_map': {'Positive': 2, 'Negative': 0}}}
        chunk_size (int, optional): Number of rows to read per chunk. Defaults to 10000.
        sentiment_normalization_map (dict, optional): A dictionary to map various raw
                                                      sentiment values to a standard format.
                                                      This map is used if no file-specific
                                                      'sentiment_map' is provided in
                                                      column_mappings for a given file.
                                                      Keys should be lowercase strings.
                                                      Example: {'negative': 0, 'positive': 2, 'neutral': 1, '0': 0, '4': 2}
                                                      Defaults to None (no normalization).
        remove_duplicates (bool, optional): If True, remove duplicate rows from the
                                            final output_file after combination.
                                            Defaults to False.
    """
    print(f"\n--- Starting CSV Combination ---")
    print(f"Input folder: {input_folder}")
    print(f"Output file: {output_file}")
    print(f"Chunk size: {chunk_size}")
    if sentiment_normalization_map:
        print("General sentiment normalization map provided.")
    else:
        print("No general sentiment normalization map provided.")
    if remove_duplicates:
        print("Duplicate removal requested for the final output.")
    else:
        print("Duplicate removal not requested.")


    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Check if output file already exists, delete if it does to start fresh
    if os.path.exists(output_file):
        print(f"Output file '{output_file}' already exists. Deleting it to start fresh.")
        os.remove(output_file)

    all_files = glob.glob(os.path.join(input_folder, '*.csv'))

    if not all_files:
        print(f"Warning: No CSV files found in '{input_folder}'. Nothing to process.")
        return # Exit the function gracefully

    print(f"Found {len(all_files)} CSV files to potentially process.")

    write_header = True  # Initialize header writing state for this specific output file
    processed_files_count = 0
    total_rows_written_before_dedup = 0 # Track rows written during combination
    total_rows_skipped_normalization = 0

    for input_file_path in all_files:
        filename = os.path.basename(input_file_path)
        print(f"\nProcessing file: {filename}...")

        mapping = None
        # Find the correct mapping based on filename patterns
        # Prioritize exact match, then substring match
        exact_match_mapping = None
        substring_match_mapping = None

        for pattern, map_values in column_mappings.items():
            if pattern == filename:
                exact_match_mapping = map_values
                break # Found exact match, use this

        if exact_match_mapping:
             mapping = exact_match_mapping
             print(f"  Using exact match column mapping for '{filename}': {mapping}")
        else:
            # If no exact match, check for substring matches
            for pattern, map_values in column_mappings.items():
                 if pattern in filename:
                     substring_match_mapping = map_values
                     # Don't break, find the potentially best substring match if needed,
                     # but for simplicity here, the first substring match is used.
                     # If more complex pattern matching is needed, consider regex.
                     break
            if substring_match_mapping:
                 mapping = substring_match_mapping
                 print(f"  Using substring match column mapping for '{filename}': {mapping}")


        if mapping is None:
            print(f"  Warning: No column mapping found for '{filename}'. Skipping this file.")
            continue

        text_col_name = mapping.get('text_col')
        sentiment_col_name = mapping.get('sentiment_col')
        file_sentiment_map = mapping.get('sentiment_map') # Get file-specific map

        if not text_col_name or not sentiment_col_name:
            print(f"  Error: Incomplete mapping for '{filename}' (missing 'text_col' or 'sentiment_col'). Skipping.")
            continue

        # Determine which sentiment map to use for this file
        current_sentiment_map = file_sentiment_map if file_sentiment_map is not None else sentiment_normalization_map

        if current_sentiment_map:
            print(f"  Using sentiment map for '{filename}': {'File-specific map' if file_sentiment_map else 'General map'}")
        else:
            print(f"  No sentiment map available for '{filename}'. Sentiment column will not be normalized.")


        try:
            # Use iterator=True with chunksize for more robust chunking
            reader = pd.read_csv(
                input_file_path,
                chunksize=chunk_size,
                on_bad_lines='skip', # Skip rows with parsing errors
                low_memory=False, # Recommended for mixed types / large files
                iterator=True # Use iterator for manual chunk reading
            )

            file_rows_processed = 0
            file_rows_skipped = 0
            chunk_num = 0
            # Iterate through the chunks
            for chunk in reader:
                chunk_num += 1
                original_chunk_size = len(chunk)

                # Check if required columns exist in this chunk *after* potential skipping
                if text_col_name not in chunk.columns or sentiment_col_name not in chunk.columns:
                    print(f"  Warning: Expected columns '{text_col_name}' or '{sentiment_col_name}' not found in chunk {chunk_num} of file '{filename}'. Skipping chunk.")
                    continue # Skip this chunk

                # Select only the necessary columns
                chunk_selected = chunk[[text_col_name, sentiment_col_name]].copy() # Use .copy() to avoid SettingWithCopyWarning

                # Rename columns to the standard names 'text' and 'sentiment'
                chunk_standardized = chunk_selected.rename(columns={
                    text_col_name: 'text',
                    sentiment_col_name: 'sentiment'
                })

                # --- Sentiment Normalization Step ---
                if current_sentiment_map:
                    # Convert sentiment column to lowercase string to handle variations like 'Positive', 'positive', 'POSITIVE' and numeric strings like '0', '4'
                    # Ensure we handle potential NaN values before string conversion
                    # Use .loc to avoid SettingWithCopyWarning after subsetting
                    sentiment_col_str = chunk_standardized.loc[chunk_standardized['sentiment'].notna(), 'sentiment'].astype(str).str.lower()

                    # Apply the mapping using .loc for safe assignment
                    # Use .get() with a default value (like the original value) to keep rows
                    # that don't have a mapping instead of turning them into NaN immediately.
                    # We will drop NaNs later if needed.
                    chunk_standardized.loc[sentiment_col_str.index, 'sentiment'] = sentiment_col_str.apply(lambda x: current_sentiment_map.get(x, pd.NA)) # Use pd.NA for missing values

                    # Count rows before dropping NaNs caused by normalization
                    rows_before_drop = len(chunk_standardized)

                    # Drop rows where sentiment could not be normalized (became NaN or pd.NA)
                    chunk_standardized.dropna(subset=['sentiment'], inplace=True)

                    # Calculate how many rows were skipped in this chunk due to normalization failure
                    rows_skipped_this_chunk = rows_before_drop - len(chunk_standardized)
                    file_rows_skipped += rows_skipped_this_chunk
                    total_rows_skipped_normalization += rows_skipped_this_chunk

                    # Optional: Convert sentiment to integer if normalization map produces numbers
                    # Check if sentiment column contains non-integer types before converting
                    # if pd.api.types.is_numeric_dtype(chunk_standardized['sentiment']) and not pd.api.types.is_integer_dtype(chunk_standardized['sentiment']):
                    #     chunk_standardized['sentiment'] = chunk_standardized['sentiment'].astype(int)


                # --- End Sentiment Normalization ---

                rows_in_processed_chunk = len(chunk_standardized)

                if rows_in_processed_chunk > 0:
                    # Append the processed chunk to the output file
                    chunk_standardized.to_csv(
                        output_file,
                        mode='a', # Append mode
                        header=write_header, # Write header only for the first chunk of the first file
                        index=False # Don't write the pandas DataFrame index
                    )

                    # Ensure header is only written once per output file
                    if write_header:
                        write_header = False
                        print(f"  Header written to {output_file}")

                    file_rows_processed += rows_in_processed_chunk
                    total_rows_written_before_dedup += rows_in_processed_chunk

                if chunk_num % 10 == 0: # Print progress periodically
                     print(f"    Processed chunk {chunk_num}: Wrote {rows_in_processed_chunk} rows. ({file_rows_processed} total from this file, {total_rows_written_before_dedup} total written to output before dedup). Skipped {file_rows_skipped} in this file due to normalization.")


            print(f"  Finished processing '{filename}'. Added {file_rows_processed} valid rows. Skipped {file_rows_skipped} rows due to failed sentiment normalization.")
            processed_files_count += 1

        except FileNotFoundError:
            print(f"  Error: File not found at '{input_file_path}'. Skipping.")
        except pd.errors.EmptyDataError:
            print(f"  Warning: File '{filename}' is empty or became empty after skipping bad lines. Skipping.")
        except ValueError as ve:
             print(f"  A ValueError occurred while processing '{filename}': {ve}. This might indicate issues during processing or normalization. Skipping file.")
        except Exception as e:
            print(f"  An unexpected error occurred while processing '{filename}': {e}")

    print(f"\n--- Initial Combination Complete for {output_file} ---")
    print(f"Processed {processed_files_count} files.")
    print(f"Total valid rows initially written to '{output_file}': {total_rows_written_before_dedup}")
    print(f"Total rows skipped during file processing due to failed sentiment normalization: {total_rows_skipped_normalization}")
    print(f"--------------------------------------------------")

    # --- Duplicate Removal Step ---
    if remove_duplicates and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"\n--- Starting Duplicate Removal for '{output_file}' ---")
        try:
            # Read the combined file back into a DataFrame
            df_combined = pd.read_csv(output_file)

            initial_rows = len(df_combined)
            print(f"  Rows before duplicate removal: {initial_rows}")

            # Remove duplicates
            df_combined.drop_duplicates(inplace=True)

            rows_after_dedup = len(df_combined)
            duplicates_removed = initial_rows - rows_after_dedup
            print(f"  Rows after duplicate removal: {rows_after_dedup}")
            print(f"  Duplicates removed: {duplicates_removed}")

            if duplicates_removed > 0:
                # Save the de-duplicated DataFrame back to the file, overwriting the original
                df_combined.to_csv(output_file, index=False)
                print(f"  De-duplicated data saved back to '{output_file}'.")
            else:
                print("  No duplicates found. File remains unchanged.")


        except FileNotFoundError:
            print(f"  Error during duplicate removal: Output file '{output_file}' not found.")
        except pd.errors.EmptyDataError:
            print(f"  Warning during duplicate removal: Output file '{output_file}' is empty.")
        except Exception as e:
            print(f"  An unexpected error occurred during duplicate removal: {e}")
        print(f"--- Duplicate Removal Complete ---")
    elif remove_duplicates:
         print(f"\n--- Skipping Duplicate Removal ---")
         print(f"  Output file '{output_file}' does not exist or is empty after combination.")
         print(f"----------------------------------")


# --- Configuration ---
CHUNK_SIZE = 10000 # Adjust based on your memory capacity

# Define input folders
train_input_folder = "data/compile_data/train"
test_input_folder = "data/compile_data/test"
val_input_folder = "data/compile_data/val"

# Define output file paths
train_output_file = "data/compile_data/final/train.csv"
test_output_file = "data/compile_data/final/test.csv"
val_output_file = "data/compile_data/final/validation.csv"

# --- Sentiment Normalization Maps ---
# Define standard outputs: 0 = negative, 1 = neutral, 2 = positive
# Keys MUST be lowercase strings, as the code converts inputs to lowercase string first.
common_sentiment_map = {
    # Text variations (lowercase)
    'negative': 0,
    'neg': 0,
    'neutral': 1,
    'neu': 1,
    'positive': 2,
    'pos': 2,
    'good': 2,
    'bad': 0,
    # Numeric variations (as strings)
    '0': 0, # Assuming 0 is negative
    '1': 1, # Assuming 1 is neutral (adjust if 1 means positive in some files)
    '2': 2, # Assuming 2 is positive (common case)
    # '-1': 0, # Example if -1 is used for negative
    # '4': 2, # Example: 0=neg, 2=neu, 4=pos -> map '4' to 2 (positive)
    # '2': 1, # Example: 0=neg, 2=neu, 4=pos -> map '2' to 1 (neutral) - uncomment *one* '2' mapping
}

# Example of a specific map for a file that uses 0, 2, 4
sentiment_map_0_2_4 = {
    '0': 0, # Negative
    '2': 1, # Neutral
    '4': 2  # Positive
}

sentiment_map_train_7 = {
    '0': 0,
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 2
}

sentiment_map_train_8 = {
    '0': 1,
    '-1': 0,
    '1': 2
}

sentiment_map_train_9 = {
    '1': 0,
    '2': 2
}


# --- Column Mappings ---
# Now includes optional 'sentiment_map' key for file-specific maps
train_column_mappings = {
    'train-1.csv': {'text_col': 'text', 'sentiment_col': 'label'}, # Uses common map
    'train-2.csv': {'text_col': 'text', 'sentiment_col': 'sentiment'}, # Uses common map
    'train-3.csv': {'text_col': 'text', 'sentiment_col': 'sentiment'}, # Uses common map
    'train-4.csv': {'text_col': 'Sentence', 'sentiment_col': 'Sentiment'}, # Uses common map
    'train-5.csv': {'text_col': "text", 'sentiment_col': "sentiment", 'sentiment_map': sentiment_map_0_2_4}, # Uses specific 0,2,4 map
    'train-6.csv': {'text_col': "tweets", 'sentiment_col': "labels"},
    'train-7.csv': {'text_col': "text", 'sentiment_col': "label", 'sentiment_map': sentiment_map_train_7},
    'train-8.csv': {'text_col': "clean_comment", 'sentiment_col': "category", 'sentiment_map': sentiment_map_train_8},
    'train-9.csv': {'text_col': "text", 'sentiment_col': "sentiment", 'sentiment_map': sentiment_map_train_9},
}

test_column_mappings = {
    'test-1.csv': {'text_col': 'text', 'sentiment_col': 'sentiment'}, # Uses common map
    'test-2.csv': {'text_col': 'text', 'sentiment_col': 'sentiment'}, # Uses common map
    'test-3.csv': {'text_col': 'text', 'sentiment_col': 'sentiment', 'sentiment_map': sentiment_map_train_9},
}

validation_column_mappings = {
    'val-1.csv': {'text_col': 'text', 'sentiment_col': 'sentiment'}, # Uses common map
    'val-2.csv': {'text_col': 'text', 'sentiment_col': 'sentiment'}, # Uses common map
}


# --- Execution ---

# Process Training Data
print("\nStarting Train Data Processing...")
combine_csv_files(
    train_input_folder,
    train_output_file,
    train_column_mappings,
    CHUNK_SIZE,
    sentiment_normalization_map=common_sentiment_map, # Pass the general normalization map
    remove_duplicates=True
)

# Process Testing Data
print("\nStarting Test Data Processing...")
combine_csv_files(
    test_input_folder,
    test_output_file,
    test_column_mappings,
    CHUNK_SIZE,
    sentiment_normalization_map=common_sentiment_map, # Use the same general map or a specific one
    remove_duplicates=True
)

# Process Validation Data
print("\nStarting Validation Data Processing...")
combine_csv_files(
    val_input_folder,
    val_output_file,
    validation_column_mappings,
    CHUNK_SIZE,
    sentiment_normalization_map=common_sentiment_map, # Use the same general map or a specific one
    remove_duplicates=True
)

print("\nAll processing finished.")
