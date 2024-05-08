
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('./Chapman_Ningbo_ECG_DB_Labeling_Info.csv')

# FILTERING AND MAPPING LABELS SECTION
# Filter out 'Unlabeled' and other specific labels
filtered_df = df[~df['Integration Code'].isin(['Unlabeled'])]

# Get unique labels and corresponding statement codes
unique_labels = filtered_df['Integration Code'].unique()
mapping_list = []
for label in unique_labels:
    statementcodes = filtered_df[filtered_df['Integration Code'] == label]['Snomed Code'].unique()
    statementcodes_str = " ".join(statementcodes.astype(str))
    mapping_list.append([label, statementcodes_str])

# Save the mapping list as a numpy file
mapping_array = np.array(mapping_list, dtype=object)
mapping_array[:, [1, 0]] = mapping_array[:, [0, 1]]
np.save('./target/mapping_labels.npy', mapping_array)

# REMOVING LABELS SECTION
# Extract and save statement codes for 'Unlabeled'
remove_labels = df[df['Integration Code'] == 'Unlabeled']['Snomed Code'].to_numpy()
np.save('./target/remove_labels.npy', remove_labels)

# TARGET LABELS SECTION
# Filter DataFrame to exclude certain groups and remove duplicates
target_df = df[~df['Group'].isin(['Unlabeled'])]
unique_codes = target_df['Integration Code'].drop_duplicates().to_numpy()
unique_names = target_df['Integration Name'].drop_duplicates().to_numpy()

# Categorize and save based on group
group_df = target_df[['Integration Code', 'Group']].drop_duplicates()
rhythm_df = group_df[group_df['Group'] == 'Rhythm']['Integration Code'].to_numpy()
duration_df = group_df[group_df['Group'] == 'Duration']['Integration Code'].to_numpy()
amplitude_df = group_df[group_df['Group'] == 'Amplitude']['Integration Code'].to_numpy()
morphology_df = group_df[group_df['Group'] == 'Morphology']['Integration Code'].to_numpy()

np.save('./target/target_labels.npy', unique_codes)
np.save('./target/target_labels_name.npy', unique_names)
np.save('./target/target_labels_rhythm.npy', rhythm_df)
np.save('./target/target_labels_duration.npy', duration_df)
np.save('./target/target_labels_amplitude.npy', amplitude_df)
np.save('./target/target_labels_morphology.npy', morphology_df)
