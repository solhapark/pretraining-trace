import pandas as pd

csv_path = "./HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
df = pd.read_csv(csv_path)

print(df.columns)

print("\n Standard category:")
print(df[df['FunctionalCategory'] == 'standard'])
print(df[df['FunctionalCategory'] == 'standard'].shape) # (200, 6)

print("\n Contextual category:")
print(df[df['FunctionalCategory'] == 'contextual'])
print(df[df['FunctionalCategory'] == 'contextual'].shape) # (100, 6)

print("\n Copyright category:")
print(df[df['FunctionalCategory'] == 'copyright'])
print(df[df['FunctionalCategory'] == 'copyright'].shape) # (100, 6)