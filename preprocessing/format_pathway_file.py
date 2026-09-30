import pandas as pd

raw_file = pd.read_csv("/Users/elias/Downloads/NCBI2Reactome.txt", sep="\t", header=None)

# 0,1,3,4,5
# 1,R-HSA-114608,Platelet degranulation ,TAS,Homo sapiens

raw_file = raw_file[[0, 1, 3, 4, 5]]

human_only = raw_file[raw_file[5] == "Homo sapiens"]

tas_only = human_only[human_only[4] == "TAS"]

print(tas_only.head())
tas_only.to_csv("data/reactome_pathways_lowest_filtered.csv", index=False)
