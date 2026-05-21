import sys
from common.balanced_sample_tool import TheorySampler
from pyspark.sql import SparkSession
import matplotlib


print(sys.version)
print("GIL enabled:", sys._is_gil_enabled())

sampler = TheorySampler("Theories_dedup_by_SCI.csv")
print(sampler.get_theory_num())
stats = sampler.get_theory_stats()
rows = stats.collect()
for row in rows:
    print(row.asDict())

selected = sampler.get_balanced_sample((0.75, 1.5), (0.75, 1.5), 50)
selected.df.show(truncate=False)

selected = sampler.get_manual_sample(["SU2nf4", "SO5adj1nf2", "SU3adj1nf2"], 50)
selected.df.show(truncate=False)
