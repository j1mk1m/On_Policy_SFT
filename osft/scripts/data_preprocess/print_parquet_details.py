import sys
import pyarrow.parquet as pq

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <file.parquet>")
    sys.exit(1)

pf = pq.ParquetFile(sys.argv[1])
print(pf.schema.names)
print(pf.metadata.num_rows)

table = pf.read_row_group(0).slice(0, 10)

df = table.to_pandas()
index = 4
print(df.columns)
print(df.iloc[index])
print(df.iloc[index]["prompt"])

