## CSV Scheme

|ID|Descriptor|Val 1 | Val 2 | ... | Val m | 

- ID: Experiment/Configuration ID
- Descriptor: Type of Benchmark (TBD)
- Vals: Value of metrics collected, semantics defined by descriptor

## Benchmar IO

read_results(filename) -> number of columns n, dictionary of columns with keys "id", "descriptor", "val1" - "val<n-2>"
