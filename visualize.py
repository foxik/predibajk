#!/usr/bin/env python3
import argparse
import os

import numpy as np

header = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>PrediBajk Evaluation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css">
    <link rel="stylesheet" href="https://unpkg.com/bootstrap-table@1.20.1/dist/bootstrap-table.min.css">
    <script src="https://cdn.jsdelivr.net/npm/jquery/dist/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
    <script src="https://unpkg.com/bootstrap-table@1.20.1/dist/bootstrap-table.min.js"></script>
  </head>
  <body>
"""

footer = """
  </body>
</html>
"""

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="Data path.")
parser.add_argument("output", type=str, help="Output path.")
args = parser.parse_args()

with open(args.data, "r") as data_file:
    columns = data_file.readline().rstrip("\r\n").split("\t")
    rows = [line.rstrip("\r\n").split("\t") for line in data_file]

output = open("{}.html".format(args.output), "w")

# First generate full table with results
def accuracy(i):
    return np.mean([row[i] == row[i + 1] for row in rows])

print(header, file=output)
print(" <table data-toggle='table' data-search='true' data-visible-search='true' data-pagination='true'><thead>", file=output)
print("  <tr>",
      *["<th rowspan='2' data-sortable='true'>{}".format(column) for column in ["Original", "Image path"]],
      *["<th colspan='2'><a href='{}.confusion-tables.html#{}'>{}&nbsp;{:.2f}%</a>".format(
          args.output, columns[i], columns[i], 100 * accuracy(i))
        for i in range(2, len(columns), 2)], sep="", file=output)
print("  <tr>",
      "<th data-sortable='true'>Gold<th data-sortable='true'>Pred" * len(columns[2::2]),
      "</thead><tbody>", sep="", file=output)
for row in rows:
    print("  <tr>",
          "<td><a href='{}'>{}</a>".format(row[1], os.path.basename(row[1])),
          "<td><a href='{}'>{}".format(row[0], row[0]),
          *["<td>{}<td class='table-{}'><span class='{}'>{}</span>".format(
              row[i], "success" if row[i + 1] == row[i] else "danger", "ok" if row[i + 1] == row[i] else "bad", row[i + 1])
            for i in range(2, len(row), 2)], sep="", file=output)
print(" </tbody></table>", file=output)
print(footer, file=output)
output.close()

# Then, generate the individual confusion tables
output = open("{}.confusion-tables.html".format(args.output), "w")
print(header, file=output)

for i in range(2, len(columns), 2):
    print(" <h3 id='{}'>{}</h3>".format(columns[i], columns[i]), file=output)
    values = sorted(set(row[i] for row in rows) | set(row[i + 1] for row in rows))
    values_map = {value: index for index, value in enumerate(values)}
    confusion = np.zeros([len(values), len(values)], np.int32)
    for row in rows:
        confusion[values_map[row[i]], values_map[row[i + 1]]] += 1
    limit = max(np.max(np.tril(confusion, -1)), np.max(np.triu(confusion, 1)))
    print(" <table class='table table-sm table-hover table-bordered' style='white-space: nowrap'><thead>", file=output)
    print("  <tr><th>Gold \\ Pred",
          *["<th style='writing-mode: vertical-rl; text-align: right'>{}".format(value) for value in values],
          "</thead><tbody>", sep="", file=output)
    for j, value in enumerate(values):
        print("  <tr><td>{}".format(value),
              *["<td style='background-color:rgb(255, {0}, {0})'>{1}".format(int(255 * max(0, 1 - confusion[j, k] / limit)), confusion[j, k])
                for k in range(len(values))], sep="", file=output)
    print(" </tbody></table>", file=output)

print(footer, file=output)
output.close()
