+======================================================================+
|                            dazdiff report                            |
+======================================================================+
   Identity: id, region
   Mode:     benchmark: stored is tested against benchmark

 Columns
 -------
   Compared 4 columns: 2 keys, 2 measures.
   Only in stored, dropped: loaded_at.
   Truncated by the cast to benchmark's types:
     units: 1 value (10.4 -> 10)

 Rows
 ----
   benchmark rows               stored rows
         4         (no change)       4

   +=+=+=+=+=+
   |-|-|-|-|-|                1 only in benchmark (25.00%)
   +-+-+-+-+-+...+=+=+=+=+=+................................+
   | | | | | | = | | | | | |  2 same              (66.67%)  |
   +-+-+-+-+-+...+-+-+-+-+-+................................+-  3 joined
   | | | | | | ! | | | | | |  1 changed           (33.33%)  |
   +=+=+=+=+=+...+-+-+-+-+-+................................+
                 |+|+|+|+|+|  1 only in stored    (25.00%)
                 +=+=+=+=+=+

 Measures
 --------
   +---------+------------+------------+--------------+
   | Measure | Match rate | Max |diff| | Max rel diff |
   +---------+------------+------------+--------------+
   | amount  |     66.67% |        0.5 |     0.002494 |
   +---------+------------+------------+--------------+
   1 other measure matches on every row found in both tables.

 Rows only in benchmark
 ----------------------
   +----+--------+
   | id | region |
   +----+--------+
   | 4  | "west" |
   +----+--------+

 Rows only in stored
 -------------------
   +----+---------+
   | id | region  |
   +----+---------+
   | 5  | "north" |
   +----+---------+
