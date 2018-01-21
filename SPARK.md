### Missing data

```
val rawData = sc.parallelize( 
    Array( 
      Array(1, null, 3, 4), 
      Array(2, 3, null, null),
      Array(3, 1, null, null)
    )
)

val count = rawData.count()

val nanCountPerColumn = rawData.map { row =>
   row.map(v => if (v == null) 1 else 0)
 }.reduce((v1, v2) => v1.indices.map(i => v1(i) + v2(i)).toArray)
 
 // res2: Array[Int] = Array(0, 1, 2, 2)
 
 val nanPctPerColumn = nanCountPerColumn.map { v =>
   if (v == 0) v else v.toFloat/count
 }
 
 // nanPctPerColumn: Array[Float] = Array(0.0, 0.33333334, 0.6666667, 0.6666667)
```
