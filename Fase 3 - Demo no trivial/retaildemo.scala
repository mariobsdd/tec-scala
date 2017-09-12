import org.apache.spark.sql.Row
import sqlContext.implicits._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.KMeans


import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}

import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}


import org.apache.spark.sql.functions.{stddev_samp, stddev_pop}

/* CLUSTERING DE CLIENTES (4 CLUSTERS) */
//Con todas las variables
//val data = sqlContext.sql("SELECT nit, last_visit, fct1, ct1, fct2, ct2, fct3, ct3, fct4, ct4, fct5, ct5, fct6, ct6, fct7, ct7, fct8, ct8, fct9, ct9, fct10, ct10, fct11, ct11, fct12, ct12, fct13, ct13, fct14, ct14, fct15, ct15, total_compras, total_visitas, mgt1, mgt2, mgt3, mgt4, mgt5, mgt6, mgt7, mgt8, mgt9, mgt10, mgt11, mgt12, mgt13, mgt14, mgt15, total_spent FROM retail.client");
//con las variables de las veces que visito, lo que gasto y la cantidad de compras realizadas en el CC
val data = sqlContext.sql("SELECT nit, last_visit, total_compras, total_visitas, total_spent FROM retaildemo.client");
data.printSchema;

//val assembler = new VectorAssembler().setInputCols(Array("last_visit", "fct1", "ct1", "fct2", "ct2", "fct3", "ct3", "fct4", "ct4", "fct5", "ct5", "fct6", "ct6", "fct7", "ct7", "fct8", "ct8", "fct9", "ct9", "fct10", "ct10", "fct11", "ct11", "fct12", "ct12", "fct13", "ct13", "fct14", "ct14", "fct15", "ct15", "total_compras", "total_visitas", "mgt1", "mgt2", "mgt3", "mgt4", "mgt5", "mgt6", "mgt7", "mgt8", "mgt9", "mgt10", "mgt11", "mgt12", "mgt13", "mgt14", "mgt15", "total_spent")).setOutputCol("features")
val assembler = new VectorAssembler().setInputCols(Array("last_visit", "total_compras", "total_visitas", "total_spent")).setOutputCol("features")
val df_all = assembler.transform(data)


// model
val kmeans_4 = new KMeans().setK(4).setFeaturesCol("features").setPredictionCol("prediction")
val model_4 = kmeans_4.fit(df_all)
val wss_4 = model_4.computeCost(df_all)


/** Predictions and metrics 50f */
model_4.clusterCenters.foreach(println)
val conteo_clust = model_4.transform(df_all)
conteo_clust.select("prediction").groupBy("prediction").agg($"prediction",count("prediction")).show

val cols = List("nit", "prediction")
val todo = conteo_clust.select(cols.head, cols.tail: _*)

todo.rdd.repartition(1).saveAsTextFile("hdfs:///user/admin/demos/customer_segmentation")

//Guardar:
sc.parallelize(Seq(model_4), 1).saveAsObjectFile("hdfs:///user/admin/demos/modelos/kmeans_4.model")