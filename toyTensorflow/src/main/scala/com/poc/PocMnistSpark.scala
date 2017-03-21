package com.poc

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * Created by ndrizard on 3/17/17.
 *
 * Need add to the JVM args for the tensorflow binding: -Djava.library.path=../jni
 */
object PocMnistSpark extends App {

  // Helper to time Spark work
  val timing = new StringBuffer
  def timed[T](label: String, code: => T): T = {
    val start = System.currentTimeMillis()
    val result = code
    val stop = System.currentTimeMillis()
    timing.append(s"Processing $label took ${stop - start} ms.\n")
    result
  }

  val sparkConf = new SparkConf()
    .setAppName("mnist")
    .setMaster("local")

  implicit val sparkContext = new SparkContext(sparkConf)

  val rdd = sparkContext.textFile("./src/main/resources/mnist_test.data")
    .zipWithIndex
    .map{ case (line, index) => index -> line.split(" ").map(c => c.toDouble)}

  println(s" Number of elements in rdd: ${rdd.count}")
  println(s" Number of partitions: ${rdd.partitions.size}")

  val predictions = timed("Predicting one by one in current partition", rdd.mapPartitions(iter =>
    iter.map{ case (index, input) => index -> PocMnist.predict(input)}).collect()
  )

  val rddBatch = sparkContext.textFile("./src/main/resources/mnist_test.data")
    .map(_.split(" ").map(_.toDouble))

  val predictionsBatch = timed("Predicting by batch in current partition", rddBatch.mapPartitions[Long](iter => {
    // TODO: keeping track of the index (row number) for every prediction
    PocMnist.batchPredict(iter.toArray).toIterator
  }).collect())

  println(timing)
  println(s"Computed ${predictions.size} predictions")
  println(predictions.take(10).mkString("\n"))
  println(s"Computed ${predictionsBatch.size} predictions")
  println(predictionsBatch.take(10).mkString("\n"))

}

