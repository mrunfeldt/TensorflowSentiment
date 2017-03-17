package com.poc

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

/**
 * Created by ndrizard on 3/17/17.
 *
 * Need add to the JVM args for the tensorflow binding: -Djava.library.path=../jni
 */
object PocMnistSpark extends App {

  val sparkConf = new SparkConf()
    .setAppName("mnist")
    .setMaster("local")

  implicit val sparkContext = new SparkContext(sparkConf)

  val rdd = sparkContext.textFile("./src/main/resources/mnist_test.data")
    .zipWithIndex
    .map{ case (line, index) => index -> line.split(" ").map(c => c.toDouble)}

  println(s" Number of input: ${rdd.count}")

  val predictions = rdd.mapPartitions(iter =>
    iter.map{ case (index, input) => index -> PocMnist.predict(input)}
  )

  println(s"Computed ${predictions.count} predictions")
  println(predictions.take(10).mkString("\n"))


}

