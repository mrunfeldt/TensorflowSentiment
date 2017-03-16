package com.poc

import java.io.File
import java.nio.file.Paths

import org.tensorflow.TensorFlow
import com.poc.GraphHelper._

/**
 * Created by ndrizard on 3/14/17.
 */
object Poc extends App{

  override def main(args: Array[String]): Unit = {
    println(s"Using Tensorflow version ${TensorFlow.version()}")
    println(new File(".").getAbsolutePath())

    val modelDir = args(0)
    val imageFile = args(1)

    val graphDef = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"))
    val labels = readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"))
    val imageBytes = readAllBytesOrExit(Paths.get(imageFile))

    val image = constructAndExecuteGraphToNormalizeImage(imageBytes)
    println(s"Input data type: ${image.dataType()}")
    println(s"Input number of dimensions: ${image.numDimensions()}")
    println(s"Input number of elements: ${image.numElements()}")
    val labelProbabilities = executeInceptionGraph(graphDef, image)
    val bestLabelIdx = maxIndex(labelProbabilities)
    println(s"BEST MATCH: ${labels.get(bestLabelIdx)} (${labelProbabilities(bestLabelIdx) * 100f} likely)")
  }
}
