package com.poc

import java.io.File
import java.nio.file.Paths

import org.tensorflow.{DataType, Graph, Session, TensorFlow}
import com.poc.GraphHelper._


/**
 * Created by ndrizard on 3/15/17.
 */
object PocMnist extends App{
  println(s"Using Tensorflow version ${TensorFlow.version()}")


  // Infer score from a pretrained model located in modelDir as a .pb file
  // Hacky function specifically tailored for the MNIST model right now
  def predict(input: Array[Double], modelDir: String = "./src/main/resources/mnist_frozen"): Long = {
    val dim = input.size
    val g = new Graph()
    val b = new GraphBuilder(g)

    // Converting Input as Float
    val inputFloat = b.reshape(
      b.cast(b.constant("input", input), DataType.FLOAT),
      b.constant("shape", Array(1, dim)))

    val s = new Session(g)
    val inputTensor = s.runner().fetch(inputFloat.op().name()).run().get(0)

//    println(s"Input data type: ${inputTensor.dataType()}")
//    println(s"Input number of dimensions: ${inputTensor.numDimensions()}")
//    println(s"Input number of elements: ${inputTensor.numElements()}")

    // Loading the graph
    val graphDef = readAllBytesOrExit(Paths.get(modelDir, "frozen_model.pb"))

    // Inferring
    val gModel = new Graph()
    gModel.importGraphDef(graphDef)
    val session = new Session(gModel)
    val result = session.runner().feed("x", inputTensor).fetch("predictions").run().get(0)

//    println(s"Output number of dimensions ${result.numDimensions()}")

    // Output is of type INT64
    result.copyTo(Array.ofDim[Long](1))(0)
  }

  def batchPredict(input: Array[Array[Double]], modelDir: String = "./src/main/resources/mnist_frozen"): Array[Long] = {
    val numSamples = input.size
    val numFeatures = input.head.size

    val g = new Graph()
    val b = new GraphBuilder(g)

    // Converting Input as Float
    val inputFloat = b.reshape(
      b.cast(b.constant("input", input), DataType.FLOAT),
      b.constant("shape", Array(numSamples, numFeatures)))

    val s = new Session(g)
    val inputTensor = s.runner().fetch(inputFloat.op().name()).run().get(0)

    //    println(s"Input data type: ${inputTensor.dataType()}")
    //    println(s"Input number of dimensions: ${inputTensor.numDimensions()}")
    //    println(s"Input number of elements: ${inputTensor.numElements()}")

    // Loading the graph
    val graphDef = readAllBytesOrExit(Paths.get(modelDir, "frozen_model.pb"))

    // Inferring
    val gModel = new Graph()
    gModel.importGraphDef(graphDef)
    val session = new Session(gModel)
    val result = session.runner().feed("x", inputTensor).fetch("predictions").run().get(0)

    //    println(s"Output number of dimensions ${result.numDimensions()}")

    // Output is of type INT64
    result.copyTo(Array.ofDim[Long](numSamples))
  }

  // Manually Building the input
  val row1 = Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.72549021, 0.62352943, 0.59215689, 0.23529413, 0.14117648, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8705883, 0.99607849, 0.99607849, 0.99607849, 0.99607849, 0.9450981, 0.77647066, 0.77647066, 0.77647066, 0.77647066, 0.77647066, 0.77647066, 0.77647066, 0.77647066, 0.66666669, 0.20392159, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.26274511, 0.44705886, 0.28235295, 0.44705886, 0.63921571, 0.89019614, 0.99607849, 0.88235301, 0.99607849, 0.99607849, 0.99607849, 0.98039222, 0.89803928, 0.99607849, 0.99607849, 0.54901963, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06666667, 0.25882354, 0.05490196, 0.26274511, 0.26274511, 0.26274511, 0.23137257, 0.08235294, 0.92549026, 0.99607849, 0.41568631, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32549021, 0.99215692, 0.81960791, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08627451, 0.913725551, 0.32549021, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50588238, 0.99607849, 0.9333334, 0.17254902, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23137257, 0.97647065, 0.99607849, 0.24313727, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52156866, 0.99607849, 0.73333335, 0.01960784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03529412, 0.80392164, 0.97254908, 0.227451, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411768, 0.99607849, 0.71372551, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29411766, 0.98431379, 0.94117653, 0.22352943, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07450981, 0.86666673, 0.99607849, 0.65098041, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01176471, 0.7960785, 0.99607849, 0.8588236, 0.13725491, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.14901961, 0.99607849, 0.99607849, 0.3019608, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.12156864, 0.87843144, 0.99607849, 0.45098042, 0.00392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.52156866, 0.99607849, 0.99607849, 0.20392159, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2392157, 0.94901967, 0.99607849, 0.99607849, 0.20392159, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47450984, 0.99607849, 0.99607849, 0.8588236, 0.15686275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47450984, 0.99607849, 0.81176478, 0.07058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  val result = predict(row1, args(0))
  println(s"Predicted number is $result")

  // 2d arrays input
  val input2d = Array(row1, row1)
  val resultBatch = batchPredict(input2d, args(0))
  println(s"Predicted number in batch is ${resultBatch.mkString(" ")}")

}
