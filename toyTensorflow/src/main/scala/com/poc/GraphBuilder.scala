package com.poc

import java.io.IOException
import java.nio.charset.Charset
import java.nio.file.{Files, Path}
import java.util

import org.tensorflow._

/**
 * Created by ndrizard on 3/14/17.
 */
class GraphBuilder(g: Graph ) {

  def div(x: Output, y: Output): Output = {
    binaryOp("Div", x, y)
  }

  def sub(x: Output, y: Output): Output = {
    binaryOp("Sub", x, y)
  }

  def resizeBilinear(images: Output, size: Output): Output = {
    binaryOp("ResizeBilinear", images, size)
  }

  def expandDims(input: Output, dim: Output): Output = {
    binaryOp("ExpandDims", input, dim)
  }

  def cast(value: Output, dtype: DataType): Output = {
    g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build().output(0)
  }

  def decodeJpeg(contents: Output, channels: Long): Output = {
    g.opBuilder("DecodeJpeg", "DecodeJpeg")
      .addInput(contents)
      .setAttr("channels", channels)
      .build()
      .output(0)
  }

  def constant(name: String, value: Any): Output = {
    val t = Tensor.create(value)
    g.opBuilder("Const", name)
      .setAttr("dtype", t.dataType())
      .setAttr("value", t)
      .build()
      .output(0)
  }

  private def binaryOp(typeString: String, in1: Output, in2: Output): Output = {
    g.opBuilder(typeString, typeString).addInput(in1).addInput(in2).build().output(0)
  }
}

object GraphHelper {

  def constructAndExecuteGraphToNormalizeImage(imageBytes: Array[Byte]): Tensor = {
    val g = new Graph()
    val b = new GraphBuilder(g);
    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    val H = 224;
    val W = 224;
    val mean = 117f;
    val scale = 1f;

    // Since the graph is being constructed once per execution here, we can use a constant for the
    // input image. If the graph were to be re-used for multiple input images, a placeholder would
    // have been more appropriate.
    val input = b.constant("input", imageBytes);
    val output =
      b.div(
        b.sub(
          b.resizeBilinear(
            b.expandDims(
              b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
              b.constant("make_batch", 0)),
            b.constant("size", Array(H, W)
            )),
          b.constant("mean", mean)),
        b.constant("scale", scale));

    val s = new Session(g)
    s.runner().fetch(output.op().name()).run().get(0)
  }

  def executeInceptionGraph(graphDef: Array[Byte], image:Tensor): Array[Float] = {
    val g = new Graph()
    g.importGraphDef(graphDef);
    val s = new Session(g)
    val result = s.runner().feed("input", image).fetch("output").run().get(0)

    val rshape = result.shape();
    if (result.numDimensions() != 2 || rshape(0) != 1)
    {
      throw new RuntimeException(
        String.format(
          "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
          util.Arrays.toString(rshape)));
    }
    val nlabels = rshape(1).toInt;

    result.copyTo(Array.ofDim[Float](1, nlabels))(0);
  }

  def maxIndex(probabilities: Array[Float]): Int = {
    var best = 0;
    probabilities.zipWithIndex
      .foreach { case (p, i) =>
        if (p > probabilities(best)) best = i
      }
    best
  }

  def readAllBytesOrExit(path: Path): Array[Byte] = {
    try {
      Files.readAllBytes(path)
    } catch {
      case e: IOException =>
        System.err.println("Failed to read [" + path + "]: " + e.getMessage());
        System.exit(1)
        Array()
    }
  }

  def readAllLinesOrExit(path: Path): util.List[String] = {
    try {
      Files.readAllLines(path, Charset.forName("UTF-8"))
    } catch {
      case e: IOException =>
        System.err.println("Failed to read [" + path + "]: " + e.getMessage());
        System.exit(0);
        new util.ArrayList[String]
    }
  }
}
