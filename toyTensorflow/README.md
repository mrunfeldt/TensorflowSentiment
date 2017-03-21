Toy example to load a tensorflow model in Scala using the Java API. It contains 3 runnables classes.

# Poc

Load the pre-trained tensorflow model `inception` and apply it on the image path provided (some are already present in the resoures)

### Usage

```
./gradlew Poc -PappArgs="['./src/main/resources/inception5h', './src/main/resources/cheeseburger_test.jpeg']"
```

NB: you can also try it with the other images provided in the resources, ie ./src/main/resources/convertible_test.jpeg or ./src/main/resources/house_test.jpeg

# PocMnist

Load the pre-trained tensorflow model to classify MNIST digit. It applies it on an input already hard coded.

### Usage

```
./gradlew PocMnist -PappArgs="['./src/main/resources/mnist_frozen']"
```
# PocMnistSpark

It builds an `RDD[Array[Double]` where every `Array[Double]` is a MNIST input image. It then loads the MNIST model locally in Spark and compute inference for every input, it does so locally on the partition either one image at a time or by batch.

### Usage
Run within Intellij idea the scala object specifying as JVM args: `-Djava.library.path=../jni`
