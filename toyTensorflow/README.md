Toy example to load a tensorflow model in Scala using the Java API.

# How to use it?
You can use gradle to run it, providing the model and the image to classify as argument.

```
./gradlew run -PappArgs="['./src/main/resources/inception5h', './src/main/resources/cheeseburger_test.jpeg']"
```

NB: you can also try it with the other images provided in the resources, ie ./src/main/resources/convertible_test.jpeg or ./src/main/resources/house_test.jpeg

