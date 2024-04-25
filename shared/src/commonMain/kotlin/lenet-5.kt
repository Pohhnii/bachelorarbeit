import layer.*


fun createLeNet5(): Model {
    val model = Model()
    model.add(ConvLayer(kernelSize = 5, featureMaps = 6, padding = 2)) // Result: 28x28x6
    model.add(ActivationLayer(activationFunction = ActivationFunctionType.SIGMOID)) // Result: 28x28x6
    model.add(PoolLayer(poolType = PoolType.AVG, kernelSize = 2, stride = 2)) // Result: 14x14x6
    model.add(ConvLayer(kernelSize = 5, featureMaps = 16, padding = 0)) // Result: 10x10x16
    model.add(ActivationLayer(activationFunction = ActivationFunctionType.SIGMOID)) // Result: 10x10x16
    model.add(PoolLayer(poolType = PoolType.AVG, kernelSize = 2, stride = 2)) // Result: 5x5x16
    model.add(FlattenLayer()) // Result: 1x400
    model.add(DenseLayer(inputs = 400, nodes = 120)) // Result: 1x120
    model.add(ActivationLayer(activationFunction = ActivationFunctionType.SIGMOID)) // Result: 1x120
    model.add(DenseLayer(inputs = 120, nodes = 84)) // Result: 1x84
    model.add(ActivationLayer(activationFunction = ActivationFunctionType.SIGMOID)) // Result: 1x84
    model.add(DenseLayer(inputs = 84, nodes = 10)) // Result: 1x10
    model.add(ActivationLayer(activationFunction = ActivationFunctionType.SOFTMAX)) // Result: 1x10

    return model
}