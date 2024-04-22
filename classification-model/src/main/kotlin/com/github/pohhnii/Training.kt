package com.github.pohhnii

import Model
import Trainer
import data.*
import layer.*
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.api.math.log
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import save
import java.io.File
import java.util.*
import kotlin.math.absoluteValue
import kotlin.math.max
import kotlin.math.min
import kotlin.time.measureTime


const val EPOCHS = 2000
const val BATCH_SIZE = 100
const val LEARNING_RATE = 0.01

@OptIn(ExperimentalUnsignedTypes::class)
fun main() {
    val databaseInfo = DATASETS.loadDatabaseInfo()
    val trainingDatasetInfo = databaseInfo.training
    val model = createModel()

    for (epoch in 0 until EPOCHS) {
        var loss: Double
        val duration = measureTime {
            val batch = randomDatasetBatch(trainingDatasetInfo, BATCH_SIZE)
//        val batch = fullDatasetBatch(trainingDatasetInfo)
            loss = batch.map { model.backpropagation(it.input, listOf(it.output), LEARNING_RATE) }.average()
        }
        log("Epoch: $epoch / $EPOCHS, Loss: $loss, Duration: $duration")
    }

    var correct = 0
    val testDatasetSize = databaseInfo.testing.labels.dimensions[0]
    for (i in 0 until testDatasetSize) {
        val label = DATASETS.TEST.labels.use { get(i) }.first().toInt()
        val image = convertImageIdx(databaseInfo.training.data, DATASETS.TEST.images.use { get(i) })
        val prediction = model.forward(image)[0].argMax()
        val isCorrect = label == prediction
        if (isCorrect) correct++

        log("Testing: $i / $testDatasetSize, Expected: $label, Predicted: $prediction, Correct: $isCorrect, Total: $correct / $i")
    }

    log("Result: $correct / $testDatasetSize")

    val modelDir = File("./Model")
    if (!modelDir.isDirectory) modelDir.mkdir()
    model.save(modelDir)
}

private fun printMatrix(mat: D2Array<Double>) {
    val cols = mat[0].size
    val rows = mat.size / cols

    for (i in 0 until rows) {
        for (j in 0 until cols) {
            val isBlack = mat[i][j] > 0.5
            print((if (isBlack) "#" else " "))
        }
        println()
    }
}

private fun createModel(): Model {
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

//    model.add(ConvLayer(kernelSize = 5, featureMaps = 5, padding = 2)) // Result: 28x28x2
//    model.add(ActivationLayer(activationFunction = ActivationFunctionType.LEAKY_RELU))
//    model.add(PoolLayer(poolType = PoolType.MAX, kernelSize = 2, stride = 2)) // Result: 14x14x2
//    model.add(ConvLayer(kernelSize = 5, featureMaps = 10, padding = 0)) // Result: 10x10x10
//    model.add(ActivationLayer(activationFunction = ActivationFunctionType.LEAKY_RELU)) // Result: 10x10x10
//    model.add(PoolLayer(poolType = PoolType.MAX, kernelSize = 2, stride = 2)) // Result: 5x5x10
//    model.add(FlattenLayer())
//    model.add(DenseLayer(inputs = 250, nodes = 100)) // Result:
//    model.add(ActivationLayer(activationFunction = ActivationFunctionType.TANH))
//    model.add(DenseLayer(inputs = 100, nodes = 10))
//    model.add(ActivationLayer(activationFunction = ActivationFunctionType.SOFTMAX))
    return model
}

private data class TrainingData(val input: D2Array<Double>, val output: D2Array<Double>)

@OptIn(ExperimentalUnsignedTypes::class)
private fun randomDatasetBatch(datasetInfo: DatasetInfo, size: Int): List<TrainingData> {
    val datasetSize = datasetInfo.labels.dimensions[0]
    val randomIndices = List(size) { (0 until datasetSize).random() }

    val images = DATASETS.TRAINING.images.use { get(randomIndices) }
    val labels = DATASETS.TRAINING.labels.use { get(randomIndices) }

    return images.mapIndexed { index, image ->
        TrainingData(
            input = convertImageIdx(datasetInfo.data, image.data),
            output = convertLabelIdx(labels[index].data)
        )
    }
}

@OptIn(ExperimentalUnsignedTypes::class)
private fun fullDatasetBatch(datasetInfo: DatasetInfo): List<TrainingData> {
    val images = DATASETS.TRAINING.images.use { getAll() }
    val labels = DATASETS.TRAINING.labels.use { getAll() }

    return images.mapIndexed { index, image ->
        TrainingData(
            input = convertImageIdx(datasetInfo.data, image.data),
            output = convertLabelIdx(labels[index].data)
        )
    }
}

@OptIn(ExperimentalUnsignedTypes::class)
private fun convertImageIdx(imageInfo: IDXFileInfo, data: UByteArray): D2Array<Double> {
    val doubleValues = data.map { it.toDouble() }
    return mk.ndarray(
        doubleValues,
        imageInfo.dimensions[imageInfo.dimensions.size - 2],
        imageInfo.dimensions[imageInfo.dimensions.size - 1]
    )
}

@OptIn(ExperimentalUnsignedTypes::class)
private fun convertLabelIdx(data: UByteArray): D2Array<Double> {
    val intValue = data.first().toInt()
    return mk.ndarray(List(10) { index -> if (index == intValue) 1.0 else 0.0 }, 1, 10)
}

private fun generateErrorFunction(batch: List<TrainingData>): (Model) -> Double {
    return { model: Model ->
        var totalError = 0.0
        for (data in batch) {
            val input = data.input
            val actualOutput = model.forward(input)[0]
//            val loss = (data.output - actualOutput).map { abs(it) }.sum()
//            val loss = crossEntropyLoss(data.output, actualOutput)
//            val loss = binaryCrossEntropyLoss(data.output, actualOutput)
//            val loss = (data.output - actualOutput).sum().absoluteValue
            val loss = mse(data.output, actualOutput)
            totalError += loss
        }
        totalError / batch.size
    }
}

private fun crossEntropyLoss(expected: D2Array<Double>, predicted: D2Array<Double>): Double {
    val clippedPrediction = predicted.map { max(it, 1e-10) }
    return -(expected * mk.math.log(clippedPrediction)).sum()
}

private fun mse(expected: D2Array<Double>, predicted: D2Array<Double>): Double {
    return (expected - predicted).map { it * it }.average()
}

private fun binaryCrossEntropyLoss(expected: D2Array<Double>, predicted: D2Array<Double>): Double {
    val p = predicted.map { max(it, 1e-10) }.map { min(it, 1 - 1e-10) }
    return -1.0 / expected.size * ((expected * mk.math.log(p)) + (expected.map { 1 - it } * mk.math.log(p.map { 1 - it }))).sum()
}