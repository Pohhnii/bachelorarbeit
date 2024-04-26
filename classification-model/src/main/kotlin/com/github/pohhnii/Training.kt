package com.github.pohhnii

import createLeNet5
import data.*
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import save
import java.io.File
import kotlin.time.measureTime


const val EPOCHS = 2000
const val BATCH_SIZE = 100
const val LEARNING_RATE = 0.01

@OptIn(ExperimentalUnsignedTypes::class)
fun main() {
    val databaseInfo = DATASETS.loadDatabaseInfo()
    val trainingDatasetInfo = databaseInfo.training
    val model = createLeNet5()

    for (epoch in 0 until EPOCHS) {
        var loss: Double
        val duration = measureTime {
            val batch = randomDatasetBatch(trainingDatasetInfo, BATCH_SIZE)
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