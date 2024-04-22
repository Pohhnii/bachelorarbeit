package com.github.pohhnii

import Model
import layer.*
import load
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.junit.jupiter.api.io.TempDir
import save
import java.io.File
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class ModelTest {

    @field:TempDir
    lateinit var tempDir: File


    @Test
    fun leNetShapesDoFit() {
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

        val input = mk.zeros<Double>(28, 28)
        val result = model.forward(input)

        assertEquals(1, result.size)
        assertEquals(1, result[0].shape[0])
        assertEquals(10, result[0].shape[1])

        val sum = result[0].sum()
        assertEquals(1.0, sum, 0.001)
    }

    @Test
    fun saveAndLoad() {
        val createModel = {
            val model = Model()
            model.add(DenseLayer(2, 5))
            model.add(ActivationLayer(ActivationFunctionType.SIGMOID))
            model.add(DenseLayer(5, 1))
            model.add(ActivationLayer(ActivationFunctionType.SIGMOID))
            model
        }

        val input = mk.ndarray(doubleArrayOf(1.0, 2.0), 1, 2)

        val model1 = createModel()
        val randomOutput = model1.forward(input)
        model1.save(tempDir)

        val model2 = createModel()
        model2.load(tempDir)
        val output = model2.forward(input)

        assertEquals(randomOutput, output)
    }

    @Test
    fun copy() {
        val createModel = {
            val model = Model()
            model.add(DenseLayer(2, 5))
            model.add(ActivationLayer(ActivationFunctionType.SIGMOID))
            model.add(DenseLayer(5, 1))
            model.add(ActivationLayer(ActivationFunctionType.SIGMOID))
            model
        }

        val input = mk.ndarray(doubleArrayOf(1.0, 2.0), 1, 2)

        val model = createModel()
        val modelCopy = model.copy()

        val output = model.forward(input)
        val outputCopy = modelCopy.forward(input)

        assertEquals(output, outputCopy)

        modelCopy.mutate()
        val output2 = model.forward(input)
        assertEquals(output, output2)
    }

    @Test
    fun crossoverAndMutate() {
        val createModel = {
            val model = Model()
            model.add(DenseLayer(2, 5))
            model.add(ActivationLayer(ActivationFunctionType.SIGMOID))
            model.add(DenseLayer(5, 1))
            model.add(ActivationLayer(ActivationFunctionType.SIGMOID))
            model
        }

        val input = mk.ndarray(doubleArrayOf(1.0, 2.0), 1, 2)

        val model1 = createModel()
        val model2 = createModel()

        val crossoverModel = model1.copy()
        crossoverModel.crossover(model2)

        val outputModel1 = model1.forward(input)
        val outputModel2 = model2.forward(input)
        val outputCrossover = crossoverModel.forward(input)

        assertNotEquals(outputModel1, outputModel2)
        assertNotEquals(outputModel1, outputCrossover)
        assertNotEquals(outputModel2, outputCrossover)

        val mutated = crossoverModel.copy()
        mutated.mutate(1.0)
        val outputMutated = mutated.forward(input)

        assertNotEquals(outputCrossover, outputMutated)
    }

    @Test
    fun xorModelBackpropagation() {
        val model = Model()
        model.add(DenseLayer(2, 8))
        model.add(ActivationLayer(ActivationFunctionType.SIGMOID))
        model.add(DenseLayer(8, 2))
        model.add(ActivationLayer(ActivationFunctionType.SOFTMAX))

        val inputDataset = listOf(
            mk.ndarray(doubleArrayOf(0.0, 0.0), 1, 2),
            mk.ndarray(doubleArrayOf(0.0, 1.0), 1, 2),
            mk.ndarray(doubleArrayOf(1.0, 0.0), 1, 2),
            mk.ndarray(doubleArrayOf(1.0, 1.0), 1, 2)
        )

        val outputDataset = listOf(
            mk.ndarray(doubleArrayOf(0.0, 1.0), 1, 2),
            mk.ndarray(doubleArrayOf(1.0, 0.0), 1, 2),
            mk.ndarray(doubleArrayOf(1.0, 0.0), 1, 2),
            mk.ndarray(doubleArrayOf(0.0, 1.0), 1, 2)
        )

        val epochs = 25000
        for (epoch in 0..epochs) {
            for (i in inputDataset.indices.shuffled()) {
                val loss = model.backpropagation(inputDataset[i], listOf(outputDataset[i]), 0.1)
                println("Epoch: $epoch / $epochs, loss: $loss")
            }
        }
        for (i in inputDataset.indices) {
            val output = model.forward(inputDataset[i])[0]
            println(output)

            assertEquals(outputDataset[i].argMax(), output.argMax())
        }
    }

    @Test
    fun convTest() {
        val input1 = mk.ndarray(
            doubleArrayOf(
                0.0, 1.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 1.0, 0.0
            ), 3, 3
        )
        val input2 = mk.ndarray(
            doubleArrayOf(
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0
            ), 3, 3
        )

        val expected1 = mk.ndarray(doubleArrayOf(1.0), 1, 1)
        val expected2 = mk.ndarray(doubleArrayOf(0.0), 1, 1)

        val model = Model()
        model.add(ConvLayer(3, 1, 1))
        model.add(ActivationLayer(ActivationFunctionType.SIGMOID))
        model.add(FlattenLayer())
        model.add(DenseLayer(9, 1))
        model.add(ActivationLayer(ActivationFunctionType.SIGMOID))

        val epochs = 5000
        for (epoch in 0..epochs) {
            val loss1 = model.backpropagation(input1, listOf(expected1), 0.3)
            val loss2 = model.backpropagation(input2, listOf(expected2), 0.3)
            val loss = (loss1 + loss2) / 2
            println("Epoch: $epoch / $epochs, loss: $loss")
        }

        val prediction1 = model.forward(input1)[0]
        val prediction2 = model.forward(input2)[0]

        println("prediction1: $prediction1, prediction2: $prediction2")

        assertEquals(prediction1.data[0], expected1.data[0], 0.1)
        assertEquals(prediction2.data[0], expected2.data[0], 0.1)
    }
}