package com.github.pohhnii

import com.github.pohhnii.layer.*
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals

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

}