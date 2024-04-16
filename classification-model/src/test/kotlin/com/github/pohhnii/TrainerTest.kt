package com.github.pohhnii

import com.github.pohhnii.layer.ActivationFunctionType
import com.github.pohhnii.layer.ActivationLayer
import com.github.pohhnii.layer.DenseLayer
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.RepeatedTest
import kotlin.math.abs
import kotlin.test.assertTrue

class TrainerTest {


    @RepeatedTest(100)
    @Disabled
    fun xorModel() {
        val initModel = Model()
        initModel.add(DenseLayer(2, 8))
        initModel.add(ActivationLayer(ActivationFunctionType.SIGMOID))
        initModel.add(DenseLayer(8, 1))
        initModel.add(ActivationLayer(ActivationFunctionType.SIGMOID))

        val trainer = Trainer(initModel, 5, 3)

        val inputDataset = listOf(
            mk.ndarray(doubleArrayOf(0.0, 0.0), 1, 2),
            mk.ndarray(doubleArrayOf(0.0, 1.0), 1, 2),
            mk.ndarray(doubleArrayOf(1.0, 0.0), 1, 2),
            mk.ndarray(doubleArrayOf(1.0, 1.0), 1, 2)
        )

        val outputDataset = listOf(
            mk.ndarray(doubleArrayOf(0.0), 1, 1),
            mk.ndarray(doubleArrayOf(1.0), 1, 1),
            mk.ndarray(doubleArrayOf(1.0), 1, 1),
            mk.ndarray(doubleArrayOf(0.0), 1, 1)
        )

        val errorFunction = { model: Model ->
            var totalError = 0.0
            for (i in inputDataset.indices) {
                val output = model.forward(inputDataset[i])
                totalError += (output[0] - outputDataset[i]).map { abs(it) }.sum()
            }
            totalError
        }

        for (gen in 0..1000) {
            trainer.errors(errorFunction)
            trainer.nextGen(mutationRate = 0.3, sigma = 2.0)
        }

        val bestModel = trainer.bestIndividual.model
        val outputs = mutableListOf<Double>()
        for (i in inputDataset.indices) {
            outputs.add(bestModel.forward(inputDataset[i])[0][0][0])
        }

        val error = errorFunction(bestModel)
        println(outputs)
        println(error)
        assertTrue(error < 0.1)
    }

}