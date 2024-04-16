package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

abstract class Layer {
    abstract var data: List<D2Array<Double>>

    abstract fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>>

    abstract fun copy(): Layer

    fun mutate(mutationRate: Double, sigma: Double) {
        for (mat in data) {
            for (i in mat.data.indices) {
                if (Random.nextDouble() < mutationRate)
                    mat.data[i] += normallyDistributedRandom(0.0, sigma)
            }
        }
    }

    private fun normallyDistributedRandom(mean: Double, stdDev: Double): Double {
        val u1 = Random.nextDouble()
        val u2 = Random.nextDouble()
        val transformed = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
        return stdDev * transformed + mean
    }

    fun crossover(other: Layer) {
        for (i in data.indices) {
            val mat = data[i]
            val otherMat = other.data[i]
            for (j in mat.data.indices) {
                mat.data[j] = if (Random.nextDouble() <= 0.5) mat.data[j] else otherMat.data[j]
            }
        }
    }
}