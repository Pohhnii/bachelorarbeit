package com.github.pohhnii.layer

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

abstract class Layer {
    abstract var data: List<D2Array<Double>>

    abstract fun forward(inputs: List<D2Array<Double>>): List<D2Array<Double>>

    abstract fun copy(): Layer
}