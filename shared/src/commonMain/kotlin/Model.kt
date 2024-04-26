import layer.Layer
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.average
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import kotlin.math.abs


class Model {

    val layers = mutableListOf<Layer>()

    fun add(layer: Layer) {
        layers.add(layer)
    }

    fun forward(input: D2Array<Double>): List<D2Array<Double>> {
        var result = listOf(input)

        for (layer in layers) {
            result = layer.forward(result)
        }

        return result
    }

    fun copy(): Model {
        val model = Model()

        for (layer in layers) {
            model.add(layer.copy())
        }

        return model
    }

    fun mutate(mutationRate: Double = 0.1, sigma: Double = 0.1) {
        for (layer in layers) {
            layer.mutate(mutationRate, sigma)
        }
    }

    fun crossover(other: Model) {
        for (i in layers.indices) layers[i].crossover(other.layers[i])
    }

    fun backpropagation(input: D2Array<Double>, expected: List<D2Array<Double>>, learnRate: Double = 0.1): Double {
        val layerInputs = mutableListOf(listOf(input))
        for (layer in layers) {
            layerInputs.add(layer.forward(layerInputs.last()))
        }

        val result = layerInputs.last()
        val errors = expected.mapIndexed { index, ex -> ex - result[index] }

        val layerErrors = mutableListOf(errors)
        for (i in layers.indices.reversed()) {
            layerErrors.add(layers[i].backpropagation(layerErrors.last(), layerInputs[i], layerInputs[i + 1], learnRate))
        }

        return errors.map { err -> err.map { abs(it) } }.reduce { acc, err -> err + acc }.average()
    }

}