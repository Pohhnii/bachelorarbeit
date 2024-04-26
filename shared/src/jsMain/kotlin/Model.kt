import layer.ConvLayer
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set


fun Model.load(layerCsvData: List<List<String>>) {
    for (i in layers.indices) {
        val layerData = mutableListOf<D2Array<Double>>()

        for (j in layers[i].data.indices) {
            val doubleData = layerCsvData[i][j].split(",").map { it.trim().toDouble() }
            layerData.add(mk.ndarray(doubleData, layers[i].data[j].shape))
        }

        layers[i].data = layerData
    }
}