import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array


fun Model.load(jsLayerData: JsArray<out JsArray<out JsString>>) {
    for (i in layers.indices) {
        val layerData = mutableListOf<D2Array<Double>>()

        for (j in layers[i].data.indices) {
            val csvData = jsLayerData[i]!![j].toString()
            layerData.add(mk.ndarray(csvData.split(",", ", ").map { it.toDouble() }, layers[i].data[j].shape))
        }

        layers[i].data = layerData
    }
}