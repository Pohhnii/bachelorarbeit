package clients

import Model
import createLeNet5Js
import kotlinx.browser.window
import kotlinx.coroutines.await
import load

const val TARGET_MODEL = "Model.0.01lr-2000epochs-extended-lr-0.001lr-500epochs"

suspend fun loadModel(): Model {
    val model = createLeNet5Js()

    val layerCsvData = MutableList<MutableList<String>>(model.layers.size) { mutableListOf() }
    for (i in model.layers.indices) {
        for (j in model.layers[i].data.indices) {
            val response = window.fetch("$REST_BASE_URL/models/$TARGET_MODEL/layer-$i/data-$j.csv").await()
            layerCsvData[i].add(response.text().await())
        }
    }

    model.load(layerCsvData)

    return model
}