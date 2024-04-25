package clients

import Model
import createLeNet5
import kotlinx.browser.window
import kotlinx.coroutines.await
import load
import kotlin.js.Promise

const val TARGET_MODEL = "Model.0.01lr-2000epochs-extended-lr-0.001lr-500epochs"

suspend fun loadModel(): Model {
    val model = createLeNet5()
    loadLayers(model).await<JsAny?>()
    return model
//    loadLayers(model).then {
//        callback(model)
//        null
//    }
}

private fun loadLayers(model: Model): Promise<JsAny?> {
    val layerDataRequests = JsArray<Promise<JsArray<out JsString>>>()
    for(i in model.layers.indices) {
        layerDataRequests[i] = loadModelLayer(model, i)
    }

    return Promise.all(layerDataRequests).then { jsLayerData ->
        model.load(jsLayerData)
        null
    }
}

private fun loadModelLayer(model: Model, layerIndex: Int): Promise<JsArray<out JsString>> {
    val layer = model.layers[layerIndex]
    val requests = JsArray<Promise<JsString>>()
    for(i in layer.data.indices) {
        requests[i] = loadLayerData(layerIndex, i)
    }

    return Promise.all(requests)
}

private fun loadLayerData(layerIndex: Int, dataIndex: Int): Promise<JsString> {
    return Promise { resolve, reject ->
        window.fetch("$REST_BASE_URL/models/$TARGET_MODEL/layer-$layerIndex/data-$dataIndex.csv").then { response ->
            if(!response.ok) {
                reject(response.statusText.toJsString())
                return@then null
            }

            response.text().then {
                resolve(it)
                null
            }
        }
    }
}