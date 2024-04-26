import clients.DatasetBatch
import clients.getBatch
import clients.getDatabaseInfo
import clients.loadModel
import data.IDXFileInfo
import kotlinx.browser.document
import kotlinx.coroutines.MainScope
import kotlinx.coroutines.launch
import kotlinx.dom.clear
import kotlinx.html.*
import kotlinx.html.dom.append
import kotlinx.html.js.div
import kotlinx.html.js.onClickFunction
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.w3c.dom.HTMLElement
import org.w3c.dom.HTMLInputElement
import ui.dataEntry
import ui.datasetDescriptions
import kotlin.time.measureTime

suspend fun main() {
    val databaseInfo = getDatabaseInfo()
    val model = loadModel()
    var currentBatch: DatasetBatch? = null

    val body = document.body ?: error("No body")
    body.append.div {
        id = "root"
        h1 { +"MNIST Dataset Predictions" }
        div {
            classes = setOf("database-infos")
            div {
                classes = setOf("dataset-info")
                datasetDescriptions("Training set", databaseInfo.training)
            }
            div {
                classes = setOf("dataset-info")
                datasetDescriptions("Testing set", databaseInfo.testing)
            }
        }

        div {
            id = "actions"

            input(InputType.number) {
                id = "batch-size-input"
                value = "10"
                min = "1"
                max = "1000"
                step = "1"
            }

            button {
                +"Load data"
                id = "load-data-btn"
                onClickFunction = { _ ->
                    MainScope().launch {
                        val batchSize = (document.getElementById("batch-size-input") as HTMLInputElement).value.toInt()
                        currentBatch = getBatch("test", randomBatch(batchSize, databaseInfo.testing.data))
                        val dataEntriesElement = document.getElementById("data-entries")!! as HTMLElement
                        dataEntriesElement.clear()
                        for (i in currentBatch!!.images.indices) {
                            dataEntriesElement.dataEntry(
                                databaseInfo.testing.data,
                                currentBatch!!.images[i],
                                currentBatch!!.labels[i]
                            )
                        }
                    }
                }
            }

            button {
                +"Predict"
                id = "predict-btn"
                onClickFunction = { _ ->
                    MainScope().launch {
                        if (currentBatch == null) return@launch

                        val inputs = List(currentBatch!!.images.size) { i ->
                            mk.ndarray(
                                currentBatch!!.images[i].data.map { it.toDouble() },
                                databaseInfo.testing.data.dimensions[1],
                                databaseInfo.testing.data.dimensions[2]
                            )
                        }

                        var totalDuration = 0L
                        for (i in inputs.indices) {
                            lateinit var prediction: List<D2Array<Double>>
                            val duration = measureTime {
                                prediction = model.forward(inputs[i])
                            }

                            val predictedLabel = prediction[0].argMax()
                            val predictionElement =
                                document.getElementById("prediction-${currentBatch!!.images[i].index}")!!
                            predictionElement.innerHTML = "Prediction: <strong>$predictedLabel</strong>"

                            totalDuration += duration.inWholeNanoseconds
                        }

                        val avgDuration = totalDuration.toDouble() / inputs.size / 1e6
                        document.getElementById("measured-time")!!.innerHTML = "Avg Time to predict: ${avgDuration}ms"
                    }
                }
            }
        }

        p { id = "measured-time" }

        div { id = "data-entries" }
    }
}

fun randomBatch(size: Int, fileInfo: IDXFileInfo): Array<Int> {
    val randomIndices = mutableSetOf<Int>()
    while (randomIndices.size < size)
        randomIndices.add((0 until fileInfo.dimensions[0]).random())
    return randomIndices.toTypedArray()
}