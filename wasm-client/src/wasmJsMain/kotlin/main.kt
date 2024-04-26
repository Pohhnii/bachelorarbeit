import clients.getBatch
import clients.getDatabaseInfo
import clients.loadModel
import data.DataEntry
import data.DatabaseInfo
import data.DatasetBatchResponse
import data.FileInfo
import kotlinx.browser.document
import kotlinx.coroutines.await
import kotlinx.dom.appendElement
import kotlinx.dom.clear
import org.jetbrains.kotlinx.multik.api.math.argMax
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import ui.dataEntry
import ui.datasetDescription
import kotlin.time.measureTime

suspend fun main() {

    val databaseInfo: DatabaseInfo = getDatabaseInfo().await()
    val model = loadModel()
    var currentBatch: DatasetBatchResponse? = null

    document.body?.appendElement("div") {
        id = "root"

        appendElement("h1") {
            textContent = "MNIST Dataset Predictions"
        }

        appendElement("div") {
            appendElement("div") {
                datasetDescription("Training set", databaseInfo.training)
                datasetDescription("Testing set", databaseInfo.testing)
            }
        }

        appendElement("button") {
            id = "load-data-btn"
            textContent = "Load data"
            setAttribute("style", "margin-top: 20px;")
            addEventListener("click") {

                getBatch("test", randomBatch(10, databaseInfo.testing.data)).then { datasetBatch ->
                    currentBatch = datasetBatch
                    val dataEntriesElement = document.getElementById("data-entries")
                    dataEntriesElement?.clear()
                    for (i in 0 until datasetBatch.images.length) {
                        dataEntriesElement?.dataEntry(
                            databaseInfo.testing.data,
                            datasetBatch.images[i]!!,
                            datasetBatch.labels[i]!!
                        )
                    }
                    null
                }
            }
        }

        appendElement("button") {
            textContent = "Predict"

            addEventListener("click") {
                if (currentBatch == null) return@addEventListener

                val inputs = MutableList(currentBatch!!.images.length) { i ->
                    convertDataEntry(
                        currentBatch!!.images[i]!!,
                        databaseInfo.testing.data.dimensions[1]!!.toInt(),
                        databaseInfo.testing.data.dimensions[2]!!.toInt()
                    )
                }

                var totalDuration = 0L
                for (i in 0 until inputs.size) {
                    lateinit var prediction: List<D2Array<Double>>
                    val duration = measureTime {
                        prediction = model.forward(inputs[i])
                    }

                    val predictedLabel = prediction[0].argMax()
                    val predictionElement = document.getElementById("prediction-${currentBatch!!.images[i]!!.index}")!!
                    predictionElement.innerHTML = "Prediction: <strong>$predictedLabel</strong>"

                    totalDuration += duration.inWholeNanoseconds
                }

                val averageDuration: Double = totalDuration.toDouble() / inputs.size / 1e6
                document.getElementById("measured-time")!!.innerHTML = "Avg Time to predict: ${averageDuration}ms"
            }
        }

        appendElement("p") {
            id = "measured-time"
        }

        appendElement("div") {
            id = "data-entries"
        }
    }
}

fun randomBatch(size: Int, fileInfo: FileInfo): Array<Int> {
    val randomIndices = mutableSetOf<Int>()
    while (randomIndices.size < size)
        randomIndices.add((0 until fileInfo.dimensions[0]!!.toInt()).random())
    return randomIndices.toTypedArray()
}

fun convertDataEntry(dataEntry: DataEntry, rows: Int, cols: Int): D2Array<Double> {
    val doubleData = mutableListOf<Double>()
    for (i in 0 until dataEntry.data.length)
        doubleData.add(i, dataEntry.data[i]!!.toDouble())
    return mk.ndarray(doubleData, rows, cols)
}