import clients.getBatch
import clients.getDatabaseInfo
import kotlinx.browser.document
import kotlinx.dom.appendElement
import kotlinx.dom.clear
import model.DatabaseInfo
import model.FileInfo
import ui.dataEntry
import ui.datasetDescription

fun main() {

    var databaseInfo: DatabaseInfo? = null

    document.body?.appendElement("div") {
        id = "root"

        appendElement("h1") {
            textContent = "MNIST Dataset Predictions"
        }

        appendElement("div") {
            getDatabaseInfo { dbInfo ->
                databaseInfo = dbInfo
                appendElement("div") {
                    datasetDescription("Training set", dbInfo.training)
                    datasetDescription("Testing set", dbInfo.testing)
                }
            }
        }

        appendElement("button") {
            id = "load-data-btn"
            textContent = "Load data"
            setAttribute("style", "margin-top: 20px;")
            addEventListener("click") {
                if (databaseInfo == null) {
                    return@addEventListener
                }

                getBatch("test", randomBatch(10, databaseInfo!!.testing.data)) { datasetBatch ->
                    val dataEntriesElement = document.getElementById("data-entries")
                    dataEntriesElement?.clear()
                    for (i in 0 until datasetBatch.images.length) {
                        dataEntriesElement?.dataEntry(
                            databaseInfo!!.testing.data,
                            datasetBatch.images[i]!!,
                            datasetBatch.labels[i]!!
                        )
                    }
                }
            }
        }

        appendElement("div") {
            id = "data-entries"
        }
    }
}

fun randomBatch(size: Int, fileInfo: FileInfo) = Array(size) { (0 until fileInfo.dimensions[0]!!.toInt()).random() }
