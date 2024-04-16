package clients

import kotlinx.browser.window
import model.DatabaseInfo
import model.DatasetBatchResponse

fun getDatabaseInfo(callback: (DatabaseInfo) -> Unit) {
    window.fetch("$REST_BASE_URL/dataset/descriptions")
        .then { response ->
            response.json().then {
                val datasetInfo = it as DatabaseInfo
                callback(datasetInfo)
                null
            }
        }
}

fun getBatch(dataset: String, batch: Array<Int>, callback: (DatasetBatchResponse) -> Unit) {
    window.fetch("$REST_BASE_URL/dataset/$dataset/batch?ids=${batch.joinToString(",")}")
        .then { response ->
            if (!response.ok) {
                return@then null
            }

            response.json().then {
                val datasetBatch = it as DatasetBatchResponse
                callback(datasetBatch)
                null
            }
        }
}