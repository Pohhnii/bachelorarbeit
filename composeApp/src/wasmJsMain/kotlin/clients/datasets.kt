package clients

import kotlinx.browser.window
import kotlinx.coroutines.await
import model.DatabaseInfo
import model.DatasetBatchResponse
import kotlin.js.Promise

fun getDatabaseInfo(): Promise<DatabaseInfo> = Promise { resolve, _ ->
    window.fetch("$REST_BASE_URL/dataset/descriptions")
        .then { response ->
            response.json().then {
                val datasetInfo = it as DatabaseInfo
                resolve(datasetInfo)
                null
            }
        }
}


fun getBatch(dataset: String, batch: Array<Int>): Promise<DatasetBatchResponse> = Promise { resolve, reject ->
    window.fetch("$REST_BASE_URL/dataset/$dataset/batch?ids=${batch.joinToString(",")}")
        .then { response ->
            if (!response.ok) {
                reject(response.statusText.toJsString())
                return@then null
            }

            response.json().then {
                val datasetBatch = it as DatasetBatchResponse
                resolve(datasetBatch)
                null
            }
        }
}
