package clients

import data.DataEntry
import data.DatabaseInfo
import kotlinx.browser.window
import kotlinx.coroutines.await

suspend fun getDatabaseInfo(): DatabaseInfo {
    val response = window.fetch("$REST_BASE_URL/dataset/descriptions").await()
    val result = response.json().await()
    return result as DatabaseInfo
}


external interface DatasetBatch {
    val labels: Array<DataEntry>
    val images: Array<DataEntry>
}

suspend fun getBatch(dataset: String, batch: Array<Int>): DatasetBatch {
    val response = window.fetch("$REST_BASE_URL/dataset/$dataset/batch?ids=${batch.joinToString(",")}").await()
    val result = response.json().await()
    return result  as DatasetBatch
}