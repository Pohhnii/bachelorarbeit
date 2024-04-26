package data

external interface DatabaseInfo : JsAny {
    val training: DatasetInfo
    val testing: DatasetInfo
}

external interface DatasetInfo : JsAny {
    val labels: FileInfo
    val data: FileInfo
}

external interface FileInfo : JsAny {
    val magicNumber: Int
    val dataType: String?
    val dimensions: JsArray<JsNumber>
}

external interface DatasetBatchResponse: JsAny {
    val labels: JsArray<DataEntry>
    val images: JsArray<DataEntry>
}

external interface DataEntry: JsAny {
    val index: Int
    val data: JsArray<JsNumber>
}