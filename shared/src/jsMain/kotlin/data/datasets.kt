package data

external interface DatasetInfo {
    val labels: IDXFileInfo
    val data: IDXFileInfo
}

external interface DatabaseInfo {
    val training: DatasetInfo
    val testing: DatasetInfo
}