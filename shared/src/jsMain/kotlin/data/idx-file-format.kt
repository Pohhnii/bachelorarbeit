package data


typealias Dimensions = Array<Int>

external interface IDXFileInfo {
    val magicNumber: Int
    val dataType: String?
    val dimensions: Dimensions
}

external interface DataEntry {
    val index: Int
    val data: Array<Int>
}