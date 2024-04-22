package data

import kotlinx.serialization.Serializable

@Serializable
enum class DataType(val magicNumber: Int) {
    UNSIGNED_BYTE(0x08),
    SIGNED_BYTE(0x09),
    SHORT(0x0B),
    INT(0x0C),
    FLOAT(0x0D),
    DOUBLE(0x0E)
}

typealias Dimensions = List<Int>

@Serializable
data class IDXFileInfo(
    val magicNumber: Int,
    val dataType: DataType?,
    val dimensions: Dimensions,
)

@Serializable
data class DataEntry @OptIn(ExperimentalUnsignedTypes::class) constructor(
    val index: Int,
    val data: UByteArray
)