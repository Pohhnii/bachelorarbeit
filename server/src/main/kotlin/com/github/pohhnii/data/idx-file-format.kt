package com.github.pohhnii.data

import com.github.pohhnii.model.DataEntry
import com.github.pohhnii.model.DataType
import com.github.pohhnii.model.IDXFileInfo
import java.io.DataInputStream


typealias IDXFileStream = DataInputStream

fun <T> DatasetPath.use(consumer: IDXFileStream.() -> T): T {
    ClassLoader.getSystemResourceAsStream(this)!!.use { inputStream ->
        DataInputStream(inputStream).use {
            return it.consumer()
        }
    }
}

private fun IDXFileStream.loadIDXFileInfo(): IDXFileInfo {
    val magicNumber = this.readInt()
    // MSB 3rd byte is the data type
    val dataTypeIndex = magicNumber.byteAt(1)
    val dataType = DataType.entries.find { it.magicNumber == dataTypeIndex }
    // MSB 4th byte is the number of dimensions
    val numberOfDimensions = magicNumber.byteAt(0)

    val dimensions = (0 until numberOfDimensions).map { this.readInt() }

    return IDXFileInfo(magicNumber, dataType, dimensions)
}

fun IDXFileStream.getInfo(): IDXFileInfo {
    val info = loadIDXFileInfo()
    this.close()
    return info
}

private fun Int.byteAt(byteIndex: Int): Int {
    return this shr (8 * byteIndex) and 0xFF
}

fun IDXFileInfo.dataSize(): Int {
    var size = 1
    for (i in 1 until dimensions.size) {
        size *= dimensions[i]
    }

    return size
}


@OptIn(ExperimentalUnsignedTypes::class)
fun IDXFileStream.get(indices: List<Int>): List<DataEntry> {
    val info = loadIDXFileInfo()
    val dataSize = info.dataSize()
    val sortedIndices = indices.sorted()

    val entries = mutableListOf<DataEntry>()

    var lastIndex = 0
    for (index in sortedIndices) {
        val offset = dataSize * (index - lastIndex)
        lastIndex = index
        val data = ByteArray(dataSize)
        this.skipBytes(offset)
        this.readFully(data)
        entries.add(DataEntry(index, data.toUByteArray()))
    }

    this.close()

    return entries
}

@OptIn(ExperimentalUnsignedTypes::class)
fun IDXFileStream.get(index: Int): UByteArray {
    val info = this.loadIDXFileInfo()
    val dataSize = info.dataSize()
    val offset = dataSize * index
    val data = ByteArray(dataSize)
    this.skipBytes(offset)
    this.readFully(data)
    this.close()
    return data.toUByteArray()
}