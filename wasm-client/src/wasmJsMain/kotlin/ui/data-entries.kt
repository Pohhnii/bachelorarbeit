package ui

import data.DataEntry
import data.FileInfo
import kotlinx.dom.appendElement
import org.w3c.dom.Element

fun Element.dataEntry(fileInfo: FileInfo, imageEntry: DataEntry, labelEntry: DataEntry) {
    appendElement("div") {
        className = "data-entry"

        appendElement("p") {
            innerHTML = "Label: <strong>${labelEntry.data[0]!!.toInt()}</strong>"
        }

        appendElement("p") {
            id = "prediction-${imageEntry.index}"
        }

        insertGrid(
            fileInfo.dimensions[fileInfo.dimensions.length - 2]!!.toInt(),
            fileInfo.dimensions[fileInfo.dimensions.length - 1]!!.toInt(),
            imageEntry
        )
    }
}


fun Element.insertGrid(rows: Int, cols: Int, dataEntry: DataEntry) {
    appendElement("table") {
        for (i in 0 until rows) {
            appendElement("tr") {
                for (j in 0 until cols) {
                    appendElement("td") {
                        val value = 255 - dataEntry.data[i * cols + j]!!.toInt()
                        val color = "rgb($value, $value, $value)"
                        setAttribute("style", "background-color: $color")
                    }
                }
            }
        }
    }
}