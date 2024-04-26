package ui

import data.DataEntry
import data.IDXFileInfo
import kotlinx.html.*
import kotlinx.html.dom.append
import org.w3c.dom.Node

fun Node.dataEntry(fileInfo: IDXFileInfo, imageEntry: DataEntry, labelEntry: DataEntry) {
    this.append {
        div {
            classes = setOf("data-entry")

            p {
                +"Label: "
                strong { +"${labelEntry.data[0]}" }
            }

            p { id = "prediction-${imageEntry.index}" }

            insertGrid(
                fileInfo.dimensions[fileInfo.dimensions.size - 2],
                fileInfo.dimensions[fileInfo.dimensions.size - 1],
                imageEntry
            )
        }
    }
}

fun FlowContent.insertGrid(rows: Int, cols: Int, dataEntry: DataEntry) {
    table {
        for (i in 0 until rows) tr {
            for (j in 0 until cols) td {
                val value = 255 - dataEntry.data[i * cols + j]
                style = "background-color: rgb($value, $value, $value);"
            }
        }
    }
}