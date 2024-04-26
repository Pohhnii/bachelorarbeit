package ui

import data.DatasetInfo
import data.FileInfo
import kotlinx.dom.appendElement
import org.w3c.dom.Element

fun Element.datasetDescription(name: String, datasetInfo: DatasetInfo) {
    appendElement("h2") {
        textContent = name
    }

    appendElement("table") {
        appendElement("tr") {
            appendElement("th") {
                textContent = "Name"
            }
            appendElement("th") {
                textContent = "Dimensions"
            }
            appendElement("th") {
                textContent = "Magic number"
            }
            appendElement("th") {
                textContent = "Data type"
            }
        }

        fileInfoTableRow("Labels", datasetInfo.labels)
        fileInfoTableRow("Data", datasetInfo.data)
    }
}

private fun Element.fileInfoTableRow(name: String, fileInfo: FileInfo) {
    appendElement("tr") {
        appendElement("th") {
            textContent = name
        }
        appendElement("td") {
            textContent = dimensionString(fileInfo.dimensions)
        }
        appendElement("td") {
            textContent = "${fileInfo.magicNumber}"
        }

        appendElement("td") {
            textContent = "${fileInfo.dataType}"
        }
    }
}

private fun dimensionString(dimensions: JsArray<JsNumber>): String = js("dimensions.join(' x ')")