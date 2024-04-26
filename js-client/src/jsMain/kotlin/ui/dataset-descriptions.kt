package ui

import data.DatasetInfo
import data.Dimensions
import data.IDXFileInfo
import kotlinx.html.*

fun FlowContent.datasetDescriptions(name: String, datasetInfo: DatasetInfo) {
    h2 { +name }

    table {
        tr {
            th { +"Name" }
            th { +"Dimensions" }
            th { +"Magic number" }
            th { +"Data type" }
        }

        fileInfoTableRow("Labels", datasetInfo.labels)
        fileInfoTableRow("Data", datasetInfo.data)
    }
}

private fun TABLE.fileInfoTableRow(name: String, fileInfo: IDXFileInfo) {
    tr {
        th { +name }
        td { +dimensionString(fileInfo.dimensions) }
        td { +"${fileInfo.magicNumber}" }
        td { +(fileInfo.dataType ?: "") }
    }
}

private fun dimensionString(dimensions: Dimensions) = dimensions.joinToString(" x ")