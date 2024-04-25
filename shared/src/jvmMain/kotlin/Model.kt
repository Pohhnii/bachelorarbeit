import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.operations.joinToString
import java.io.File


fun Model.save(directory: File): Boolean {
    if (!directory.isDirectory) return false

    for (i in layers.indices) {
        val layer = layers[i]
        val layerDir = File(directory, "layer-$i")
        layerDir.mkdirs()
        for (j in layer.data.indices) {
            val dataFile = File(layerDir, "data-$j.csv")
            dataFile.writeText(layer.data[j].joinToString(separator = ", "))
        }
    }

    return true
}

fun Model.load(directory: File): Boolean {
    if (!directory.isDirectory) return false

    for (i in layers.indices) {
        val layer = layers[i]
        val layerDir = File(directory, "layer-$i")

        val newData = mutableListOf<D2Array<Double>>()
        for (j in layer.data.indices) {
            val dataFile = File(layerDir, "data-$j.csv")
            if (!dataFile.isFile) return false
            val fileContent = dataFile.readText()
            newData.add(mk.ndarray(fileContent.split(", ", ",").map { it.toDouble() }, layer.data[j].shape))
        }

        layer.data = newData
    }

    return true
}