package data

typealias DatasetPath = String

data class Dataset(val name: String, val images: DatasetPath, val labels: DatasetPath)

object DATASETS {
    val TEST = Dataset("test", "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    val TRAINING = Dataset("training", "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
}

fun DATASETS.loadDatabaseInfo(): DatabaseInfo {
    val info = DatabaseInfo(
        training = DatasetInfo(
            labels = TRAINING.labels.getDatasetDescription(),
            data = TRAINING.images.getDatasetDescription()
        ),
        testing = DatasetInfo(
            labels = TEST.labels.getDatasetDescription(),
            data = TEST.images.getDatasetDescription()
        )
    )

    return info
}

private fun DatasetPath.getDatasetDescription(): IDXFileInfo {
    return use {
        getInfo()
    }
}