package data

typealias DatasetPath = String

data class Dataset(val name: String, val images: DatasetPath, val labels: DatasetPath)