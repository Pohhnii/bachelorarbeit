package com.github.pohhnii.model

import kotlinx.serialization.Serializable

@Serializable
data class DatasetInfo(val labels: IDXFileInfo, val data: IDXFileInfo)

@Serializable
data class DatabaseInfo(val training: DatasetInfo, val testing: DatasetInfo) {
    companion object
}