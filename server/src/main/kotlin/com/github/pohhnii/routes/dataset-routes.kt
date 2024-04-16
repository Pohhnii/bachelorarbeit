package com.github.pohhnii.routes

import com.github.pohhnii.data.*
import io.ktor.http.*
import io.ktor.server.application.*
import io.ktor.server.response.*
import io.ktor.server.routing.*

@OptIn(ExperimentalUnsignedTypes::class)
fun Route.datasetRoute(dataset: Dataset) {
    route("/${dataset.name}") {
        get("/{id}") {
            val id = call.parameters["id"]?.toIntOrNull()
                ?: return@get call.respond(HttpStatusCode.BadRequest, "Invalid id")

            val label = dataset.labels.use { get(id) }
            val image = dataset.images.use { get(id) }

            call.respond(mapOf("label" to label, "image" to image))
        }

        get("/batch") {
            val idParams = call.parameters["ids"]
                ?: return@get call.respond(HttpStatusCode.BadRequest, "No ids provided")

            val ids = idParams.split(",").mapNotNull { it.toIntOrNull() }
            val labels = dataset.labels.use { get(ids) }
            val images = dataset.images.use { get(ids) }

            call.respond(mapOf("labels" to labels, "images" to images))
        }
    }
}