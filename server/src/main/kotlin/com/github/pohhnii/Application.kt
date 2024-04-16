package com.github.pohhnii

import SERVER_PORT
import com.github.pohhnii.data.DATASETS
import com.github.pohhnii.data.loadDatabaseInfo
import com.github.pohhnii.routes.datasetRoute
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.plugins.cors.routing.*
import io.ktor.server.response.*
import io.ktor.server.routing.*

fun main() {
    embeddedServer(Netty, port = SERVER_PORT, host = "0.0.0.0") {
        install(ContentNegotiation) {
            json()
        }

        install(CORS) {
            anyHost()
        }

        module()
    }.start(wait = true)
}

fun Application.module() {

    routing {
        get("/") {
            call.respondText("Healthy")
        }

        route("/dataset") {
            get("/descriptions") {
                val info = DATASETS.loadDatabaseInfo()
                call.respond(info)
            }

            datasetRoute(DATASETS.TEST)
            datasetRoute(DATASETS.TRAINING)
        }

    }
}