package com.github.pohhnii

import java.io.File

private const val LOG_FILE = "logs.txt"

fun log(message: String) {
    println(message)
    File(LOG_FILE).appendText("$message\n")
}