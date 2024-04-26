import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig

plugins {
    alias(libs.plugins.kotlinMultiplatform)
}

group = "com.github.pohhnii"
version = "unspecified"


kotlin {
    js {
        moduleName = "js-client"
        browser {
            commonWebpackConfig {
                outputFileName = "js-client.js"
                cssSupport {
                    enabled.set(true)
                }
                devServer = (devServer ?: KotlinWebpackConfig.DevServer()).apply {
                    port = 9080
                    static = (static ?: mutableListOf()).apply {
                        // Serve sources to debug inside browser
                        add(project.projectDir.path)
                    }
                }
            }
        }
        binaries.executable()
    }

    sourceSets {
        commonMain.dependencies {
            implementation(projects.shared)
            implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1-Beta")
            implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
            implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
            implementation("org.jetbrains.kotlinx:kotlinx-html:0.11.0")
            implementation("org.jetbrains.kotlinx:kotlinx-html-js:0.11.0")
        }
    }
}

