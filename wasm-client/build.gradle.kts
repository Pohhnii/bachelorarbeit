import org.jetbrains.kotlin.gradle.targets.js.dsl.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpackConfig

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    
    alias(libs.plugins.jetbrainsCompose)
}

kotlin {
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        moduleName = "wasm-client"
        browser {
            commonWebpackConfig {
                outputFileName = "wasm-client.js"
                cssSupport {
                    enabled.set(true)
                }
                devServer = (devServer ?: KotlinWebpackConfig.DevServer()).apply {
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
            implementation(compose.runtime)
            implementation(compose.foundation)
            implementation(compose.material)
            implementation(compose.ui)
            implementation(compose.components.resources)
            implementation(compose.components.uiToolingPreview)
            implementation(projects.shared)
            implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1-Beta")
            implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
            implementation("org.jetbrains.kotlinx:multik-default:0.2.3")
        }
    }
}

compose.experimental {
    web.application {}
}