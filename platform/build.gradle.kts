plugins {
    id("org.jetbrains.kotlin.multiplatform")
    id("com.android.kotlin.multiplatform.library")
    id("org.jetbrains.kotlin.plugin.compose")
}

// What every card-game app shares, whatever the game: profiles, saved games
// and statistics files, the two-language strings, the table themes, the
// cards (drawn through each game's Deck), speech bubbles, prompts, the table
// and its scenery, and the worker thread the bots think on. No game rules
// live here.
kotlin {
    androidLibrary {
        namespace = "com.cardtable.platform"
        compileSdk = 37
        minSdk = 26
        withHostTestBuilder {}
    }
    jvm("desktop")
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        commonMain.dependencies {
            api("org.jetbrains.compose.runtime:runtime:1.12.2")
            api("org.jetbrains.compose.foundation:foundation:1.12.2")
            api("org.jetbrains.compose.ui:ui:1.12.2")
            api("org.jetbrains.compose.material3:material3:1.12.0-alpha03")
            api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.11.0")
            api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.11.0")
        }
        androidMain { kotlin.srcDir("src/jvmShared/kotlin") }
        val desktopMain by getting { kotlin.srcDir("src/jvmShared/kotlin") }
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
    }
}
