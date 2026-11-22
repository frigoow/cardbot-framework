plugins {
    id("org.jetbrains.kotlin.multiplatform")
    id("com.android.kotlin.multiplatform.library")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Bulgarian Belot: the table (BelotController: the referee's moves shown in
// order, the bots' turns, hints, statistics, the saved game) and its screens
// (BelotUi). The rules and bots are C++ (cpp/belot, over engines/belote)
// behind BelotEngine; on Android and the desktop through JNI (src/jvmShared).
kotlin {
    androidLibrary {
        namespace = "com.cardtable.belot"
        compileSdk = 37
        minSdk = 26
        withHostTestBuilder {}
    }
    jvm("desktop")
    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        commonMain.dependencies {
            api(project(":platform"))
        }
        androidMain {
            kotlin.srcDir("src/jvmShared/kotlin")
            dependencies { implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0") }
        }
        val desktopMain by getting {
            kotlin.srcDir("src/jvmShared/kotlin")
            dependencies { implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.11.0") }
        }
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
    }
}

// The desktop tests play real games on the C++ engine's JNI library and the
// nets, as the desktop app ships them (belot-desktop builds both).
tasks.matching { it.name == "desktopTest" }.configureEach {
    dependsOn(":belot-desktop:buildNative", ":belot-desktop:syncModels")
    (this as Test).systemProperty("belot.resources", rootProject.file("belot-desktop/build/appResources/common").path)
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
