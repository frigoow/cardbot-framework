import org.jetbrains.compose.desktop.application.dsl.TargetFormat

plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.compose")
}

// The Belot desktop shell (Windows, macOS on Apple Silicon, Linux): the game
// and screens from :belot, the C++ engine (cpp/belot) as a JNI library built
// here with CMake for the OS that runs the build, and the nets from the
// Android assets (scripts/belot_models.sh).
//
//   ./gradlew :belot-desktop:run
//   ./gradlew :belot-desktop:packageDistributionForCurrentOS

val repoRoot: File = rootDir.parentFile
val appVersion: String = Regex("""versionName = "([^"]+)"""")
    .find(file("../belot-app/build.gradle.kts").readText())!!.groupValues[1]

dependencies {
    implementation(project(":belot"))
    implementation(compose.desktop.currentOs)
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.11.0")
    testImplementation(kotlin("test"))
}

// The app offscreen, saved as PNGs (ScreenshotTest):
//   ./gradlew :belot-desktop:test -Pbelot.screenshots=/some/dir [-Pbelot.width=900 -Pbelot.height=560]
tasks.test {
    dependsOn(syncModels, buildNative)
    systemProperty("compose.application.resources.dir", appResources.get().dir("common").asFile.path)
    for (p in listOf("belot.screenshots", "belot.width", "belot.height", "belot.prefix", "belot.lang"))
        systemProperty(p, (findProperty(p) as String?) ?: "")
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    outputs.upToDateWhen { false }
}

val appResources = layout.buildDirectory.dir("appResources")

val syncModels by tasks.registering(Sync::class) {
    description = "Copies the Belot nets from the Android assets (scripts/belot_models.sh)."
    from(file("../belot-app/src/main/assets/models")) { exclude("README.md") }
    into(appResources.map { it.dir("common/models") })
}

val buildNative by tasks.registering(CMakeJniLibrary::class) {
    description = "Builds the Belot engine's JNI library for this OS."
    sources.set(File(repoRoot, "cpp/belot"))
    extraSources.set(File(repoRoot, "engines/belote/cpp"))
    workDir.set(layout.buildDirectory.dir("cmake"))
    outputDir.set(appResources.map { it.dir("common/native") })
    cmake.set(provider { findCmake() })
    jniInclude.set(provider { findJniInclude() })
    target.set("belot_mobile")
    definitions.set(appResources.map { listOf("BELOT_JNI=ON", "BELOT_TESTS=OFF", "BELOT_JNI_OUT=${it.dir("common/native").asFile.path}") })
}

compose.desktop {
    application {
        mainClass = "com.cardtable.belot.desktop.MainKt"
        jvmArgs += listOf(
            "--enable-native-access=ALL-UNNAMED",
            "-XX:+UseSerialGC", "-Xms32m", "-Xmx512m",
            "-XX:MinHeapFreeRatio=10", "-XX:MaxHeapFreeRatio=30",
            "-Dskiko.gpu.resourceCacheLimit=64M",
        )
        nativeDistributions {
            targetFormats(TargetFormat.Msi, TargetFormat.Dmg, TargetFormat.Deb)
            packageName = "Belot"
            packageVersion = appVersion
            description = "Bulgarian bridge-belot against strong bots"
            vendor = "Belot"
            appResourcesRootDir.set(appResources)
            modules("jdk.unsupported")
            linux { iconFile.set(file("icons/belot.png")) }
            windows {
                iconFile.set(file("icons/belot.ico"))
                menuGroup = "Belot"
                shortcut = true
                // fixed, so a new version installs over the old one
                upgradeUuid = "b2d6e0a4-5c31-4f7e-8a92-1d4c6f0e3b57"
            }
            macOS {
                // a macOS bundle's version must start at 1 or more (jpackage):
                // until Belot reaches 1.0, 0.x.y is packaged there as 1.x.y
                appVersion.split(".").let { v -> if (v[0] == "0") packageVersion = (listOf("1") + v.drop(1)).joinToString(".") }
                iconFile.set(file("icons/belot.icns"))
                bundleID = "com.cardtable.belot.desktop"
            }
        }
    }
}

tasks.matching { it.name == "prepareAppResources" }.configureEach { dependsOn(syncModels, buildNative) }
