plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

// The Belot Android app: the game and screens from :belot, the C++ engine
// (cpp/belot over engines/belote) built by the NDK, the nets from the assets
// (scripts/belot_models.sh).
//
// Release signing comes from the environment, as for Manille but with the
// app's own key (BELOT_KEYSTORE...); without it a release build falls back
// to the debug key, which is fine for local testing.
val releaseKeystore = System.getenv("BELOT_KEYSTORE")?.let { file(it) }?.takeIf { it.exists() }

android {
    namespace = "com.cardtable.belot.app"
    compileSdk = 37
    ndkVersion = "30.0.16138531"

    defaultConfig {
        applicationId = "com.cardtable.belot"
        minSdk = 26
        targetSdk = 37
        versionCode = 2
        versionName = "0.1.1"
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                arguments += listOf("-DBELOT_JNI=ON", "-DBELOT_TESTS=OFF")
                cppFlags += listOf("-std=c++17", "-O3")
            }
        }
    }

    signingConfigs {
        if (releaseKeystore != null) create("release") {
            storeFile = releaseKeystore
            storePassword = System.getenv("BELOT_KEYSTORE_PASSWORD")
            keyAlias = System.getenv("BELOT_KEY_ALIAS")
            keyPassword = System.getenv("BELOT_KEY_PASSWORD")
        }
    }

    buildTypes {
        release {
            // No R8: the JNI bridge reaches Kotlin by name.
            isMinifyEnabled = false
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }
    externalNativeBuild {
        cmake {
            path = file("../../cpp/belot/CMakeLists.txt")
        }
    }
    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }
}

dependencies {
    implementation(project(":belot"))
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
}
