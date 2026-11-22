pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

// cardbot-framework on its own: just the shared module, to build and check it. The
// game apps include it from their own builds (see README.md).
rootProject.name = "cardbot-framework"
include(":platform")
