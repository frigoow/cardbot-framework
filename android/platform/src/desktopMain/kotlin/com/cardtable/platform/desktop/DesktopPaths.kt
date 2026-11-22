package com.cardtable.platform.desktop

import java.io.File

/** Where Compose put build/appResources/common (models/, native/). */
fun appResources(): File =
    System.getProperty("compose.application.resources.dir")?.let(::File)
        ?: error("compose.application.resources.dir is not set: start the app with ./gradlew :<app>:run")

/**
 * The player's profiles, statistics and saved game, in the usual place per
 * OS: %APPDATA%\[name], ~/Library/Application Support/[name], or
 * ~/.local/share/[unixName].
 */
fun dataDir(name: String, unixName: String = name.lowercase()): File {
    val os = System.getProperty("os.name").lowercase()
    val home = System.getProperty("user.home")
    return when {
        "win" in os -> File(System.getenv("APPDATA") ?: home, name)
        "mac" in os -> File(home, "Library/Application Support/$name")
        else -> File(System.getenv("XDG_DATA_HOME")?.takeIf { it.isNotBlank() } ?: "$home/.local/share", unixName)
    }
}
