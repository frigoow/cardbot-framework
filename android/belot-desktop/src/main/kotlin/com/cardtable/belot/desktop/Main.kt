package com.cardtable.belot.desktop

import androidx.compose.ui.graphics.painter.BitmapPainter
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import com.cardtable.belot.BelotPlatform
import com.cardtable.belot.BelotRoot
import com.cardtable.belot.BULGARIAN_BOT_NAMES
import com.cardtable.belot.jvm.BelotNative
import com.cardtable.belot.jvm.JniBelotEngine
import com.cardtable.platform.AppConfig
import com.cardtable.platform.desktop.appResources
import com.cardtable.platform.desktop.dataDir
import com.cardtable.platform.jvm.FileStorage
import org.jetbrains.skia.Image
import java.io.File

/** The desktop shell: the engine and its nets, then the Belot screens in a phone-shaped window. */
fun main() {
    val resources = appResources()
    val library = File(resources, "native/" + System.mapLibraryName("belot_mobile"))
    BelotNative.load = { System.load(library.absolutePath) }
    AppConfig.botNames = BULGARIAN_BOT_NAMES
    val log = { m: String -> System.err.println("belot: $m") }
    val platform = BelotPlatform(
        engine = JniBelotEngine(File(resources, "models").path, log),
        storage = FileStorage(dataDir("Belot")),
        log = log,
        epochMillis = System::currentTimeMillis,
        systemLanguage = java.util.Locale.getDefault().language,
    )
    val icon = Thread.currentThread().contextClassLoader.getResourceAsStream("icon.png")
        ?.use { BitmapPainter(Image.makeFromEncoded(it.readBytes()).toComposeImageBitmap()) }
    application {
        Window(onCloseRequest = ::exitApplication, title = "Belot", icon = icon,
            state = rememberWindowState(size = DpSize(460.dp, 900.dp))) {
            BelotRoot(platform)
        }
    }
}
