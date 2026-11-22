package com.cardtable.belot.desktop

import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.ImageComposeScene
import androidx.compose.ui.unit.Density
import com.cardtable.belot.BULGARIAN_BOT_NAMES
import com.cardtable.belot.BelotController
import com.cardtable.belot.BelotPlatform
import com.cardtable.belot.BelotRoot
import com.cardtable.belot.Phase
import com.cardtable.belot.jvm.BelotNative
import com.cardtable.belot.jvm.JniBelotEngine
import com.cardtable.platform.AppConfig
import com.cardtable.platform.jvm.FileStorage
import org.jetbrains.skia.EncodedImageFormat
import java.io.File
import java.nio.file.Files
import javax.swing.SwingUtilities
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The real app, offscreen: the engine, the nets and the controller as the
 * desktop app runs them, rendered frame by frame into PNGs at the moments a
 * player would look (lobby, auction, play, hint, deal sheet). Runs only with
 * -Pbelot.screenshots=<dir> (see build.gradle.kts); otherwise it skips.
 */
@OptIn(ExperimentalComposeUiApi::class)
class ScreenshotTest {
    private fun <T> edt(block: () -> T): T { var r: T? = null; SwingUtilities.invokeAndWait { r = block() }; @Suppress("UNCHECKED_CAST") return r as T }

    @Test fun screens() {
        val out = System.getProperty("belot.screenshots")?.takeIf { it.isNotBlank() }?.let(::File) ?: return
        out.mkdirs()
        val resources = File(System.getProperty("compose.application.resources.dir"))
        BelotNative.load = { System.load(File(resources, "native/" + System.mapLibraryName("belot_mobile")).absolutePath) }
        AppConfig.botNames = BULGARIAN_BOT_NAMES
        val platform = BelotPlatform(JniBelotEngine(File(resources, "models").path), FileStorage(Files.createTempDirectory("belot").toFile()), {}, System::currentTimeMillis, System.getProperty("belot.lang")?.takeIf { it.isNotBlank() } ?: "bg")
        var c: BelotController? = null
        val w = System.getProperty("belot.width")?.toIntOrNull() ?: 460; val h = System.getProperty("belot.height")?.toIntOrNull() ?: 900
        val scene = edt { ImageComposeScene(w * 2, h * 2, Density(2f)) { BelotRoot(platform) { c = it } } }
        val shots = mutableListOf<String>()
        fun frame() = edt { scene.render(System.nanoTime()) }
        fun shot(name: String) {
            val img = frame()
            File(out, "$name.png").writeBytes(img.encodeToData(EncodedImageFormat.PNG)!!.bytes)
            shots += name
        }
        /** Renders for [ms] (animations, bots), then until [cond] holds or [limitMs] passes. */
        fun runUntil(limitMs: Long = 20_000, ms: Long = 0, cond: (BelotController) -> Boolean = { true }): Boolean {
            val start = System.currentTimeMillis()
            while (System.currentTimeMillis() - start < ms) { frame(); Thread.sleep(40) }
            while (System.currentTimeMillis() - start < limitMs) {
                frame()
                if (edt { cond(c!!) }) return true
                Thread.sleep(40)
            }
            return false
        }
        val prefix = System.getProperty("belot.prefix") ?: ""
        runUntil(ms = 600); shot("${prefix}1-lobby")
        edt { c!!.start(true) }
        // the player's first bid (bots before them bid with their thinking time)
        assertTrue(runUntil { it.view?.phase == Phase.BID && it.legal.isNotEmpty() })
        runUntil(ms = 500); shot("${prefix}2-auction")
        edt { c!!.toggleHint() }
        runUntil(10_000) { it.hints.isNotEmpty() }; runUntil(ms = 300); shot("${prefix}3-bid-hint")
        // bid the hint's pick, then play on until the player's first card
        edt { val p = c!!.hintPick; if (p != null) c!!.act(p) else c!!.bid(0) }
        fun playToCard(): Boolean = runUntil(60_000) {
            when {
                it.dealOver -> true
                it.legal.isNotEmpty() && it.view?.phase == Phase.PLAY -> true
                it.legal.isNotEmpty() -> { it.act(it.legal.first().let { a -> if (it.view?.phase == Phase.BID) 32 else a }); false }
                else -> false
            }
        }
        playToCard()
        runUntil(ms = 800); shot("${prefix}4-play")
        if (edt { !c!!.dealOver }) {
            edt { c!!.toggleHint() }
            runUntil(15_000) { it.hints.isNotEmpty() }; runUntil(ms = 300); shot("${prefix}5-card-hint")
        }
        // play the deal out, a trick on the table half way
        val trickShown = runUntil(240_000) {
            if (it.legal.isNotEmpty()) it.act(it.legal.first())
            it.dealOver || (it.trickCards.size >= 3 && (it.view?.winners?.size ?: 0) >= 3)
        }
        if (trickShown && edt { !c!!.dealOver }) shot("${prefix}6-trick")
        runUntil(240_000) {
            if (it.legal.isNotEmpty()) it.act(it.legal.first())
            it.dealOver
        }
        runUntil(ms = 1200); shot("${prefix}7-deal-sheet")
        println("screenshots: $shots in $out")
        edt { scene.close() }
    }
}
