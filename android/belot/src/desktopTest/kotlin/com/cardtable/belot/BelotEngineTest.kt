package com.cardtable.belot

import com.cardtable.belot.jvm.BelotNative
import com.cardtable.belot.jvm.JniBelotEngine
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The Belot engine as the app uses it: through JNI, with the nets the
 * desktop app ships (belot-desktop builds both; see belot/build.gradle.kts).
 */
class BelotEngineTest {
    private val engine: JniBelotEngine by lazy {
        val resources = File(System.getProperty("belot.resources") ?: error("belot.resources is not set"))
        val lib = File(resources, "native/" + System.mapLibraryName("belot_mobile"))
        BelotNative.load = { System.load(lib.absolutePath) }
        JniBelotEngine(File(resources, "models").path)
    }

    /** Plays the bots in every seat until [stop] or the end; checks every view on the way. */
    private fun play(g: BelotGame, level: Int, stop: (BelotView) -> Boolean = { false }): BelotView {
        var seed = 1L
        while (true) {
            val v = BelotView.parse(g.viewJson(HUMAN))
            // the parsed view holds together
            assertEquals(v.hand.size, v.handCounts[HUMAN])
            assertEquals(v.log.count { it is LogEntry.Played } / 4, v.winners.size)
            if (v.phase == Phase.PLAY) assertTrue(v.contract in 1..6)
            if (stop(v) || v.phase == Phase.GAME_OVER) return v
            if (v.phase == Phase.DEAL_OVER) { assertNotNull(v.result); assertTrue(g.nextDeal()); continue }
            val seat = v.toMove
            val seen = BelotView.parse(g.viewJson(seat))
            assertTrue(seen.legal.isNotEmpty())
            val a = g.botAction(seat, level, seed++)
            assertTrue(a in seen.legal, "bot chose $a, legal ${seen.legal}")
            assertTrue(g.apply(seat, a))
        }
    }

    @Test fun wholeGamesTo151() {
        repeat(3) { i ->
            engine.newGame(100L + i, match = true).use { g ->
                val v = play(g, level = 0)
                assertEquals(Phase.GAME_OVER, v.phase)
                assertTrue(v.winner in 0..1)
                assertTrue(v.score[v.winner] >= WINNING_SCORE)
            }
        }
    }

    @Test fun fullStrengthDealAndHints() {
        engine.newGame(7L, match = false).use { g ->
            // a hint at the first decision of the player, whatever it is
            val v = play(g, level = 2) { it.toMove == HUMAN && (it.phase == Phase.BID || it.phase == Phase.PLAY) }
            if (v.phase != Phase.GAME_OVER) {
                val h = g.hint(HUMAN, 2, 3L)
                assertNotNull(h)
                assertTrue(h.values.keys.all { it in v.legal })
            }
            val end = play(g, level = 2)
            assertEquals(Phase.GAME_OVER, end.phase)
        }
    }

    @Test fun aSavedGameContinuesTheSame() {
        engine.newGame(42L, match = true).use { g ->
            play(g, level = 0) { it.dealNo == 2 && it.log.count { e -> e is LogEntry.Played } == 9 }
            val text = g.save()
            val h = assertNotNull(engine.load(text))
            h.use {
                for (s in 0..3) assertEquals(g.viewJson(s), h.viewJson(s))
                assertEquals(text, h.save())
            }
        }
    }
}
