package com.cardtable.belot.jvm

import com.cardtable.belot.BelotEngine
import com.cardtable.belot.BelotGame
import com.cardtable.belot.Hint

/**
 * The C++ engine through JNI (cpp/belot/jni_belot.cpp): one static native
 * per belot_api.h function. Games are handles; [JniBelotGame] owns one.
 */
object BelotNative {
    /** How the library is loaded: System.loadLibrary on Android, System.load(path) on the desktop. */
    @Volatile var load: () -> Unit = { System.loadLibrary("belot_mobile") }
    private val loaded by lazy { load(); true }
    fun ensureLoaded() { check(loaded) }

    @JvmStatic external fun lastError(): String
    @JvmStatic external fun modelsLoad(dir: String): Int
    @JvmStatic external fun modelsInfo(): String
    @JvmStatic external fun create(seed: Long, match: Boolean): Long
    @JvmStatic external fun copy(handle: Long): Long
    @JvmStatic external fun free(handle: Long)
    @JvmStatic external fun nextDeal(handle: Long): Int
    @JvmStatic external fun view(handle: Long, seat: Int): String?
    @JvmStatic external fun hand(handle: Long, seat: Int): Long
    @JvmStatic external fun apply(handle: Long, seat: Int, action: Int): Int
    @JvmStatic external fun botAction(handle: Long, seat: Int, level: Int, seed: Long): Int
    @JvmStatic external fun hint(handle: Long, seat: Int, level: Int, seed: Long): DoubleArray?
    @JvmStatic external fun save(handle: Long): String?
    @JvmStatic external fun load(text: String): Long
}

/** The JNI engine; [modelDir] holds the nets (belot_api.h: belot_models_load), loaded once. */
class JniBelotEngine(modelDir: String?, log: (String) -> Unit = {}) : BelotEngine {
    init {
        BelotNative.ensureLoaded()
        if (modelDir != null) {
            val n = BelotNative.modelsLoad(modelDir)
            log("belot models: $n loaded (${BelotNative.modelsInfo()})")
        }
    }

    override fun newGame(seed: Long, match: Boolean): BelotGame {
        val h = BelotNative.create(seed, match)
        check(h != 0L) { "new game: ${BelotNative.lastError()}" }
        return JniBelotGame(h)
    }

    override fun load(text: String): BelotGame? = BelotNative.load(text).takeIf { it != 0L }?.let(::JniBelotGame)
}

class JniBelotGame(private var handle: Long) : BelotGame {
    private fun h(): Long { check(handle != 0L) { "the game is closed" }; return handle }

    override fun viewJson(seat: Int): String = BelotNative.view(h(), seat) ?: error("view: ${BelotNative.lastError()}")
    override fun apply(seat: Int, action: Int): Boolean = BelotNative.apply(h(), seat, action) >= 0
    override fun nextDeal(): Boolean = BelotNative.nextDeal(h()) >= 0
    override fun copy(): BelotGame = JniBelotGame(BelotNative.copy(h()).also { check(it != 0L) { "copy failed" } })
    override fun botAction(seat: Int, level: Int, seed: Long): Int =
        BelotNative.botAction(h(), seat, level, seed).also { check(it >= 0) { "bot: ${BelotNative.lastError()}" } }

    override fun hint(seat: Int, level: Int, seed: Long): Hint? {
        val raw = BelotNative.hint(h(), seat, level, seed) ?: return null
        if (raw.isEmpty()) return null
        val n = (raw.size - 1) / 2
        val values = (0 until n).associate { raw[2 * it].toInt() to raw[2 * it + 1] }
        return Hint(values, raw.last().toInt())
    }

    override fun hand(seat: Int): Long = BelotNative.hand(h(), seat)
    override fun save(): String = BelotNative.save(h()) ?: error("save: ${BelotNative.lastError()}")

    override fun close() {
        if (handle != 0L) { BelotNative.free(handle); handle = 0L }
    }
}
