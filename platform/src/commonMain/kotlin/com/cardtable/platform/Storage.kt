package com.cardtable.platform

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put

/** Small named text files in the app's private storage. */
interface Storage {
    fun read(name: String): String?
    /** Replaces the file atomically (write aside, then rename). */
    fun write(name: String, text: String)
    fun delete(name: String)
    fun exists(name: String): Boolean
    fun rename(from: String, to: String): Boolean
}

// The Manille files are the ones its Android app wrote with org.json before
// the shared module existed (same names, same keys): an update keeps every
// profile, statistic and saved game. The accessors read leniently, as
// org.json's opt* did.

fun JsonObject.prim(key: String): JsonPrimitive? = this[key] as? JsonPrimitive

fun JsonObject.optInt(key: String, default: Int = 0): Int =
    prim(key)?.let { it.intOrNull ?: it.doubleOrNull?.toInt() ?: it.content.toIntOrNull() } ?: default

fun JsonObject.optLong(key: String, default: Long = 0L): Long =
    prim(key)?.let { it.longOrNull ?: it.doubleOrNull?.toLong() ?: it.content.toLongOrNull() } ?: default

fun JsonObject.optBoolean(key: String, default: Boolean = false): Boolean =
    prim(key)?.let { it.booleanOrNull ?: it.content.toBooleanStrictOrNull() } ?: default

fun JsonObject.optString(key: String, default: String = ""): String =
    prim(key)?.takeIf { it !is kotlinx.serialization.json.JsonNull }?.content ?: default

fun JsonObject.req(key: String): JsonPrimitive =
    prim(key) ?: throw NoSuchElementException("missing $key")

fun JsonObject.getInt(key: String): Int = req(key).let { it.intOrNull ?: error("$key is not an int") }
fun JsonObject.getLong(key: String): Long = req(key).let { it.longOrNull ?: error("$key is not a long") }
fun JsonObject.getBoolean(key: String): Boolean = req(key).let { it.booleanOrNull ?: error("$key is not a boolean") }
fun JsonObject.getString(key: String): String = req(key).content
fun JsonObject.optArray(key: String): JsonArray? = this[key] as? JsonArray
fun JsonObject.optObject(key: String): JsonObject? = this[key] as? JsonObject

/** Atomic small-JSON files in the app's private storage. */
class JsonFiles(private val storage: Storage) {
    fun read(name: String): JsonObject? =
        runCatching { storage.read(name)?.let { Json.parseToJsonElement(it).jsonObject } }.getOrNull()

    fun write(name: String, value: JsonObject) = storage.write(name, value.toString())
}

data class Profile(val id: String, val name: String)

/**
 * Local player profiles: each has its own saved game and statistics, and
 * the settings are shared. The first run creates "You" and moves the
 * single-profile files of earlier versions into it.
 *
 * @param defaultLanguage the app's own language, used until the player
 *   picks one in the lobby or menu.
 * @param defaultBot the bots' strength setting until the player picks one
 *   (Manille: play-out plies; Belot: a level).
 */
class ProfileStore(
    private val storage: Storage,
    private val defaultLanguage: String = "nl",
    private val defaultBot: Int = 0,
    private val defaultTheme: String = "cafe",
) {
    private val files = JsonFiles(storage)

    var profiles: List<Profile> = emptyList(); private set
    var activeId: String = ""; private set
    var fourColor: Boolean = true; private set
    var smileys: Boolean = true; private set
    var animations: Boolean = true; private set
    /** The bots' strength setting (stored as "netPlies", the Manille app's first such setting). */
    var botSetting: Int = defaultBot; private set
    /** The language code; only an explicit choice overrides [defaultLanguage]. */
    var language: String = defaultLanguage; private set
    private var languageChosen = false
    var theme: String = defaultTheme; private set

    init { load() }

    val active: Profile get() = profiles.first { it.id == activeId }

    private fun load() {
        val j = files.read(FILE)
        if (j == null) {
            profiles = listOf(Profile("p1", "You")); activeId = "p1"
            // migrate the pre-profile files into the first profile
            if (storage.exists("stats.json")) storage.rename("stats.json", "stats_p1.json")
            if (storage.exists("savegame.json")) storage.rename("savegame.json", "savegame_p1.json")
            save(); return
        }
        profiles = (j.optArray("profiles") ?: JsonArray(emptyList())).mapNotNull { e ->
            (e as? JsonObject)?.let { p -> Profile(p.getString("id"), p.getString("name")) }
        }
        if (profiles.isEmpty()) profiles = listOf(Profile("p1", "You"))
        activeId = j.optString("active").takeIf { id -> profiles.any { it.id == id } } ?: profiles[0].id
        fourColor = j.optBoolean("fourColor", true)
        smileys = j.optBoolean("smileys", true)
        animations = j.optBoolean("animations", true)
        botSetting = j.optInt("netPlies", defaultBot)
        languageChosen = j.optBoolean("languageChosen", false)
        language = if (languageChosen) j.optString("language", defaultLanguage) else defaultLanguage
        theme = j.optString("theme", defaultTheme)
    }

    private fun save() {
        val arr = JsonArray(profiles.map { buildJsonObject { put("id", it.id); put("name", it.name) } })
        runCatching {
            files.write(FILE, buildJsonObject {
                put("profiles", arr); put("active", activeId); put("fourColor", fourColor)
                put("smileys", smileys); put("animations", animations); put("netPlies", botSetting)
                put("language", language); put("languageChosen", languageChosen); put("theme", theme)
            })
        }
    }

    fun select(id: String) { if (profiles.any { it.id == id }) { activeId = id; save() } }

    fun create(name: String): Profile {
        val next = (profiles.mapNotNull { it.id.removePrefix("p").toIntOrNull() }.maxOrNull() ?: 0) + 1
        val p = Profile("p$next", name.trim().ifEmpty { "Player $next" })
        profiles = profiles + p; activeId = p.id; save()
        return p
    }

    fun rename(id: String, name: String) {
        if (name.isBlank()) return
        profiles = profiles.map { if (it.id == id) it.copy(name = name.trim()) else it }; save()
    }

    /** Deletes a profile with its save and statistics; the last one stays. */
    fun delete(id: String) {
        if (profiles.size <= 1) return
        profiles = profiles.filter { it.id != id }
        storage.delete("stats_$id.json")
        storage.delete("savegame_$id.json")
        if (activeId == id) activeId = profiles[0].id
        save()
    }

    fun setFourColor(on: Boolean) { fourColor = on; save() }
    fun setSmileys(on: Boolean) { smileys = on; save() }
    fun setAnimations(on: Boolean) { animations = on; save() }
    fun setBotSetting(value: Int) { botSetting = value; save() }
    fun setLanguage(lang: String) { language = lang; languageChosen = true; save() }
    fun setTheme(id: String) { theme = id; save() }

    private companion object { const val FILE = "profiles.json" }
}

/**
 * The game in progress for one profile: written at every settled point of
 * a deal, read back by "Resume" in the lobby, removed when the game is over.
 */
class SaveStore(private val storage: Storage, profileId: String) {
    private val files = JsonFiles(storage)
    private val name = "savegame_$profileId.json"

    fun read(): JsonObject? = files.read(name)?.takeIf { it.optInt("version") == VERSION }
    fun write(save: JsonObject) = runCatching {
        files.write(name, JsonObject(save + ("version" to JsonPrimitive(VERSION))))
    }
    fun clear() { storage.delete(name) }

    companion object { const val VERSION = 1 }
}

/**
 * Lifetime statistics of one profile: named counters (so new ones can be
 * added without breaking old files) plus the last results.
 */
class Stats(val c: Map<String, Int>, val history: List<JsonObject>) {
    operator fun get(key: String) = c[key] ?: 0
    fun rate(won: String, of: String): String = if (get(of) == 0) "–" else "${formatFixed(get(won) * 100.0 / get(of), 0)}%"
    fun avg(sum: String, of: String): String = if (get(of) == 0) "–" else formatFixed(get(sum).toDouble() / get(of), 1)
}

class StatsStore(storage: Storage, profileId: String, private val clock: () -> Long) {
    private val files = JsonFiles(storage)
    private val name = "stats_$profileId.json"

    fun load(): Stats {
        val root = files.read(name) ?: JsonObject(emptyMap())
        val c = mutableMapOf<String, Int>()
        root.optObject("counters")?.let { j -> j.keys.forEach { c[it] = j.optInt(it) } }
        // the first Manille app version kept only these totals
        root.optObject("totals")?.let { j -> j.keys.forEach { k -> if (k !in c) c[k] = j.optInt(k) } }
        val h = root.optArray("history") ?: JsonArray(emptyList())
        return Stats(c, h.mapNotNull { it as? JsonObject })
    }

    private fun write(c: Map<String, Int>, history: List<JsonObject>) {
        runCatching {
            files.write(name, buildJsonObject {
                put("counters", JsonObject(c.mapValues { JsonPrimitive(it.value) }))
                put("history", JsonArray(history.takeLast(HISTORY_CAP)))
            })
        }
    }

    /** Changes the counters (the game's own records), then returns the new statistics. */
    fun update(change: (Counters) -> Unit): Stats {
        val s = load(); val c = Counters(s.c.toMutableMap())
        change(c)
        write(c.map, s.history)
        return load()
    }

    /** A single game or a match ended: [us] and [them] are the final scores. */
    fun recordResult(match: Boolean, us: Int, them: Int): Stats {
        val s = load(); val c = Counters(s.c.toMutableMap())
        if (match) {
            c.add("matches"); if (us > them) c.add("matches_won") else c.add("matches_lost")
            c.streak("match_streak", "best_match_streak", us > them)
            if (us > them) c.max("best_match_margin", us - them)
        } else {
            c.add("games"); when { us > them -> c.add("games_won"); us < them -> c.add("games_lost"); else -> c.add("games_tied") }
        }
        val h = s.history + buildJsonObject {
            put("t", clock() / 1000); put("mode", if (match) "match" else "game"); put("us", us); put("them", them)
        }
        write(c.map, h)
        return load()
    }

    fun countHint(): Stats = update { it.add("hints_used") }

    private companion object { const val HISTORY_CAP = 200 }
}

/** The counters while a record is written. */
class Counters(val map: MutableMap<String, Int>) {
    operator fun get(key: String) = map[key] ?: 0
    fun add(key: String, v: Int = 1) { map[key] = get(key) + v }
    fun max(key: String, v: Int) { map[key] = maxOf(get(key), v) }
    /** A run of wins: [current] grows on [won], else restarts; [best] keeps the longest. */
    fun streak(current: String, best: String, won: Boolean) {
        val s = if (won) get(current) + 1 else 0
        map[current] = s; max(best, s)
    }
}

/**
 * [x] with [decimals] digits after the point, rounded half up like
 * String.format; the decimal separator follows the app language (a comma in
 * Dutch and Bulgarian).
 */
fun formatFixed(x: Double, decimals: Int): String {
    var scale = 1L
    repeat(decimals) { scale *= 10 }
    val neg = x < 0
    val v = kotlin.math.floor(kotlin.math.abs(x) * scale + 0.5).toLong()
    val whole = v / scale
    val frac = (v % scale).toString().padStart(decimals, '0')
    val sep = if (Lang.local) "," else "."
    return (if (neg && v != 0L) "-" else "") + whole + (if (decimals > 0) sep + frac else "")
}
