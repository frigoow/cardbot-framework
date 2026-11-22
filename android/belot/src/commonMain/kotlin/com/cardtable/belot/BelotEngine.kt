package com.cardtable.belot

import com.cardtable.platform.Storage
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.boolean

/*
 * The C++ side (cpp/belot/belot_api.h): the referee, the bots and the hints.
 * Seats are the engine's: S=0 (the player), E=1, N=2 (partner), W=3, in
 * playing order (counter-clockwise); teams S/N = 0 ("us") and E/W = 1.
 * Actions: 0..31 a card, BID + 0..8 a bid, DECLARE + mask the declarations.
 */

const val ACTION_BID = 32
const val ACTION_DECLARE = 64
const val BOT_LEVELS = 3
const val WINNING_SCORE = 151
const val HUMAN = 0
const val PARTNER = 2
fun teamOf(seat: Int) = seat and 1

/** Makes games: a new one, or one continued from [BelotGame.save]. */
interface BelotEngine {
    fun newGame(seed: Long, match: Boolean): BelotGame
    /** null when [text] is not a saved game this engine can continue. */
    fun load(text: String): BelotGame?
}

/** One game (a match to 151, or a single deal). Not thread-safe: copy it for another thread. */
interface BelotGame : AutoCloseable {
    /** What [seat] may see, as belot_view's JSON. */
    fun viewJson(seat: Int): String
    fun apply(seat: Int, action: Int): Boolean
    fun nextDeal(): Boolean
    fun copy(): BelotGame
    /** The bot's action for [seat] (blocks while it searches). */
    fun botAction(seat: Int, level: Int, seed: Long): Int
    /** Each legal action's value in game points and the bot's pick, or null. */
    fun hint(seat: Int, level: Int, seed: Long): Hint?
    /** Another seat's cards (open-cards view). */
    fun hand(seat: Int): Long
    fun save(): String
}

class Hint(val values: Map<Int, Double>, val pick: Int)

/** What the Belot screens need from the platform. */
class BelotPlatform(
    val engine: BelotEngine,
    val storage: Storage,
    val log: (String) -> Unit,
    val epochMillis: () -> Long,
    /** The device's language code ("bg", "en", ...): Bulgarian devices start in Bulgarian, all others in English. */
    val systemLanguage: String = "en",
)

enum class Phase { BID, DECLARE, PLAY, DEAL_OVER, GAME_OVER }

sealed interface LogEntry {
    val seat: Int
    data class Bid(override val seat: Int, val bid: Int) : LogEntry
    data class Declared(override val seat: Int, val type: Int) : LogEntry
    data class Played(override val seat: Int, val card: Int, val belot: Boolean) : LogEntry
}

class Offered(val type: Int, val cards: List<Int>)
class Declaration(val seat: Int, val type: Int, val scored: Int, val cards: List<Int>)

class DealResult(
    /** Game points this deal, per team. */
    val points: List<Int>,
    /** Raw points: cards, last ten, scoring declarations (×2 in no trumps, +90 for a capot). */
    val raw: List<Int>,
    val capot: Boolean,
    /** Points left hanging for the next deal. */
    val hanging: Int,
    val declared: List<Declaration>,
)

/** belot_view parsed: everything one seat may know. */
class BelotView(
    val me: Int,
    val phase: Phase,
    val toMove: Int,
    val first: Int,
    val dealNo: Int,
    val match: Boolean,
    val hand: List<Int>,
    val handCounts: List<Int>,
    val legal: List<Int>,
    /** 0 none yet, 1..4 a suit (clubs, diamonds, hearts, spades), 5 no trumps, 6 all trumps. */
    val contract: Int,
    val contractSeat: Int,
    /** 0, 1 doubled, 2 redoubled. */
    val doubling: Int,
    val offered: List<Offered>,
    val log: List<LogEntry>,
    val winners: List<Int>,
    val cardPoints: List<Int>,
    val score: List<Int>,
    val hanging: Int,
    val result: DealResult?,
    val winner: Int,
) {
    val dealer: Int get() = (first + 3) % 4

    companion object {
        fun parse(text: String): BelotView {
            val j = Json.parseToJsonElement(text).jsonObject
            fun ints(key: String, o: JsonObject = j) = o[key]!!.jsonArray.map { it.jsonPrimitive.int }
            fun int(key: String, o: JsonObject = j) = o[key]!!.jsonPrimitive.int
            val log = j["log"]!!.jsonArray.map { e ->
                val o = e.jsonObject
                when (o["k"]!!.jsonPrimitive.content) {
                    "bid" -> LogEntry.Bid(int("s", o), int("b", o))
                    "decl" -> LogEntry.Declared(int("s", o), int("t", o))
                    else -> LogEntry.Played(int("s", o), int("c", o), int("bel", o) != 0)
                }
            }
            val result = (j["result"] as? JsonObject)?.let { r ->
                DealResult(ints("points", r), ints("raw", r), r["capot"]!!.jsonPrimitive.boolean, int("hanging", r),
                    (r["declared"] as JsonArray).map { d -> val o = d.jsonObject
                        Declaration(int("s", o), int("t", o), int("scored", o), ints("cards", o)) })
            }
            return BelotView(
                me = int("me"),
                phase = when (j["phase"]!!.jsonPrimitive.content) {
                    "bid" -> Phase.BID; "declare" -> Phase.DECLARE; "play" -> Phase.PLAY
                    "deal_over" -> Phase.DEAL_OVER; else -> Phase.GAME_OVER
                },
                toMove = int("toMove"), first = int("first"), dealNo = int("dealNo"),
                match = j["match"]!!.jsonPrimitive.boolean,
                hand = ints("hand"), handCounts = ints("handCounts"), legal = ints("legal"),
                contract = int("contract"), contractSeat = int("contractSeat"), doubling = int("doubling"),
                offered = j["offered"]!!.jsonArray.map { o -> Offered(int("type", o.jsonObject), ints("cards", o.jsonObject)) },
                log = log, winners = ints("winners"),
                cardPoints = ints("cardPoints"), score = ints("score"),
                hanging = j["hanging"]!!.jsonPrimitive.intOrNull ?: 0,
                result = result, winner = int("winner"),
            )
        }
    }
}
