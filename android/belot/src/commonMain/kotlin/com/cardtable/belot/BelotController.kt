package com.cardtable.belot

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.cardtable.platform.AppConfig
import com.cardtable.platform.Lang
import com.cardtable.platform.ProfileStore
import com.cardtable.platform.SaveStore
import com.cardtable.platform.Stats
import com.cardtable.platform.StatsStore
import com.cardtable.platform.Worker
import com.cardtable.platform.optArray
import com.cardtable.platform.optBoolean
import com.cardtable.platform.optInt
import com.cardtable.platform.optString
import com.cardtable.platform.themeById
import com.cardtable.platform.tr
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlin.math.exp
import kotlin.random.Random
import kotlin.time.TimeSource

/** A speech bubble at a seat: what was said, and whether it is a bid that raises the stakes. */
data class Bubble(val text: String, val loud: Boolean = false, val suit: Int? = null)

/**
 * The Belot table on screen. The engine is the referee; this class runs the
 * table around it on the main thread: it shows the engine's public record
 * (bids, declarations, cards) one entry at a time, waits for the player's
 * choices, lets the bots think on the worker (on a copy of the game), holds
 * each finished trick for a moment, and keeps the profiles, statistics,
 * settings, hints and the saved game.
 */
class BelotController(private val platform: BelotPlatform) : AutoCloseable {
    private val storage = platform.storage
    // the same themes as Manille, its café table first; the language follows the device until the player picks one
    private val profileStore = ProfileStore(storage, defaultLanguage = if (platform.systemLanguage == "bg") "bg" else "en",
        defaultBot = 2, defaultTheme = "cafe")
    init { Lang.code = if (profileStore.language == "en") "en" else "bg" }
    private var saves = SaveStore(storage, profileStore.activeId)
    private var statsStore = newStatsStore(profileStore.activeId)
    private fun newStatsStore(id: String) = StatsStore(storage, id, platform.epochMillis)

    private val worker = Worker()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
    private var director: Job? = null
    private var game: BelotGame? = null
    /** The deal (dealNo, match) whose result is in the statistics. */
    private var recordedDeal = -1

    // ---- what the screen shows ------------------------------------------------
    var inGame by mutableStateOf(false); private set
    /** The engine's view of seat 0, as far as the screen has caught up with it. */
    var view by mutableStateOf<BelotView?>(null); private set
    /** Log entries of [view] already on screen. */
    private var shown = 0
    var hand by mutableStateOf<List<Int>>(emptyList()); private set
    var handCounts by mutableStateOf(List(4) { 0 }); private set
    /** The cards on the table and who played them. */
    var trickCards by mutableStateOf<List<Int>>(emptyList()); private set
    var trickSeats by mutableStateOf<List<Int>>(emptyList()); private set
    /** The finished trick's cards sweep towards this seat (-1: not now). */
    var collectTo by mutableStateOf(-1); private set
    var lastTrickCards by mutableStateOf<List<Int>>(emptyList()); private set
    var lastTrickSeats by mutableStateOf<List<Int>>(emptyList()); private set
    var lastTrickWinner by mutableStateOf(-1); private set
    var bubbles by mutableStateOf<Map<Int, List<Bubble>>>(emptyMap()); private set
    var reactions by mutableStateOf<Map<Int, String>>(emptyMap()); private set
    var message by mutableStateOf(""); private set
    /** What the player may do now (empty: not their turn, or the screen is still catching up). */
    var legal by mutableStateOf<List<Int>>(emptyList()); private set
    /** The deal is over and on screen: the score sheet is up. */
    var dealOver by mutableStateOf(false); private set
    var hints by mutableStateOf<Map<Int, Float>>(emptyMap()); private set
    var hintPick by mutableStateOf<Int?>(null); private set
    var thinkingHint by mutableStateOf(false); private set
    var showOpenCards by mutableStateOf(false); private set
    var botNames by mutableStateOf(AppConfig.botNames.take(3)); private set

    // ---- profile and settings ---------------------------------------------------
    var stats by mutableStateOf(statsStore.load()); private set
    var profiles by mutableStateOf(profileStore.profiles); private set
    var profileName by mutableStateOf(profileStore.active.name); private set
    var activeProfileId by mutableStateOf(profileStore.activeId); private set
    var fourColor by mutableStateOf(profileStore.fourColor); private set
    var smileys by mutableStateOf(profileStore.smileys); private set
    var animations by mutableStateOf(profileStore.animations); private set
    var theme by mutableStateOf(themeById(profileStore.theme)); private set
    var level by mutableStateOf(profileStore.botSetting.coerceIn(0, BOT_LEVELS - 1)); private set
    var english by mutableStateOf(Lang.code == "en"); private set
    var savedGame by mutableStateOf(saveSummary()); private set
    /** The bots' thinking time over their recent real decisions (ms). */
    var botMsMean by mutableStateOf<Long?>(null); private set
    private val botMs = ArrayDeque<Long>()

    private fun reloadProfile() {
        saves = SaveStore(storage, profileStore.activeId)
        statsStore = newStatsStore(profileStore.activeId)
        stats = statsStore.load(); savedGame = saveSummary()
        profiles = profileStore.profiles; profileName = profileStore.active.name
        activeProfileId = profileStore.activeId
    }
    fun selectProfile(id: String) { lobby(); profileStore.select(id); reloadProfile() }
    fun createProfile(name: String) { lobby(); profileStore.create(name); reloadProfile() }
    fun renameProfile(id: String, name: String) { profileStore.rename(id, name); reloadProfile() }
    fun deleteProfile(id: String) { lobby(); profileStore.delete(id); reloadProfile() }
    fun setFourColorDeck(on: Boolean) { profileStore.setFourColor(on); fourColor = on }
    fun chooseTheme(id: String) { profileStore.setTheme(id); theme = themeById(id) }
    fun enableAnimations(on: Boolean) { profileStore.setAnimations(on); animations = on }
    fun enableSmileys(on: Boolean) { profileStore.setSmileys(on); smileys = on; if (!on) reactions = emptyMap() }
    fun chooseLevel(l: Int) {
        if (l !in 0 until BOT_LEVELS) return
        profileStore.setBotSetting(l); level = l; botMs.clear(); botMsMean = null
    }
    fun chooseEnglish(on: Boolean) {
        profileStore.setLanguage(if (on) "en" else "bg"); Lang.code = if (on) "en" else "bg"; english = on
        savedGame = saveSummary()
    }
    fun toggleOpenCards() { showOpenCards = !showOpenCards }
    fun openHand(seat: Int): Long = game?.hand(seat) ?: 0L

    fun seatName(seat: Int) = if (seat == HUMAN) tr("You", "Вие") else displayName(botNames[seat - 1])

    // ---- starting, resuming, leaving ----------------------------------------------
    fun start(match: Boolean) {
        stop()
        saves.clear()
        botNames = AppConfig.botNames.shuffled().take(3)
        val g = platform.engine.newGame(Random.nextLong(), match)
        begin(g, fresh = true)
    }

    fun restart() { val match = view?.match ?: true; start(match) }

    /** Continue the saved game; false if there is none or it cannot be read. */
    fun resume(): Boolean {
        val j = saves.read() ?: return false
        val g = platform.engine.load(j.optString("engine"))
        if (g == null) {
            saves.clear(); savedGame = null
            message = tr("The saved game could not be restored.", "Записаната игра не можа да бъде възстановена.")
            return false
        }
        stop()
        j.optArray("names")?.let { a -> if (a.size == 3) botNames = List(3) { a[it].jsonPrimitive.content } }
        begin(g, fresh = false)
        return true
    }

    fun lobby() {
        stop()
        inGame = false
        savedGame = saveSummary()
    }

    private fun stop() {
        director?.cancel(); director = null
        pendingHuman?.cancel(); pendingHuman = null
        pendingNext?.cancel(); pendingNext = null
        val g = game; game = null
        // a bot or hint in flight still uses its own copy; the game itself is ours
        g?.close()
        clearTable()
    }

    private fun clearTable() {
        view = null; shown = 0; hand = emptyList(); handCounts = List(4) { 0 }
        trickCards = emptyList(); trickSeats = emptyList(); collectTo = -1
        lastTrickCards = emptyList(); lastTrickSeats = emptyList(); lastTrickWinner = -1
        bubbles = emptyMap(); reactions = emptyMap(); legal = emptyList(); dealOver = false
        hints = emptyMap(); hintPick = null; thinkingHint = false
    }

    private fun begin(g: BelotGame, fresh: Boolean) {
        game = g
        inGame = true
        val v = BelotView.parse(g.viewJson(HUMAN))
        if (!fresh) {
            // a resumed deal: what was played is on the table already, without replaying it
            shown = v.log.size
            view = v
            restoreTable(v)
            // a finished deal in the save was already in the statistics
            if (v.phase == Phase.DEAL_OVER || v.phase == Phase.GAME_OVER) recordedDeal = v.dealNo * 2 + (if (v.match) 1 else 0)
        }
        director = scope.launch { direct() }
    }

    // ---- the table: the referee's record, one entry at a time ------------------------

    private var pendingHuman: CompletableDeferred<Int>? = null
    private var pendingNext: CompletableDeferred<Unit>? = null
    /** The five cards held during the auction (the last three come after it). */
    private var hand5: List<Int>? = null

    private suspend fun direct() {
        while (true) {
            val g = game ?: return
            val v = BelotView.parse(g.viewJson(HUMAN))
            if (v.dealNo != view?.dealNo) newDeal(v)
            view = v
            if (v.phase == Phase.BID && hand5 == null) hand5 = v.hand
            while (shown < v.log.size) {
                present(v, shown)
                shown++
            }
            refreshTable(v)
            when {
                v.phase == Phase.DEAL_OVER || v.phase == Phase.GAME_OVER -> {
                    dealFinished(v)
                    val next = CompletableDeferred<Unit>(); pendingNext = next
                    next.await(); pendingNext = null
                    if (v.phase == Phase.GAME_OVER) return
                    dealOver = false
                    if (!g.nextDeal()) return
                }
                v.toMove == HUMAN -> {
                    legal = v.legal
                    message = when (v.phase) {
                        Phase.BID -> tr("Your bid.", "Вашата обява.")
                        Phase.DECLARE -> tr("Declare your combinations?", "Обявявате ли комбинациите си?")
                        else -> tr("Your turn.", "Ваш ред.")
                    }
                    val wait = CompletableDeferred<Int>(); pendingHuman = wait
                    val action = wait.await(); pendingHuman = null
                    legal = emptyList(); hints = emptyMap(); hintPick = null
                    if (!g.apply(HUMAN, action)) message = tr("That is not allowed now.", "Това не е позволено сега.")
                    persist()
                }
                else -> {
                    val seat = v.toMove
                    message = tr("${seatName(seat)} is thinking…", "${seatName(seat)} мисли…")
                    val action = think(g, seat, v.phase)
                    if (game !== g) return
                    g.apply(seat, action)
                    persist()
                }
            }
        }
    }

    /** The bot's decision on a copy, on the worker, taking at least a moment (so it can be followed). */
    private suspend fun think(g: BelotGame, seat: Int, phase: Phase): Int {
        val copy = g.copy()
        val t0 = TimeSource.Monotonic.markNow()
        val action = withContext(worker.dispatcher) {
            worker.lock.withLock { try { copy.botAction(seat, level, Random.nextLong()) } finally { copy.close() } }
        }
        val ms = t0.elapsedNow().inWholeMilliseconds
        if (phase == Phase.PLAY) recordBotMs(ms)
        val least = when (phase) { Phase.BID -> 650L; Phase.DECLARE -> 400L; else -> 700L }
        if (ms < least) delay(least - ms)
        return action
    }

    private fun recordBotMs(ms: Long) {
        botMs.addLast(ms); while (botMs.size > 24) botMs.removeFirst()
        botMsMean = botMs.sum() / botMs.size
    }

    private fun newDeal(v: BelotView) {
        shown = 0; hand5 = null; dealOver = false
        trickCards = emptyList(); trickSeats = emptyList(); collectTo = -1
        lastTrickCards = emptyList(); lastTrickSeats = emptyList(); lastTrickWinner = -1
        bubbles = emptyMap()
        message = if (v.first == HUMAN) tr("New deal: you bid first.", "Нова раздавка: Вие обявявате първи.")
                  else tr("New deal: ${seatName(v.first)} bids first.", "Нова раздавка: ${seatName(v.first)} обявява първи.")
    }

    private fun auctionShown(v: BelotView): Boolean {
        val bids = v.log.count { it is LogEntry.Bid }
        return shown >= bids && v.phase != Phase.BID
    }

    /** The hand and the counts as far as the screen has shown the record. */
    private fun refreshTable(v: BelotView) {
        val shownLog = v.log.subList(0, shown)
        val auctionDone = auctionShown(v) || shownLog.any { it is LogEntry.Played }
        val played = shownLog.filterIsInstance<LogEntry.Played>()
        val unshownMine = v.log.subList(shown, v.log.size).filterIsInstance<LogEntry.Played>().filter { it.seat == HUMAN }.map { it.card }
        val cards = if (!auctionDone && hand5 != null) hand5!! else v.hand + unshownMine
        hand = sortHand(cards, if (auctionDone) v.contract else 0)
        handCounts = List(4) { s -> if (!auctionDone) 5 else 8 - played.count { it.seat == s } }
        bubbles = bubblesFor(v, shownLog, auctionDone)
    }

    private fun bubblesFor(v: BelotView, shownLog: List<LogEntry>, auctionDone: Boolean): Map<Int, List<Bubble>> {
        val out = mutableMapOf<Int, MutableList<Bubble>>()
        if (!auctionDone) {
            // each seat's latest word in the auction
            shownLog.filterIsInstance<LogEntry.Bid>().forEach { b ->
                out[b.seat] = mutableListOf(Bubble(bidLabel(b.bid), loud = b.bid >= 7, suit = if (b.bid in 1..4) b.bid - 1 else null))
            }
        } else if (v.contract > 0) {
            out.getOrPut(v.contractSeat) { mutableListOf() } += Bubble(contractLabel(v.contract, v.doubling), loud = v.doubling > 0, suit = trumpSuitOf(v.contract))
        }
        shownLog.forEach { e ->
            when (e) {
                is LogEntry.Declared -> out.getOrPut(e.seat) { mutableListOf() } += Bubble(declarationName(e.type))
                is LogEntry.Played -> if (e.belot) out.getOrPut(e.seat) { mutableListOf() } += Bubble(declarationName(1))
                else -> {}
            }
        }
        return out
    }

    private suspend fun present(v: BelotView, index: Int) {
        val e = v.log[index]
        when (e) {
            is LogEntry.Bid -> {
                refreshTableUpTo(v, index + 1)
                message = if (e.seat == HUMAN) tr("You: ${bidLabel(e.bid)}.", "Вие: ${bidLabel(e.bid)}.") else "${seatName(e.seat)}: ${bidLabel(e.bid)}."
                if (e.bid >= 7 && e.seat != HUMAN) react(e.seat, "😈", "😏", tr("Double! 😈", "Контра! 😈"))
                if (e.seat != HUMAN) delay(BID_SHOW_MS)
            }
            is LogEntry.Declared -> {
                refreshTableUpTo(v, index + 1)
                message = "${seatName(e.seat)}: ${declarationName(e.type)}!"
                delay(DECL_SHOW_MS)
            }
            is LogEntry.Played -> {
                val before = v.log.subList(0, index).count { it is LogEntry.Played }
                val inTrick = before % 4
                if (inTrick == 0) { trickCards = emptyList(); trickSeats = emptyList() }
                trickCards = trickCards + e.card; trickSeats = trickSeats + e.seat
                refreshTableUpTo(v, index + 1)
                if (e.belot) message = "${seatName(e.seat)}: ${declarationName(1)}!"
                if (inTrick == 3) {
                    val trick = before / 4
                    val winner = v.winners.getOrElse(trick) { -1 }
                    lastTrickCards = trickCards; lastTrickSeats = trickSeats; lastTrickWinner = winner
                    message = if (winner == HUMAN) tr("You take the trick.", "Вие вземате ръката.")
                              else tr("${seatName(winner)} takes the trick.", "${seatName(winner)} взема ръката.")
                    reactToTrick(trickCards, trickSeats, winner, v.contract)
                    delay(TRICK_HOLD_MS - COLLECT_MS)
                    if (animations) collectTo = winner
                    delay(COLLECT_MS)
                    collectTo = -1; trickCards = emptyList(); trickSeats = emptyList()
                } else if (e.seat != HUMAN) delay(CARD_SHOW_MS)
            }
        }
    }

    private fun refreshTableUpTo(v: BelotView, n: Int) { val keep = shown; shown = n; refreshTable(v); shown = keep }

    /** A resumed game: the current trick and the last finished one, straight from the record. */
    private fun restoreTable(v: BelotView) {
        val plays = v.log.filterIsInstance<LogEntry.Played>()
        val done = plays.size / 4 * 4
        val current = plays.subList(done, plays.size)
        trickCards = current.map { it.card }; trickSeats = current.map { it.seat }
        if (done >= 4) {
            val last = plays.subList(done - 4, done)
            lastTrickCards = last.map { it.card }; lastTrickSeats = last.map { it.seat }
            lastTrickWinner = v.winners.getOrElse(done / 4 - 1) { -1 }
        }
        if (v.phase == Phase.BID) hand5 = v.hand
        refreshTable(v)
    }

    // ---- the player's moves -------------------------------------------------------
    fun act(action: Int) {
        val wait = pendingHuman ?: return
        if (action !in legal) { message = tr("That is not allowed now.", "Това не е позволено сега."); return }
        legal = emptyList()
        wait.complete(action)
    }

    fun bid(bid: Int) = act(ACTION_BID + bid)
    fun declare(mask: Int) = act(ACTION_DECLARE + mask)
    fun play(card: Int) = act(card)

    /** After the score sheet: the next deal, or a new game when this one is over. */
    fun next() {
        val v = view ?: return
        if (v.phase == Phase.GAME_OVER) { start(v.match); return }
        pendingNext?.complete(Unit)
    }

    // ---- hints -------------------------------------------------------------------------
    fun toggleHint() {
        if (hints.isNotEmpty() || thinkingHint) { hints = emptyMap(); hintPick = null; return }
        val g = game ?: return
        val v = view ?: return
        if (pendingHuman == null || (v.phase != Phase.BID && v.phase != Phase.PLAY)) return
        stats = statsStore.countHint()
        val copy = g.copy()
        thinkingHint = true
        val before = message
        message = tr("Thinking about a hint…", "Мисля за подсказка…")
        scope.launch {
            val hint = withContext(worker.dispatcher) {
                worker.lock.withLock { try { copy.hint(HUMAN, maxOf(level, 1), Random.nextLong()) } finally { copy.close() } }
            }
            thinkingHint = false
            if (game !== g || pendingHuman == null) return@launch
            message = before
            if (hint == null || hint.values.isEmpty()) return@launch
            // a softmax at one game point: relative preference, not a probability of being right
            val top = hint.values.values.max()
            val w = hint.values.mapValues { exp((it.value - top) / TAU_POINTS) }
            val total = w.values.sum()
            hints = w.mapValues { (it.value / total * 100.0).toFloat() }
            hintPick = hint.pick.takeIf { it in hint.values }
        }
    }

    // ---- reactions -----------------------------------------------------------------------
    private fun react(seat: Int, vararg choices: String) {
        if (!smileys || seat == HUMAN || seat !in 0..3) return
        val emoji = choices[Random.nextInt(choices.size)]
        reactions = reactions + (seat to emoji)
        later(REACTION_MS) { if (reactions[seat] == emoji) reactions = reactions - seat }
    }

    private fun later(ms: Long, block: () -> Unit) { scope.launch { delay(ms); block() } }

    private var quietTricks = 1

    /** A few words on a trick worth a remark: a big one, a capot coming, an ace or ten given away. */
    private fun reactToTrick(cards: List<Int>, seats: List<Int>, winner: Int, contract: Int) {
        if (winner < 0) return
        val points = cards.sumOf { cardPoints(it, contract) }
        val lines = mutableListOf<Pair<Int, List<String>>>()
        // a ten or an ace (or the trump jack) thrown onto the other team's trick
        cards.forEachIndexed { i, c ->
            val s = seats[i]
            if (teamOf(s) != teamOf(winner) && cardPoints(c, contract) >= 10 && i > 0)
                lines += winner to listOf("🎁", "😋", tr("Thanks for that! 🎁", "Благодаря! 🎁"))
        }
        if (points >= 30) lines += winner to listOf("🤑", "💪", tr("What a trick!", "Каква ръка!"))
        if (winner == HUMAN && points >= 20) lines += PARTNER to listOf("👏", tr("Well played! 👏", "Браво! 👏"))
        val pick = lines.firstOrNull()
        if (pick == null || (quietTricks < 1 && points < 30) || Random.nextInt(100) >= 55) { quietTricks++; return }
        quietTricks = 0
        react(pick.first, *pick.second.toTypedArray())
    }

    /** The human's smiley or line; the bots may answer. */
    fun sendReaction(text: String) {
        reactions = reactions + (HUMAN to text)
        later(REACTION_MS) { if (reactions[HUMAN] == text) reactions = reactions - HUMAN }
        if (!smileys) return
        later(900L) {
            when (text) {
                "😏", "😎", "😂" -> react(if (Random.nextBoolean()) 1 else 3, "🙄", "😒", tr("Just wait...", "Ще видим..."))
                "👍", "👏", "🎉" -> react(PARTNER, "😄", "👍", tr("Together! 🙌", "Заедно! 🙌"))
                "😩", "🙈" -> react(PARTNER, "💪", tr("We'll get them 💪", "Ще ги бием 💪"))
                else -> if (Random.nextInt(3) == 0) react(Random.nextInt(1, 4), "🙂", "😉")
            }
        }
    }

    val playerSmileys = listOf("😀", "😏", "😎", "😂", "😅", "😩", "🙈", "👍", "👏", "🎉", "🤝", "🔥")

    // ---- the end of a deal: the books ------------------------------------------------------
    private fun dealFinished(v: BelotView) {
        dealOver = true
        val r = v.result
        message = when {
            r == null -> ""
            v.contract == 0 -> tr("Everyone passed: a new deal.", "Всички пас: нова раздавка.")
            else -> tr("Deal: us ${r.points[0]}, them ${r.points[1]}.", "Раздавка: ние ${r.points[0]}, те ${r.points[1]}.")
        }
        if (r == null || recordedDeal == v.dealNo * 2 + (if (v.match) 1 else 0)) return
        recordedDeal = v.dealNo * 2 + (if (v.match) 1 else 0)
        if (v.contract > 0) {
            val us = r.points[0]; val them = r.points[1]
            val ourContract = teamOf(v.contractSeat) == 0
            stats = statsStore.update { c ->
                c.add("hands"); if (us > them) c.add("hands_won"); if (us < them) c.add("hands_lost")
                c.add("score_for", us); c.add("score_against", them)
                c.add("points_for", r.raw[0]); c.add("points_against", r.raw[1])
                if (ourContract) { c.add("contracts_us"); if (us > 0) c.add("contracts_us_made") }
                else { c.add("contracts_them"); if (them == 0) c.add("contracts_them_set") }
                if (v.contractSeat == HUMAN) { c.add("contracts_mine"); if (us > 0) c.add("contracts_mine_made") }
                if (r.capot) { if (us > them) c.add("capots_for") else c.add("capots_against") }
                r.declared.filter { it.seat == HUMAN && it.scored == 1 }.forEach { c.add("declarations_mine"); c.add("declaration_points_mine", declarationValue(it.type)) }
                c.streak("hand_streak", "best_hand_streak", us > them)
                if (v.doubling > 0) { c.add("doubled_hands"); if (us > them) c.add("doubled_won") }
            }
            // reactions to the deal
            for (seat in 1..3) {
                val mine = teamOf(seat) == 0
                val won = if (mine) us > them else them > us
                when {
                    v.phase == Phase.GAME_OVER && won -> react(seat, "🏆", tr("We won! 🎉", "Победа! 🎉"))
                    r.capot && won -> react(seat, "👑", tr("Capot! 👑", "Капо! 👑"))
                    won && Random.nextInt(3) == 0 -> react(seat, "🎉", "😄")
                    !won && Random.nextInt(4) == 0 -> react(seat, "😞", "😅")
                }
            }
        }
        if (v.phase == Phase.GAME_OVER) {
            stats = if (v.match) statsStore.recordResult(true, v.score[0], v.score[1])
                    else statsStore.recordResult(false, r.points[0], r.points[1])
            saves.clear(); savedGame = null
        } else persist()
    }

    // ---- the saved game -----------------------------------------------------------------
    private fun persist() {
        val g = game ?: return
        val save = buildJsonObject {
            put("engine", g.save())
            put("names", JsonArray(botNames.map { JsonPrimitive(it) }))
            val v = view
            put("match", v?.match ?: true); put("us", v?.score?.get(0) ?: 0); put("them", v?.score?.get(1) ?: 0)
            put("deal", v?.dealNo ?: 1)
        }
        saves.write(save)
    }

    private fun saveSummary(): String? {
        val j = saves.read() ?: return null
        val kind = if (j.optBoolean("match")) tr("Game ${j.optInt("us")}–${j.optInt("them")}", "Игра ${j.optInt("us")}–${j.optInt("them")}")
                   else tr("Single deal", "Една раздавка")
        return tr("$kind · deal ${j.optInt("deal")}", "$kind · раздавка ${j.optInt("deal")}")
    }

    override fun close() {
        stop(); scope.cancel()
    }

    private companion object {
        const val BID_SHOW_MS = 450L
        const val DECL_SHOW_MS = 700L
        const val CARD_SHOW_MS = 250L
        const val TRICK_HOLD_MS = 1400L
        const val COLLECT_MS = 520L
        const val REACTION_MS = 2600L
        /** Hint softmax temperature, in game points. */
        const val TAU_POINTS = 1.0
    }
}
