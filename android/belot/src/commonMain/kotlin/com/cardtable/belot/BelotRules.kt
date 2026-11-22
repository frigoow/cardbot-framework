package com.cardtable.belot

import com.cardtable.platform.Deck
import com.cardtable.platform.PlayingCard
import com.cardtable.platform.Rank
import com.cardtable.platform.Suit
import com.cardtable.platform.tr

// Names and orders the screens need: the engine decides everything else.

/** Belot's card numbers: suit * 8 + rank, suits ♣ ♦ ♥ ♠, ranks 7 8 9 10 J Q K A. */
val BelotDeck = Deck { card -> PlayingCard(Suit.entries[card / 8], Rank.entries[card % 8]) }

fun suitOf(card: Int) = card / 8

/** The trump suit of a contract (1..4), or null (none yet, no trumps, all trumps). */
fun trumpSuitOf(contract: Int): Int? = if (contract in 1..4) contract - 1 else null

/** Order within a suit, high first: trump order J 9 A 10 K Q 8 7 or plain order A 10 K Q J 9 8 7. */
private val TRUMP_ORDER = listOf(4, 2, 7, 3, 6, 5, 1, 0)
private val PLAIN_ORDER = listOf(7, 3, 6, 5, 4, 2, 1, 0)

/** Suits on screen alternate colours: ♠ ♥ ♣ ♦. */
private val DISPLAY_SUITS = listOf(3, 2, 0, 1)

/** The hand as it is held: by suit, each suit strongest first under the contract. */
fun sortHand(cards: List<Int>, contract: Int): List<Int> = cards.sortedWith(compareBy({ DISPLAY_SUITS.indexOf(it / 8) }, { c ->
    val trump = contract == 6 || trumpSuitOf(contract) == c / 8
    (if (trump) TRUMP_ORDER else PLAIN_ORDER).indexOf(c % 8)
}))

/** Card points of a card under a contract (for the last-trick sheet). */
fun cardPoints(card: Int, contract: Int): Int {
    val r = card % 8
    val trump = contract == 6 || trumpSuitOf(contract) == card / 8
    return if (trump) intArrayOf(0, 0, 14, 10, 20, 3, 4, 11)[r] else intArrayOf(0, 0, 0, 10, 2, 3, 4, 11)[r]
}

private val SUIT_SYMBOLS = listOf("♣", "♦", "♥", "♠")

fun suitName(suit: Int) = listOf(tr("Clubs", "Спатия"), tr("Diamonds", "Каро"), tr("Hearts", "Купа"), tr("Spades", "Пика"))[suit]

/** A bid (0..8) as said at the table. */
fun bidLabel(bid: Int): String = when (bid) {
    0 -> tr("Pass", "Пас")
    in 1..4 -> SUIT_SYMBOLS[bid - 1] + "︎ " + suitName(bid - 1)
    5 -> tr("No trumps", "Без коз")
    6 -> tr("All trumps", "Всичко коз")
    7 -> tr("Double", "Контра")
    else -> tr("Redouble", "Реконтра")
}

/** A contract (1..6) with its doubling, short: "♠ ×2", "No trumps ×4". */
fun contractLabel(contract: Int, doubling: Int): String {
    val base = when (contract) {
        in 1..4 -> SUIT_SYMBOLS[contract - 1] + "︎ " + suitName(contract - 1)
        5 -> tr("No trumps", "Без коз")
        6 -> tr("All trumps", "Всичко коз")
        else -> "—"
    }
    return base + when (doubling) { 1 -> " ×2"; 2 -> " ×4"; else -> "" }
}

/** A declaration type (the engine's AnnounceType) as said at the table. */
fun declarationName(type: Int): String = when (type) {
    1 -> tr("Belot", "Белот")
    2 -> tr("Tierce", "Терца")
    3 -> tr("Quarte", "Кварта")
    in 4..7 -> tr("Quinte", "Квинта")
    8 -> tr("Four of a kind", "Каре")
    9 -> tr("Four nines", "Каре деветки")
    10 -> tr("Four jacks", "Каре валета")
    else -> "?"
}

fun declarationValue(type: Int): Int = when (type) {
    1, 2 -> 20; 3 -> 50; in 4..8 -> 100; 9 -> 150; 10 -> 200; else -> 0
}

/** "Bots" strength levels (BOT_LEVELS). */
fun levelLabel(level: Int) = when (level) {
    0 -> tr("Easy", "Лесно")
    1 -> tr("Normal", "Нормално")
    else -> tr("Strong", "Силно")
}

/** The bots' names at a Bulgarian table (the shells set AppConfig.botNames to these). */
val BULGARIAN_BOT_NAMES = listOf("Иван", "Мария", "Георги", "Елена", "Петър", "Десислава", "Николай", "Йоана", "Димитър", "Радост")

/** The same names in Latin letters, for the English screens. */
private val LATIN_NAMES = mapOf("Иван" to "Ivan", "Мария" to "Maria", "Георги" to "Georgi", "Елена" to "Elena", "Петър" to "Petar",
    "Десислава" to "Desislava", "Николай" to "Nikolay", "Йоана" to "Yoana", "Димитър" to "Dimitar", "Радост" to "Radost")

/** A bot's name as the current language writes it. */
fun displayName(name: String): String = if (com.cardtable.platform.Lang.code == "en") LATIN_NAMES[name] ?: name else name
