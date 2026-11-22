package com.cardtable.platform

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue

/**
 * English and the game's own language (Dutch for Manille, Bulgarian for
 * Belot). Each app writes its strings as tr(english, local) at their point of
 * use, so the two stay side by side and a missing translation is visible in
 * review. The shared screens of this module know every app's language and
 * use tl(english, dutch, bulgarian). The language is observable state:
 * switching it recomposes the screen at once (messages already built keep
 * their language until the next one).
 */
object Lang {
    /** "en", "nl" or "bg". */
    var code by mutableStateOf("en")
    /** Not English: the app's own language. */
    val local: Boolean get() = code != "en"
}

/** An app string: English, or the app's own language. */
fun tr(en: String, local: String): String = if (Lang.code == "en") en else local

/** A string of the shared screens, in every app's language. */
fun tl(en: String, nl: String, bg: String): String = when (Lang.code) {
    "nl" -> nl
    "bg" -> bg
    else -> en
}
