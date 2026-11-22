package com.cardtable.belot

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.zIndex
import com.cardtable.platform.CardBack
import com.cardtable.platform.CardFace
import com.cardtable.platform.Chip
import com.cardtable.platform.ChoiceChip
import com.cardtable.platform.FeltTable
import com.cardtable.platform.Felt0
import com.cardtable.platform.Felt1
import com.cardtable.platform.FloatingPrompt
import com.cardtable.platform.Gold
import com.cardtable.platform.GoldBright
import com.cardtable.platform.Ink
import com.cardtable.platform.InkDim
import com.cardtable.platform.LobbyCard
import com.cardtable.platform.LocalAppTheme
import com.cardtable.platform.LocalDeck
import com.cardtable.platform.ModeButton
import com.cardtable.platform.Panel
import com.cardtable.platform.ProfileDialog
import com.cardtable.platform.ReactionBubble
import com.cardtable.platform.ScoreBox
import com.cardtable.platform.SpeechBubble
import com.cardtable.platform.StatCell
import com.cardtable.platform.StatLine
import com.cardtable.platform.StatSection
import com.cardtable.platform.Suit
import com.cardtable.platform.TableDecor
import com.cardtable.platform.Tail
import com.cardtable.platform.ThemePicker
import com.cardtable.platform.Wood
import com.cardtable.platform.glideMs
import com.cardtable.platform.suitTint
import com.cardtable.platform.tableBackdrop
import com.cardtable.platform.tr
import kotlin.math.cos
import kotlin.math.roundToInt

private val ThemColor = Color(0xFFE07A6E)

/**
 * The whole Belot app: the platform shell (Android: MainActivity, desktop:
 * Main.kt) calls this with its [BelotPlatform]; [onController] sees the
 * controller once it exists (screenshots, debugging).
 */
@Composable
fun BelotRoot(platform: BelotPlatform, onController: (BelotController) -> Unit = {}) {
    MaterialTheme {
        val controller = remember { BelotController(platform).also(onController) }
        DisposableEffect(controller) { onDispose { controller.close() } }
        // the table is laid out to exact text sizes: cap a large system font scale
        val d = LocalDensity.current
        CompositionLocalProvider(
            LocalDensity provides Density(d.density, d.fontScale.coerceAtMost(1.15f)),
            LocalAppTheme provides controller.theme,
            LocalDeck provides BelotDeck,
        ) {
            var screen by remember { mutableStateOf("lobby") }
            Box(Modifier.fillMaxSize().tableBackdrop().systemBarsPadding()) {
                when {
                    controller.inGame -> GameScene(controller)
                    screen == "stats" -> StatsScreen(controller) { screen = "lobby" }
                    else -> Lobby(controller) { screen = "stats" }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- lobby ----

@Composable
private fun Lobby(c: BelotController, openStats: () -> Unit) {
    var profileDialog by remember { mutableStateOf(false) }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val landscape = maxWidth > maxHeight
        val title = @Composable {
            Text(tr("BELOT", "БЕЛОТ"), color = GoldBright, fontSize = if (landscape) 32.sp else 42.sp, fontWeight = FontWeight.ExtraBold, letterSpacing = 7.sp)
            Text(tr("Bulgarian bridge-belot, against strong bots.", "Бридж-белот срещу силни ботове."), color = InkDim, fontSize = 13.sp)
        }
        val who = @Composable {
            LobbyCard("1", tr("WHO IS PLAYING?  (tap to change)", "КОЙ ИГРАЕ?  (докоснете за смяна)"), c.profileName,
                tr("${c.profiles.size} profile(s) on this device", "Профили на устройството: ${c.profiles.size}")) { profileDialog = true }
            Spacer(Modifier.height(10.dp))
            LobbyCard("2", tr("YOUR OPPONENTS", "ВАШИТЕ ПРОТИВНИЦИ"), tr("Bots · ${levelLabel(c.level)}", "Ботове · ${levelLabel(c.level)}"),
                tr("Neural bidding, card counting + play-outs · offline", "Невронно обявяване, броене на карти + разигравания · офлайн"))
        }
        val play = @Composable {
            c.savedGame?.let { summary ->
                Row(Modifier.widthIn(max = 480.dp)) {
                    ModeButton(tr("Resume", "Продължи"), summary, Modifier.weight(1f)) { c.resume() }
                }
                Spacer(Modifier.height(10.dp))
            }
            Row(Modifier.widthIn(max = 480.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                ModeButton(tr("Single deal", "Една раздавка"), tr("One deal on its own", "Само една раздавка"), Modifier.weight(1f)) { c.start(false) }
                ModeButton(tr("Game to 151", "Игра до 151"), tr("Keep score across deals", "Резултат през раздавките"), Modifier.weight(1f)) { c.start(true) }
            }
            Spacer(Modifier.height(10.dp))
            StatsSummary(c, openStats)
            Spacer(Modifier.height(10.dp))
            Settings(c)
        }
        if (landscape) {
            Row(Modifier.fillMaxSize().padding(horizontal = 24.dp, vertical = 12.dp), horizontalArrangement = Arrangement.spacedBy(24.dp)) {
                Column(Modifier.weight(1f).fillMaxHeight().verticalScroll(rememberScrollState()),
                    horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                    title(); Spacer(Modifier.height(18.dp)); who()
                }
                Column(Modifier.weight(1f).fillMaxHeight().verticalScroll(rememberScrollState()),
                    horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) { play() }
            }
        } else {
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 24.dp, vertical = 30.dp),
                horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
                title(); Spacer(Modifier.height(28.dp)); who(); Spacer(Modifier.height(12.dp)); play()
                Spacer(Modifier.height(20.dp))
                Text(tr("All game rules and bots run on your device.", "Всички правила и ботове работят на устройството ви."), color = InkDim, fontSize = 11.sp)
            }
        }
    }
    if (profileDialog) ProfileDialog(c.profiles, c.activeProfileId, { profileDialog = false },
        c::selectProfile, c::createProfile, c::renameProfile, c::deleteProfile)
}

@Composable
private fun Settings(c: BelotController) = Surface(Modifier.widthIn(max = 480.dp).fillMaxWidth(), shape = RoundedCornerShape(16.dp), color = Panel,
    border = BorderStroke(1.dp, Gold.copy(alpha = .25f))) {
    Column(Modifier.padding(horizontal = 14.dp, vertical = 10.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(tr("Language", "Език"), color = Ink, fontSize = 13.sp, modifier = Modifier.weight(1f))
            ChoiceChip("Български", !c.english) { c.chooseEnglish(false) }
            Spacer(Modifier.width(6.dp))
            ChoiceChip("English", c.english) { c.chooseEnglish(true) }
        }
        Spacer(Modifier.height(6.dp))
        Text(tr("Bot strength", "Сила на ботовете"), color = Ink, fontSize = 13.sp)
        Row(Modifier.horizontalScroll(rememberScrollState()).padding(top = 4.dp)) {
            for (l in 0 until BOT_LEVELS) {
                if (l > 0) Spacer(Modifier.width(6.dp))
                ChoiceChip(levelLabel(l), c.level == l) { c.chooseLevel(l) }
            }
        }
        val mean = c.botMsMean
        Text(if (mean == null) tr("Thinking time: play a few cards to measure it", "Време за мислене: изиграйте няколко карти, за да се измери")
             else tr("Thinking time: $mean ms per card", "Време за мислене: $mean ms на карта"),
            color = InkDim, fontSize = 11.sp, modifier = Modifier.padding(top = 4.dp))
        Spacer(Modifier.height(6.dp))
        SwitchRow(tr("Bot smileys", "Емотикони на ботовете"), tr("Bots react to the cards: 😎 🙈 👍 🎉", "Ботовете реагират на играта: 😎 🙈 👍 🎉"), c.smileys, c::enableSmileys)
        SwitchRow(tr("Card animations", "Анимации на картите"), tr("Cards glide onto the table and to the trick's winner", "Картите се плъзгат към масата и към взелия ръката"), c.animations, c::enableAnimations)
        SwitchRow(tr("Four-colour deck", "Четирицветно тесте"), tr("Clubs and diamonds in their own colours", "Спатия и каро в свои цветове"), c.fourColor, c::setFourColorDeck)
        Spacer(Modifier.height(8.dp))
        ThemePicker(c.theme.id, c::chooseTheme)
    }
}

@Composable
private fun SwitchRow(title: String, sub: String, on: Boolean, set: (Boolean) -> Unit) = Row(Modifier.fillMaxWidth().padding(vertical = 2.dp), verticalAlignment = Alignment.CenterVertically) {
    Column(Modifier.weight(1f)) {
        Text(title, color = Ink, fontSize = 13.sp)
        Text(sub, color = InkDim, fontSize = 11.sp)
    }
    Switch(checked = on, onCheckedChange = set)
}

@Composable
private fun StatsSummary(c: BelotController, open: () -> Unit) = Surface(
    modifier = Modifier.fillMaxWidth().widthIn(max = 480.dp).clickable(onClick = open), color = Panel,
    shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, Color.White.copy(alpha = .12f)),
) {
    val s = c.stats
    Column(Modifier.padding(16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(tr("YOUR RECORD", "ВАШИТЕ РЕЗУЛТАТИ"), color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp, modifier = Modifier.weight(1f))
            Text(tr("All statistics ›", "Цялата статистика ›"), color = GoldBright, fontSize = 11.sp)
        }
        if (s["hands"] == 0) {
            Text(tr("No deals played yet.", "Още няма изиграни раздавки."), color = Ink, fontSize = 13.sp)
        } else {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                StatCell(tr("Games", "Игри"), "${s["matches_won"]}–${s["matches_lost"]}")
                StatCell(tr("Deals won", "Спечелени"), s.rate("hands_won", "hands"))
                StatCell(tr("Contracts made", "Изкарани игри"), s.rate("contracts_us_made", "contracts_us"))
                StatCell(tr("Pts / deal", "Т. / раздавка"), s.avg("score_for", "hands"))
            }
        }
    }
}

@Composable
private fun StatsScreen(c: BelotController, back: () -> Unit) {
    val s = c.stats
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 16.dp),
        horizontalAlignment = Alignment.CenterHorizontally) {
        Row(Modifier.widthIn(max = 560.dp).fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(tr("‹ Back", "‹ Назад"), color = GoldBright, fontSize = 15.sp, modifier = Modifier.clickable(onClick = back).padding(8.dp))
            Spacer(Modifier.weight(1f))
            Text(c.profileName.uppercase(), color = GoldBright, fontWeight = FontWeight.ExtraBold, letterSpacing = 2.sp)
        }
        if (s["hands"] == 0) {
            Spacer(Modifier.height(40.dp)); Text(tr("No deals played yet.", "Още няма изиграни раздавки."), color = Ink); return@Column
        }
        StatSection(tr("Games", "Игри")) {
            StatLine(tr("Games to 151 won – lost", "Игри до 151 спечелени – загубени"), "${s["matches_won"]} – ${s["matches_lost"]}", s.rate("matches_won", "matches"))
            StatLine(tr("Single deals won – lost", "Единични раздавки спечелени – загубени"), "${s["games_won"]} – ${s["games_lost"]}", s.rate("games_won", "games"))
            StatLine(tr("Best winning margin", "Най-голяма разлика"), "${s["best_match_margin"]}")
            StatLine(tr("Games won in a row (best)", "Поредни победи (рекорд)"), "${s["match_streak"]} (${s["best_match_streak"]})")
        }
        StatSection(tr("Deals", "Раздавки")) {
            StatLine(tr("Deals played", "Изиграни раздавки"), "${s["hands"]}")
            StatLine(tr("Won – lost", "Спечелени – загубени"), "${s["hands_won"]} – ${s["hands_lost"]}", s.rate("hands_won", "hands"))
            StatLine(tr("Deals won in a row (best)", "Поредни спечелени (рекорд)"), "${s["hand_streak"]} (${s["best_hand_streak"]})")
            StatLine(tr("Game points per deal (us – them)", "Точки на раздавка (ние – те)"), "${s.avg("score_for", "hands")} – ${s.avg("score_against", "hands")}")
            StatLine(tr("Capots (for – against)", "Капо (за – срещу)"), "${s["capots_for"]} – ${s["capots_against"]}")
        }
        StatSection(tr("Contracts", "Игри")) {
            StatLine(tr("Your team played the contract", "Вашият отбор игра"), "${s["contracts_us"]}", s.rate("contracts_us_made", "contracts_us"))
            StatLine(tr("  …you bid it", "  …вие обявихте"), "${s["contracts_mine"]}", s.rate("contracts_mine_made", "contracts_mine"))
            StatLine(tr("Opponents' contracts you set", "Вкарани игри на противниците"), "${s["contracts_them_set"]} / ${s["contracts_them"]}", s.rate("contracts_them_set", "contracts_them"))
            StatLine(tr("Doubled deals won", "Спечелени с контра"), "${s["doubled_won"]} / ${s["doubled_hands"]}", s.rate("doubled_won", "doubled_hands"))
        }
        StatSection(tr("Other", "Други")) {
            StatLine(tr("Your declarations that scored", "Ваши обявени комбинации"), "${s["declarations_mine"]}", "${s["declaration_points_mine"]} " + tr("pts", "т."))
            StatLine(tr("Hints used", "Използвани подсказки"), "${s["hints_used"]}")
        }
        if (s.history.isNotEmpty()) StatSection(tr("Last results", "Последни резултати")) {
            s.history.takeLast(12).reversed().forEach { h ->
                val us = (h["us"] as? kotlinx.serialization.json.JsonPrimitive)?.content?.toIntOrNull() ?: 0
                val them = (h["them"] as? kotlinx.serialization.json.JsonPrimitive)?.content?.toIntOrNull() ?: 0
                val match = (h["mode"] as? kotlinx.serialization.json.JsonPrimitive)?.content == "match"
                StatLine(if (match) tr("Game to 151", "Игра до 151") else tr("Single deal", "Една раздавка"), "$us – $them",
                    if (us > them) tr("won", "победа") else if (us < them) tr("lost", "загуба") else tr("tied", "равно"))
            }
        }
        Spacer(Modifier.height(20.dp))
    }
}

// ----------------------------------------------------------------- game ----

private class Sizes(val handW: Dp, val trickW: Dp, val seatCardW: Dp, val compactSeats: Boolean) {
    val handH get() = handW * 1.45f
    val trickH get() = trickW * 1.45f
}

@Composable
private fun GameScene(c: BelotController) {
    val v = c.view ?: return
    var confirmRestart by remember { mutableStateOf(false) }
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val landscape = maxWidth > maxHeight
        if (landscape) {
            val panelW = 196.dp
            val leftW = maxWidth - panelW
            val handW = minOf(120.dp, maxHeight * .27f / 1.45f, (leftW - 24.dp) / 5.7f)
            val tableH = maxHeight - handW * 1.45f - 34.dp
            val trickW = minOf(handW * .86f, (tableH - 64.dp) / 2f / 1.45f)
            val sizes = Sizes(handW, trickW, trickW * .55f, compactSeats = true)
            Row(Modifier.fillMaxSize()) {
                Column(Modifier.width(leftW).fillMaxHeight()) {
                    Table(c, v, sizes, Modifier.fillMaxWidth().weight(1f))
                    Hand(c, sizes, leftW, Modifier.fillMaxWidth().padding(bottom = 4.dp))
                }
                SidePanel(c, v, Modifier.width(panelW).fillMaxHeight()) { confirmRestart = true }
            }
        } else {
            val compact = maxWidth < 390.dp || maxHeight < 700.dp
            val handW = minOf(78.dp, (maxWidth - 24.dp) / 5.7f)
            val trickW = if (compact) 52.dp else 60.dp
            val sizes = Sizes(handW, trickW, trickW * .55f, compactSeats = false)
            val fullW = maxWidth
            Column(Modifier.fillMaxSize()) {
                TopBar(c, v) { confirmRestart = true }
                DealBar(c, v)
                Table(c, v, sizes, Modifier.fillMaxWidth().weight(1f).padding(top = 6.dp))
                BottomBar(c, Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 4.dp))
                Hand(c, sizes, fullW, Modifier.fillMaxWidth().padding(bottom = 6.dp))
            }
        }
        if (c.dealOver) DealSheet(c, v)
        if (confirmRestart) AlertDialog(
            onDismissRequest = { confirmRestart = false },
            confirmButton = { TextButton(onClick = { confirmRestart = false; c.restart() }) { Text(tr("Restart", "Отначало")) } },
            dismissButton = { TextButton(onClick = { confirmRestart = false }) { Text(tr("Cancel", "Отказ")) } },
            title = { Text(tr("Restart?", "Отначало?")) },
            text = { Text(if (v.match) tr("Start a new game from 0–0? This game is abandoned.", "Нова игра от 0–0? Текущата игра се прекратява.")
                          else tr("Deal again? This deal is abandoned.", "Ново раздаване? Тази раздавка се прекратява.")) },
        )
    }
}

@Composable
private fun TopBar(c: BelotController, v: BelotView, restart: () -> Unit) = Surface(
    modifier = Modifier.fillMaxWidth(), color = Color(0xB8000000), border = BorderStroke(1.dp, Color.White.copy(alpha = .08f)),
) {
    Row(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp), verticalAlignment = Alignment.CenterVertically) {
        GameMenu(c, restart)
        Spacer(Modifier.width(10.dp))
        Scoreboard(v, Modifier.weight(1f))
        Spacer(Modifier.width(14.dp))
        ContractChip(c, v, 48.dp)
        Spacer(Modifier.width(4.dp))
    }
}

@Composable
private fun Scoreboard(v: BelotView, modifier: Modifier, big: Boolean = true) = Row(modifier, verticalAlignment = Alignment.CenterVertically) {
    ScoreBox(tr("US", "НИЕ"), v.score[0], GoldBright, Modifier.weight(1f), big, WINNING_SCORE)
    Spacer(Modifier.width(8.dp))
    ScoreBox(tr("THEM", "ТЕ"), v.score[1], ThemColor, Modifier.weight(1f), big, WINNING_SCORE)
}

/** The contract (or ? during the auction) with its doubling. */
@Composable
private fun ContractChip(c: BelotController, v: BelotView, size: Dp) = Column(horizontalAlignment = Alignment.CenterHorizontally) {
    val known = v.contract > 0 && v.phase != Phase.BID
    Box {
        Surface(modifier = Modifier.size(size), shape = CircleShape, color = Color(0xFFFFFDF6), border = BorderStroke(2.dp, Gold), shadowElevation = 6.dp) {
            Box(contentAlignment = Alignment.Center) {
                val trump = trumpSuitOf(v.contract)
                when {
                    !known -> Text("?", color = Wood, fontSize = 24.sp, fontWeight = FontWeight.Bold)
                    trump != null -> Text(com.cardtable.platform.suitSymbol(Suit.entries[trump]), color = suitTint(Suit.entries[trump], c.fourColor), fontSize = (size.value * .55f).sp)
                    v.contract == 5 -> Text(tr("NT", "БК"), color = Wood, fontSize = 17.sp, fontWeight = FontWeight.ExtraBold)
                    else -> Text(tr("AT", "ВК"), color = Wood, fontSize = 17.sp, fontWeight = FontWeight.ExtraBold)
                }
            }
        }
        if (known && v.doubling > 0) Surface(modifier = Modifier.align(Alignment.TopEnd).offset(x = 8.dp, y = (-6).dp),
            shape = RoundedCornerShape(8.dp), color = com.cardtable.platform.CardRed) {
            Text(if (v.doubling == 1) "×2" else "×4", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.ExtraBold,
                modifier = Modifier.padding(horizontal = 5.dp, vertical = 1.dp))
        }
    }
    Text(if (!known) tr("BIDDING", "ОБЯВЯВАНЕ") else tr("CONTRACT", "ИГРА"), color = InkDim, fontSize = 8.sp, letterSpacing = 1.sp)
}

/** This deal: its number, the points taken so far, and the hanging points. */
@Composable
private fun DealBar(c: BelotController, v: BelotView) = Surface(modifier = Modifier.fillMaxWidth(), color = Color(0x99000000)) {
    Row(Modifier.padding(horizontal = 14.dp, vertical = 5.dp), verticalAlignment = Alignment.CenterVertically) {
        Text(if (v.match) tr("DEAL ${v.dealNo}", "РАЗДАВКА ${v.dealNo}") else tr("SINGLE DEAL", "ЕДНА РАЗДАВКА"), color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp)
        Spacer(Modifier.weight(1f))
        Text(tr("Us ", "Ние "), color = GoldBright, fontSize = 13.sp, fontWeight = FontWeight.Bold)
        Text("${v.cardPoints[0]}", color = Ink, fontSize = 18.sp, fontWeight = FontWeight.ExtraBold)
        Text("  –  ", color = InkDim, fontSize = 14.sp)
        Text("${v.cardPoints[1]}", color = Ink, fontSize = 18.sp, fontWeight = FontWeight.ExtraBold)
        Text(tr(" Them", " Те"), color = ThemColor, fontSize = 13.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.weight(1f))
        Text(if (v.hanging > 0) tr("hanging ${v.hanging}", "висящи ${v.hanging}") else tr("card points", "точки от карти"), color = InkDim, fontSize = 11.sp)
    }
}

@Composable
private fun BottomBar(c: BelotController, modifier: Modifier) = Row(
    modifier.then(if (LocalAppTheme.current.style == "cafe") Modifier.background(Panel, RoundedCornerShape(16.dp)).padding(horizontal = 8.dp, vertical = 2.dp) else Modifier),
    verticalAlignment = Alignment.CenterVertically) {
    // always two lines tall: a message that wraps must not resize the table above
    Text(c.message, color = Ink, fontSize = 13.sp, lineHeight = 16.sp, minLines = 2, maxLines = 2,
        overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
    SmileyButton(c)
    Spacer(Modifier.width(8.dp))
    LastTrickButton(c)
    Spacer(Modifier.width(8.dp))
    HintButton(c)
}

@Composable
private fun SidePanel(c: BelotController, v: BelotView, modifier: Modifier, restart: () -> Unit) = Surface(
    modifier = modifier, color = Color(0xB8000000), border = BorderStroke(1.dp, Color.White.copy(alpha = .08f)),
) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(10.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Row(verticalAlignment = Alignment.CenterVertically) { GameMenu(c, restart); Spacer(Modifier.width(12.dp)); ContractChip(c, v, 44.dp) }
        Spacer(Modifier.height(8.dp))
        Scoreboard(v, Modifier.fillMaxWidth(), big = false)
        Spacer(Modifier.height(8.dp))
        Text(tr("Cards: us ${v.cardPoints[0]} – ${v.cardPoints[1]} them", "Карти: ние ${v.cardPoints[0]} – ${v.cardPoints[1]} те"), color = Ink, fontSize = 12.sp)
        if (v.hanging > 0) Text(tr("Hanging: ${v.hanging}", "Висящи: ${v.hanging}"), color = GoldBright, fontSize = 12.sp)
        Spacer(Modifier.height(8.dp))
        Text(c.message, color = Ink, fontSize = 12.sp, lineHeight = 15.sp, textAlign = TextAlign.Center,
            minLines = 3, maxLines = 3, overflow = TextOverflow.Ellipsis)
        Spacer(Modifier.height(8.dp))
        // stacked: the Bulgarian labels are too long for one row in the panel
        HintButton(c)
        Spacer(Modifier.height(6.dp))
        LastTrickButton(c)
        Spacer(Modifier.height(8.dp))
        SmileyButton(c)
    }
}

@Composable
private fun GameMenu(c: BelotController, restart: () -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        Surface(modifier = Modifier.size(40.dp).clickable { open = true }, shape = CircleShape,
            color = Color.White.copy(alpha = .08f), border = BorderStroke(1.dp, Color.White.copy(alpha = .18f))) {
            Box(contentAlignment = Alignment.Center) { Text("☰", color = Ink, fontSize = 18.sp) }
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            DropdownMenuItem(text = { Text(if (c.showOpenCards) tr("Hide all cards", "Скрий всички карти") else tr("Show all cards", "Покажи всички карти")) },
                onClick = { open = false; c.toggleOpenCards() })
            DropdownMenuItem(text = { Text(tr("Bot strength: ", "Сила на ботовете: ") + levelLabel(c.level)) },
                onClick = { c.chooseLevel((c.level + 1) % BOT_LEVELS) })
            DropdownMenuItem(text = { Text(if (c.animations) tr("Animations: on", "Анимации: вкл.") else tr("Animations: off", "Анимации: изкл.")) },
                onClick = { c.enableAnimations(!c.animations) })
            DropdownMenuItem(text = { Text(if (c.smileys) tr("Smileys: on", "Емотикони: вкл.") else tr("Smileys: off", "Емотикони: изкл.")) },
                onClick = { c.enableSmileys(!c.smileys) })
            DropdownMenuItem(text = { Text(if (c.english) "Език: български" else "Language: English") }, onClick = { c.chooseEnglish(!c.english) })
            DropdownMenuItem(text = { Text(tr("Restart", "Отначало")) }, onClick = { open = false; restart() })
            DropdownMenuItem(text = { Text(tr("Back to the lobby", "Към началото")) }, onClick = { open = false; c.lobby() })
        }
    }
}

@Composable
private fun SmileyButton(c: BelotController) {
    var open by remember { mutableStateOf(false) }
    Box {
        Surface(modifier = Modifier.size(38.dp).clickable { open = true }, shape = CircleShape,
            color = Color.White.copy(alpha = .12f), border = BorderStroke(1.dp, Gold.copy(alpha = .7f))) {
            Box(contentAlignment = Alignment.Center) { Text("😀", fontSize = 18.sp) }
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            c.playerSmileys.chunked(6).forEach { row ->
                Row(Modifier.padding(horizontal = 6.dp)) {
                    row.forEach { e -> Text(e, fontSize = 26.sp, modifier = Modifier.clickable { open = false; c.sendReaction(e) }.padding(5.dp)) }
                }
            }
        }
    }
}

@Composable
private fun LastTrickButton(c: BelotController) {
    var open by remember { mutableStateOf(false) }
    val enabled = c.lastTrickCards.isNotEmpty()
    Surface(
        modifier = Modifier.height(38.dp).then(if (enabled) Modifier.clickable { open = true } else Modifier),
        shape = RoundedCornerShape(19.dp), color = Color.White.copy(alpha = if (enabled) .12f else .05f),
        border = BorderStroke(1.dp, if (enabled) Gold.copy(alpha = .7f) else Color.White.copy(alpha = .1f)),
    ) {
        Row(Modifier.padding(horizontal = 12.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(tr("Last trick", "Последна ръка"), color = if (enabled) Ink else InkDim.copy(alpha = .5f), fontWeight = FontWeight.Bold, fontSize = 13.sp)
        }
    }
    if (open) LastTrickDialog(c) { open = false }
}

@Composable
private fun LastTrickDialog(c: BelotController, close: () -> Unit) {
    val cards = c.lastTrickCards; val seats = c.lastTrickSeats
    val contract = c.view?.contract ?: 0
    val points = cards.sumOf { cardPoints(it, contract) }
    Dialog(onDismissRequest = close) {
        Surface(modifier = Modifier.clickable(onClick = close), shape = RoundedCornerShape(18.dp),
            color = Felt1, border = BorderStroke(1.5.dp, Gold.copy(alpha = .8f)), shadowElevation = 14.dp) {
            Column(Modifier.padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(tr("LAST TRICK", "ПОСЛЕДНА РЪКА"), color = GoldBright, fontWeight = FontWeight.ExtraBold, letterSpacing = 2.sp)
                Spacer(Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    cards.zip(seats).forEach { (card, s) ->
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            CardFace(card, c.fourColor, Modifier, 58.dp, 84.dp, recommended = s == c.lastTrickWinner)
                            Text(c.seatName(s), color = if (s == c.lastTrickWinner) GoldBright else Ink, fontSize = 12.sp,
                                fontWeight = if (s == c.lastTrickWinner) FontWeight.Bold else FontWeight.Normal)
                        }
                    }
                }
                Spacer(Modifier.height(8.dp))
                Text(tr("${c.seatName(c.lastTrickWinner)} took it · $points points", "${c.seatName(c.lastTrickWinner)} я взе · $points точки"), color = Ink, fontSize = 13.sp)
                Text(tr("tap to close", "докоснете, за да затворите"), color = InkDim, fontSize = 10.sp)
            }
        }
    }
}

@Composable
private fun HintButton(c: BelotController) {
    val v = c.view
    val enabled = c.legal.isNotEmpty() && v != null && (v.phase == Phase.BID || v.phase == Phase.PLAY)
    val showing = c.hints.isNotEmpty() || c.thinkingHint
    Surface(
        modifier = Modifier.height(38.dp).then(if (enabled) Modifier.clickable(onClick = c::toggleHint) else Modifier),
        shape = RoundedCornerShape(19.dp), color = if (enabled) GoldBright else Color.White.copy(alpha = .08f),
        border = BorderStroke(1.dp, if (enabled) Gold else Color.White.copy(alpha = .12f)),
    ) {
        Row(Modifier.padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Text(if (showing) tr("Hide hint", "Скрий") else tr("Hint", "Подсказка"), color = if (enabled) Wood else InkDim.copy(alpha = .5f), fontWeight = FontWeight.Bold, fontSize = 14.sp)
        }
    }
}

// ---------------------------------------------------------------- table ----

/** Where each seat sits on screen: the player at the bottom, play going counter-clockwise (East on the right). */
private enum class Side { BOTTOM, RIGHT, TOP, LEFT }
private fun sideOf(seat: Int) = Side.entries[seat]

@Composable
private fun Table(c: BelotController, v: BelotView, sizes: Sizes, modifier: Modifier) {
    FeltTable(modifier) {
        val cafe = LocalAppTheme.current.style == "cafe"
        if (cafe) TableDecor(0)
        val myBubbleX = if (cafe) 70.dp else 10.dp
        val seatH = if (sizes.compactSeats) 52.dp else 34.dp + 14.dp + sizes.seatCardW * 1.45f + 8.dp
        val seatW = if (sizes.compactSeats) 64.dp else 60.dp
        Seat(c, v, 2, Modifier.align(Alignment.TopCenter).offset(y = 4.dp), horizontal = true, sizes = sizes)
        Seat(c, v, 3, Modifier.align(Alignment.CenterStart).offset(x = 4.dp), horizontal = false, sizes = sizes)
        Seat(c, v, 1, Modifier.align(Alignment.CenterEnd).offset(x = (-4).dp), horizontal = false, sizes = sizes)
        // the player's own words: bottom left of the table
        Column(Modifier.align(Alignment.BottomStart).offset(x = myBubbleX, y = (-10).dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
            val mine = c.reactions[HUMAN]
            AnimatedVisibility(visible = mine != null, enter = scaleIn() + fadeIn(), exit = scaleOut() + fadeOut()) { ReactionBubble(mine ?: "") }
            // hidden while the player is asked something: the prompt sits there
            val asked = c.legal.isNotEmpty() && (v.phase == Phase.BID || v.phase == Phase.DECLARE)
            if (!asked) c.bubbles[HUMAN].orEmpty().forEachIndexed { i, b -> SpeechBubble(b.text, bubbleColor(c, b), if (i == c.bubbles[HUMAN]!!.lastIndex) Tail.BottomStart else null) }
        }
        val dv = minOf(sizes.trickH * .62f, maxHeight / 2 - seatH - 6.dp - sizes.trickH / 2).coerceAtLeast(sizes.trickH * .3f)
        val dh = minOf(sizes.trickW * 1.05f + 4.dp, maxWidth / 2 - seatW - 10.dp - sizes.trickW / 2).coerceAtLeast(sizes.trickW * .6f)
        val seatAt = mapOf(
            Side.TOP to DpOffset(0.dp, -(maxHeight / 2 - 30.dp)), Side.LEFT to DpOffset(-(maxWidth / 2 - 30.dp), 0.dp),
            Side.RIGHT to DpOffset(maxWidth / 2 - 30.dp, 0.dp), Side.BOTTOM to DpOffset(0.dp, maxHeight / 2 + 60.dp))
        val home = mapOf(Side.TOP to DpOffset(0.dp, -dv), Side.LEFT to DpOffset(-dh, 0.dp), Side.RIGHT to DpOffset(dh, 0.dp), Side.BOTTOM to DpOffset(0.dp, dv))
        c.trickCards.zip(c.trickSeats).forEach { (card, seat) ->
            val h = home.getValue(sideOf(seat))
            TrickCard(c, card, seat, Modifier.align(Alignment.Center).offset(x = h.x, y = h.y), sizes, h, seatAt)
        }
        // the auction and the declarations: at the bottom of the table, below the side seats
        // (the player's own spot in the trick is empty while they are asked)
        Box(Modifier.align(Alignment.BottomCenter).offset(y = (-6).dp).widthIn(max = maxWidth - 16.dp)) { TablePrompt(c, v) }
    }
}

@Composable
private fun bubbleColor(c: BelotController, b: Bubble): Color = when {
    b.suit != null -> suitTint(Suit.entries[b.suit], c.fourColor)
    b.loud -> com.cardtable.platform.CardRed
    else -> Wood
}

@Composable
private fun Seat(c: BelotController, v: BelotView, seat: Int, modifier: Modifier, horizontal: Boolean, sizes: Sizes) {
    val active = v.toMove == seat && !c.dealOver && c.legal.isEmpty() && c.trickCards.size < 4
    val reaction = c.reactions[seat]
    val said = c.bubbles[seat].orEmpty()
    val avatar = @Composable {
        Box {
            Surface(modifier = Modifier.size(34.dp), shape = CircleShape, color = if (active) Gold else Felt1,
                border = BorderStroke(if (active) 2.dp else 1.dp, if (active) GoldBright else Color.White.copy(alpha = .18f)), shadowElevation = if (active) 6.dp else 0.dp) {
                Box(contentAlignment = Alignment.Center) { Text(c.seatName(seat).first().toString(), color = if (active) Wood else Ink, fontWeight = FontWeight.Bold, fontSize = 14.sp) }
            }
            // speech hangs off zero-size anchors so it never resizes the seat
            val (anchor, tail, grow) = when (seat) {
                3 -> Triple(Modifier.align(Alignment.TopCenter).offset(x = (-15).dp, y = (-7).dp), Tail.BottomStart, Alignment.BottomStart)
                1 -> Triple(Modifier.align(Alignment.TopCenter).offset(x = 15.dp, y = (-7).dp), Tail.BottomEnd, Alignment.BottomEnd)
                else -> Triple(Modifier.align(Alignment.CenterEnd).offset(x = 10.dp), Tail.Start, Alignment.CenterStart)
            }
            Box(anchor.size(0.dp).wrapContentSize(grow, unbounded = true)) {
                Column(verticalArrangement = Arrangement.spacedBy(5.dp), horizontalAlignment = if (tail == Tail.BottomEnd) Alignment.End else Alignment.Start) {
                    AnimatedVisibility(visible = reaction != null, enter = scaleIn() + fadeIn(), exit = scaleOut() + fadeOut()) { ReactionBubble(reaction ?: "") }
                    said.forEachIndexed { i, b -> SpeechBubble(b.text, bubbleColor(c, b), if (tail == Tail.Start || i == said.lastIndex) tail else null) }
                }
            }
        }
    }
    val label = @Composable {
        val first = seat == v.first
        Text(c.seatName(seat) + if (first) " · 1" else "",
            color = if (seat == v.contractSeat && v.contract > 0) GoldBright else Ink, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = .5.sp,
            style = androidx.compose.ui.text.TextStyle(shadow = androidx.compose.ui.graphics.Shadow(Color.Black.copy(alpha = .75f), blurRadius = 6f)))
    }
    Box(modifier) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            avatar(); label()
            if (sizes.compactSeats && !c.showOpenCards) {
                Text(tr("${c.handCounts[seat]} cards", "${c.handCounts[seat]} карти"), color = InkDim, fontSize = 8.sp)
            } else BotCards(c, seat, horizontal, sizes.seatCardW, sizes.seatCardW * 1.45f)
        }
    }
}

@Composable
private fun BotCards(c: BelotController, seat: Int, horizontal: Boolean, width: Dp, height: Dp) {
    val count = c.handCounts[seat]
    if (c.showOpenCards) {
        val mask = c.openHand(seat)
        val cards = sortHand((0 until 32).filter { mask and (1L shl it) != 0L }, c.view?.contract ?: 0)
        if (horizontal) Row { cards.forEachIndexed { i, card -> CardFace(card, c.fourColor, Modifier.offset(x = if (i == 0) 0.dp else (-width * .45f) * i), width, height, small = true) } }
        else Column { cards.forEachIndexed { i, card -> CardFace(card, c.fourColor, Modifier.offset(y = if (i == 0) 0.dp else (-height * .55f) * i), width, height, small = true) } }
    } else Box(Modifier.padding(top = 3.dp, end = 7.dp)) {
        CardBack(Modifier, width, height)
        Surface(
            modifier = Modifier.size(19.dp).align(Alignment.TopEnd).offset(x = 8.dp, y = (-7).dp),
            shape = CircleShape, color = GoldBright, border = BorderStroke(1.dp, Wood),
        ) { Box(contentAlignment = Alignment.Center) { Text("$count", color = Wood, fontSize = 9.sp, fontWeight = FontWeight.ExtraBold) } }
    }
}

@Composable
private fun TrickCard(c: BelotController, card: Int, seat: Int, modifier: Modifier, sizes: Sizes, home: DpOffset, seatAt: Map<Side, DpOffset>) {
    key(card) {
        val animate = c.animations
        val from = seatAt.getValue(sideOf(seat))
        val collectTo = c.collectTo
        val to = if (collectTo >= 0) seatAt.getValue(sideOf(collectTo)) else home
        val enter = remember { Animatable(if (animate) 0f else 1f) }
        LaunchedEffect(Unit) { enter.animateTo(1f, tween(glideMs(300, from, home, max = 550), easing = FastOutSlowInEasing)) }
        val collect by animateFloatAsState(if (collectTo >= 0 && animate) 1f else 0f,
            tween(glideMs(350, home, to, max = 500), easing = FastOutSlowInEasing), label = "collect")
        val spin = if (seat == 3 || seat == 2) -1f else 1f
        CardFace(card, c.fourColor, modifier.graphicsLayer {
            val p = enter.value
            translationX = ((from.x - home.x) * (1 - p) + (to.x - home.x) * collect).toPx()
            translationY = ((from.y - home.y) * (1 - p) + (to.y - home.y) * collect).toPx()
            rotationZ = (1 - p) * 14f * spin
            val sc = (.82f + .18f * p) * (1f - .45f * collect)
            scaleX = sc; scaleY = sc
            alpha = (.4f + .6f * p) * (1f - .7f * collect)
        }, sizes.trickW, sizes.trickH)
    }
}

@Composable
private fun Hand(c: BelotController, sizes: Sizes, available: Dp, modifier: Modifier) = Column(modifier, horizontalAlignment = Alignment.CenterHorizontally) {
    val cards = c.hand
    val v = c.view
    val playing = v?.phase == Phase.PLAY && c.legal.isNotEmpty()
    val w = sizes.handW; val h = sizes.handH
    val gap = w * .14f
    val suits = cards.zipWithNext().count { (a, b) -> suitOf(a) != suitOf(b) }
    val per = if (cards.size > 1) minOf(3.5f, 22f / (cards.size - 1)) else 0f
    // the outer cards lean out by their height times the sine of the fan's outer angle: keep that on screen too
    val lean = h * kotlin.math.sin(per * (cards.size - 1) / 2f * kotlin.math.PI.toFloat() / 180f)
    val step = minOf(w * .78f, (available - 20.dp - lean * 2 - w - gap * suits) / maxOf(1, cards.size - 1))
    val total = w + step * (cards.size - 1).coerceAtLeast(0) + gap * suits
    val radius = h.value * 1.6f
    val pulse = rememberInfiniteTransition(label = "hint")
    val beat by pulse.animateFloat(0f, 1f, infiniteRepeatable(tween(900, easing = FastOutSlowInEasing), RepeatMode.Reverse), label = "beat")
    val slide = if (c.animations) tween<Dp>(240, easing = FastOutSlowInEasing) else snap()
    val turn = if (c.animations) tween<Float>(240, easing = FastOutSlowInEasing) else snap()
    Box(Modifier.fillMaxWidth().height(h + 30.dp)) {
        var x = -total / 2 + w / 2
        cards.forEachIndexed { index, card -> key(card) {
            if (index > 0 && suitOf(cards[index - 1]) != suitOf(card)) x += gap
            val legal = playing && card in c.legal
            val angle = (index - (cards.size - 1) / 2f) * per
            val drop = (radius * (1 - cos(angle.toDouble() * kotlin.math.PI / 180))).toFloat().dp
            val best = playing && c.hintPick == card
            val lift = (if (legal) (-14).dp else 0.dp) + (if (best) (-16).dp else 0.dp)
            val ax = animateDpAsState(x, slide, label = "x")
            val ay = animateDpAsState(lift + drop - 12.dp, slide, label = "y")
            val aAngle = animateFloatAsState(angle, turn, label = "angle")
            Column(
                Modifier.align(Alignment.BottomCenter)
                    .offset { androidx.compose.ui.unit.IntOffset(ax.value.roundToPx(), ay.value.roundToPx()) }
                    .graphicsLayer { rotationZ = aAngle.value; transformOrigin = TransformOrigin(0.5f, 1f) }
                    .then(if (best) Modifier.zIndex(1f) else Modifier),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                val pct = if (playing) c.hints[card]?.let { "${(it * 10f).roundToInt() / 10f}%" } else null
                Box(Modifier.size(0.dp).wrapContentSize(Alignment.BottomCenter, unbounded = true)) {
                    if (pct != null) Surface(Modifier.offset(y = (-3).dp), shape = RoundedCornerShape(10.dp),
                        color = if (best) GoldBright else Color.Black.copy(alpha = .55f),
                        border = BorderStroke(1.dp, if (best) GoldBright else Gold.copy(alpha = .4f)), shadowElevation = if (best) 6.dp else 0.dp) {
                        Text(if (best) "★ $pct" else pct, color = if (best) Wood else Ink.copy(alpha = .85f), fontSize = if (best) 12.sp else 10.sp,
                            fontWeight = FontWeight.ExtraBold, modifier = Modifier.padding(horizontal = 6.dp, vertical = 1.dp))
                    }
                }
                CardFace(card, c.fourColor, if (best) Modifier.graphicsLayer { val k = 1f + .035f * beat; scaleX = k; scaleY = k } else Modifier,
                    w, h, enabled = !playing || legal, recommended = best, glow = legal,
                    onClick = if (legal) ({ c.play(card) }) else null)
            }
            x += step
        } }
    }
}

// -------------------------------------------------------------- prompts ----

@Composable
private fun TablePrompt(c: BelotController, v: BelotView) {
    if (c.legal.isEmpty()) return
    when (v.phase) {
        Phase.BID -> BidPrompt(c)
        Phase.DECLARE -> DeclarePrompt(c, v)
        else -> Unit
    }
}

/** The auction: pass, the four suits, no trumps, all trumps, double, redouble. */
@Composable
private fun BidPrompt(c: BelotController) = FloatingPrompt(tr("YOUR BID", "ВАШАТА ОБЯВА"), "") {
    val legal = c.legal.filter { it in ACTION_BID until ACTION_BID + 9 }.map { it - ACTION_BID }.toSet()
    @Composable fun bidButton(bid: Int, label: String, color: Color, modifier: Modifier = Modifier) {
        val ok = bid in legal
        val pct = c.hints[ACTION_BID + bid]
        val best = c.hintPick == ACTION_BID + bid
        Column(modifier, horizontalAlignment = Alignment.CenterHorizontally) {
            Text(pct?.let { "${(it * 10).roundToInt() / 10f}%" } ?: " ", color = GoldBright, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            Surface(
                modifier = Modifier.height(36.dp).widthIn(min = 40.dp).then(if (ok) Modifier.clickable { c.bid(bid) } else Modifier),
                shape = RoundedCornerShape(10.dp), color = if (ok) Color(0xFFFFFDF6) else Color.White.copy(alpha = .12f),
                border = BorderStroke(if (best) 3.dp else 1.dp, if (best) GoldBright else Gold.copy(alpha = if (ok) 1f else .3f)),
                shadowElevation = if (ok) 3.dp else 0.dp,
            ) {
                Box(Modifier.padding(horizontal = 8.dp), contentAlignment = Alignment.Center) {
                    Text(label, color = if (ok) color else InkDim.copy(alpha = .5f), fontSize = if (label.length <= 2) 20.sp else 13.sp,
                        fontWeight = FontWeight.Bold, maxLines = 1)
                }
            }
        }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        for (s in listOf(0, 1, 2, 3)) bidButton(s + 1, com.cardtable.platform.suitSymbol(Suit.entries[s]), suitTint(Suit.entries[s], c.fourColor))
        bidButton(5, tr("NT", "БК"), Wood)
        bidButton(6, tr("AT", "ВК"), Wood)
    }
    Spacer(Modifier.height(4.dp))
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.Bottom) {
        if (7 in legal) bidButton(7, tr("Double", "Контра"), com.cardtable.platform.CardRed)
        if (8 in legal) bidButton(8, tr("Redouble", "Реконтра"), com.cardtable.platform.CardRed)
        val pct = c.hints[ACTION_BID]
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(pct?.let { "${(it * 10).roundToInt() / 10f}%" } ?: " ", color = GoldBright, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            Button(onClick = { c.bid(0) }, modifier = Modifier.height(36.dp), contentPadding = PaddingValues(horizontal = 18.dp, vertical = 4.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Gold, contentColor = Wood),
                border = if (c.hintPick == ACTION_BID) BorderStroke(3.dp, GoldBright) else null,
            ) { Text(tr("Pass", "Пас"), fontWeight = FontWeight.Bold) }
        }
    }
}

/** Trick 1: which of the combinations in hand to declare (all, by default). */
@Composable
private fun DeclarePrompt(c: BelotController, v: BelotView) {
    var chosen by remember(v.offered.size, v.log.size) { mutableStateOf((1 shl v.offered.size) - 1) }
    FloatingPrompt(tr("DECLARE", "ОБЯВЕТЕ"), tr("Tap to leave one out", "Докоснете, за да пропуснете")) {
        v.offered.forEachIndexed { i, o ->
            val on = chosen and (1 shl i) != 0
            Row(Modifier.clickable { chosen = chosen xor (1 shl i) }.padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(if (on) "☑" else "☐", color = GoldBright, fontSize = 18.sp)
                Spacer(Modifier.width(6.dp))
                Text("${declarationName(o.type)} · ${declarationValue(o.type)}", color = Ink, fontSize = 13.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(6.dp))
                Row(horizontalArrangement = Arrangement.spacedBy((-14).dp)) {
                    o.cards.forEach { card -> CardFace(card, c.fourColor, Modifier, 26.dp, 38.dp, small = true) }
                }
            }
        }
        Spacer(Modifier.height(6.dp))
        Button(onClick = { c.declare(chosen) }, colors = ButtonDefaults.buttonColors(containerColor = GoldBright, contentColor = Wood)) {
            Text(if (chosen == 0) tr("Declare nothing", "Без обявяване") else tr("Declare", "Обяви"), fontWeight = FontWeight.Bold)
        }
    }
}

// ------------------------------------------------------------ deal over ----

@Composable
private fun DealSheet(c: BelotController, v: BelotView) {
    val r = v.result ?: return
    val gameOver = v.phase == Phase.GAME_OVER
    val passed = v.contract == 0
    val us = r.points[0]; val them = r.points[1]
    val ourContract = v.contract > 0 && teamOf(v.contractSeat) == 0
    val title = when {
        gameOver && v.match -> if (v.winner == 0) tr("YOU WIN THE GAME!", "ПЕЧЕЛИТЕ ИГРАТА!") else tr("GAME LOST", "ИГРАТА Е ЗАГУБЕНА")
        passed -> tr("ALL PASSED", "ВСИЧКИ ПАС")
        r.hanging > v.hanging && us == 0 && them == 0 -> tr("HANGING", "ВИСЯЩА")
        us > them -> tr("DEAL WON", "СПЕЧЕЛЕНА РАЗДАВКА")
        us < them -> tr("DEAL LOST", "ЗАГУБЕНА РАЗДАВКА")
        else -> tr("EVEN", "РАВНО")
    }
    val icon = when { gameOver && v.winner == 0 -> "🏆"; passed -> "🤷"; us > them -> "🎉"; us < them -> "😞"; else -> "🤝" }
    // the contract made or not: the contract team scores nothing when it is "inside"
    val ct = if (v.contract > 0) teamOf(v.contractSeat) else -1
    val inside = ct >= 0 && r.points[ct] == 0 && r.points[1 - ct] > 0
    val shownP = remember { Animatable(0f) }
    LaunchedEffect(Unit) { shownP.animateTo(1f, tween(900, easing = FastOutSlowInEasing)) }
    Box(Modifier.fillMaxSize().background(Color(0xB0000000)), contentAlignment = Alignment.Center) {
        Surface(modifier = Modifier.padding(18.dp).widthIn(max = 400.dp), shape = RoundedCornerShape(22.dp),
            color = Color.Transparent, border = BorderStroke(1.5.dp, Gold.copy(alpha = .85f)), shadowElevation = 16.dp) {
            Column(Modifier.background(Brush.verticalGradient(listOf(Felt1.copy(alpha = .97f), Felt0.copy(alpha = .98f))))
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp, vertical = 16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text(icon, fontSize = 34.sp)
                Text(title, color = GoldBright, fontWeight = FontWeight.ExtraBold, fontSize = 21.sp, letterSpacing = 2.sp, textAlign = TextAlign.Center)
                if (!passed) {
                    Spacer(Modifier.height(6.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Chip(contractLabel(v.contract, v.doubling), trumpSuitOf(v.contract)?.let { suitTint(Suit.entries[it], c.fourColor, onDark = true) } ?: Ink)
                        Chip(c.seatName(v.contractSeat), if (ourContract) GoldBright else ThemColor)
                        if (inside) Chip(tr("inside", "вкарани"), ThemColor)
                        if (r.capot) Chip(tr("capot", "капо"), GoldBright)
                    }
                    Spacer(Modifier.height(12.dp))
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(tr("US", "НИЕ"), color = GoldBright, fontWeight = FontWeight.Bold, fontSize = 12.sp, letterSpacing = 2.sp)
                            Text("${(us * shownP.value).roundToInt()}", color = if (us > 0) Ink else InkDim, fontSize = 44.sp, fontWeight = FontWeight.ExtraBold, lineHeight = 46.sp)
                            Text(tr("${r.raw[0]} raw", "${r.raw[0]} точки"), color = InkDim, fontSize = 11.sp)
                        }
                        Text("–", color = InkDim, fontSize = 30.sp)
                        Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(tr("THEM", "ТЕ"), color = ThemColor, fontWeight = FontWeight.Bold, fontSize = 12.sp, letterSpacing = 2.sp)
                            Text("${(them * shownP.value).roundToInt()}", color = if (them > 0) Ink else InkDim, fontSize = 44.sp, fontWeight = FontWeight.ExtraBold, lineHeight = 46.sp)
                            Text(tr("${r.raw[1]} raw", "${r.raw[1]} точки"), color = InkDim, fontSize = 11.sp)
                        }
                    }
                    val scored = r.declared.filter { it.scored == 1 }
                    if (scored.isNotEmpty()) {
                        Spacer(Modifier.height(8.dp))
                        Text(tr("DECLARATIONS", "ОБЯВЕНИ"), color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp)
                        scored.forEach { d ->
                            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(vertical = 2.dp)) {
                                Text("${c.seatName(d.seat)}: ${declarationName(d.type)} ${declarationValue(d.type)}",
                                    color = if (teamOf(d.seat) == 0) GoldBright else ThemColor, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                Spacer(Modifier.width(6.dp))
                                Row(horizontalArrangement = Arrangement.spacedBy((-14).dp)) {
                                    d.cards.forEach { card -> CardFace(card, c.fourColor, Modifier, 22.dp, 32.dp, small = true) }
                                }
                            }
                        }
                    }
                }
                if (r.hanging > 0) {
                    Spacer(Modifier.height(6.dp))
                    Text(tr("${r.hanging} points hang over to the next deal", "${r.hanging} точки остават висящи за следващата раздавка"), color = GoldBright, fontSize = 12.sp, textAlign = TextAlign.Center)
                }
                if (v.match) {
                    Spacer(Modifier.height(10.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        ScoreBox(tr("US", "НИЕ"), v.score[0], GoldBright, Modifier.width(120.dp), big = true, target = WINNING_SCORE)
                        Spacer(Modifier.width(10.dp))
                        ScoreBox(tr("THEM", "ТЕ"), v.score[1], ThemColor, Modifier.width(120.dp), big = true, target = WINNING_SCORE)
                    }
                }
                Spacer(Modifier.height(14.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Button(onClick = c::lobby, colors = ButtonDefaults.buttonColors(containerColor = Color.White.copy(alpha = .1f), contentColor = Ink)) {
                        Text(tr("Lobby", "Начало"))
                    }
                    Button(onClick = c::next, colors = ButtonDefaults.buttonColors(containerColor = GoldBright, contentColor = Wood)) {
                        Text(if (gameOver) tr("Play again", "Нова игра") else tr("Next deal", "Следваща раздавка"), fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}
