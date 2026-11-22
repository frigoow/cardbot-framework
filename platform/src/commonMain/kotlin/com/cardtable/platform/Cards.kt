package com.cardtable.platform

import androidx.compose.ui.layout.positionInWindow
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.material3.HorizontalDivider
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.zIndex
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.LinearEasing
import androidx.compose.foundation.Canvas
import androidx.compose.ui.unit.DpOffset
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.key
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.animation.core.snap
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.core.Animatable
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
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
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.TransformOrigin
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Density
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import kotlin.math.cos
import kotlin.math.roundToInt

/** The suits, in no game's order: each game maps its own card numbers onto these ([Deck]). */
enum class Suit(val symbol: String) { CLUBS("♣"), DIAMONDS("♦"), HEARTS("♥"), SPADES("♠") }

/** The ranks of the 32-card deck; [label] is the corner index. */
enum class Rank(val label: String) { SEVEN("7"), EIGHT("8"), NINE("9"), TEN("10"), JACK("J"), QUEEN("Q"), KING("K"), ACE("A") }

data class PlayingCard(val suit: Suit, val rank: Rank)

/** A game's card numbering: card id -> the card it shows. */
fun interface Deck { fun face(card: Int): PlayingCard }

/** The deck of the game on screen: [CardFace] draws card ids through it. */
val LocalDeck = staticCompositionLocalOf<Deck> { error("no Deck provided: wrap the screen in CompositionLocalProvider(LocalDeck provides ...)") }

// U+FE0E asks for the text glyph: without it ♥ ♦ ♣ render as colour emoji
// that ignore the suit colour (the four-colour deck showed red diamonds).
fun suitSymbol(suit: Suit?) = if (suit == null) "∅" else suit.symbol + "\uFE0E"

/**
 * Suit colour: red suits red and black suits black, as on a real deck, but in
 * distinct shades so they never blur together: spades jet black, clubs a dark
 * slate, hearts crimson, diamonds a warmer orange-red. null: no trump.
 */
@Suppress("UNUSED_PARAMETER")
@Composable
fun suitTint(suit: Suit?, fourColor: Boolean = false, onDark: Boolean = false): Color = when (suit) {
    null -> GoldBright
    Suit.HEARTS -> Color(0xFFD0142E)
    Suit.DIAMONDS -> Color(0xFFE2581C)
    Suit.CLUBS -> if (onDark) Color(0xFFC6D2E8) else Color(0xFF2D3E60)
    Suit.SPADES -> if (onDark) Ink else Color(0xFF0E0E0E)
}

// ---------------------------------------------------------------- cards ----

@Composable
fun CardBack(modifier: Modifier, width: Dp, height: Dp) {
    val t = LocalAppTheme.current
    Box(
        modifier.size(width, height).shadow(3.dp, RoundedCornerShape(5.dp)).clip(RoundedCornerShape(5.dp))
            .background(Brush.linearGradient(listOf(t.backA, t.backB)))
            .border(1.5.dp, t.paperTop, RoundedCornerShape(5.dp))
            .padding(3.dp)
            .drawBehind {
                if (t.style == "cafe") {
                    // a dream in miniature: night sky over sand, an eye at the horizon
                    val hz = size.height * .66f
                    drawRect(Brush.verticalGradient(listOf(Color(0xFF0F1D36), Color(0xFF3D5F8A), Color(0xFFE6B987)), endY = hz),
                        size = androidx.compose.ui.geometry.Size(size.width, hz))
                    drawRect(Brush.verticalGradient(listOf(Color(0xFFD8A866), Color(0xFF9A6A35)), startY = hz, endY = size.height),
                        topLeft = Offset(0f, hz), size = androidx.compose.ui.geometry.Size(size.width, size.height - hz))
                    drawCircle(Color(0xFFF3E9D2), size.width * .07f, Offset(size.width * .75f, size.height * .14f))
                    surrealEye(Offset(size.width / 2, size.height * .45f), size.width * .7f, Color(0xFF3D5F8A))
                } else {
                    // diamond lattice
                    val step = 7.dp.toPx()
                    var x = -size.height
                    while (x < size.width) {
                        drawLine(t.backLine, Offset(x, 0f), Offset(x + size.height, size.height), strokeWidth = .8f)
                        drawLine(t.backLine, Offset(x + size.height, 0f), Offset(x, size.height), strokeWidth = .8f)
                        x += step
                    }
                }
            }
            .border(1.dp, t.accent.copy(alpha = .6f), RoundedCornerShape(3.dp)),
        contentAlignment = Alignment.Center,
    ) {
        if (t.style != "cafe") Surface(shape = CircleShape, color = t.backB, border = BorderStroke(1.dp, t.accent.copy(alpha = .8f))) {
            Text(t.emblem, color = t.accentBright, fontSize = (width.value * .30f).sp, lineHeight = (width.value * .32f).sp,
                modifier = Modifier.padding(horizontal = 3.dp))
        }
    }
}

/** Pip positions (fractions of the pip area) of the number cards, as on a real deck. */
val PIPS: Map<Rank, List<Pair<Float, Float>>> = mapOf(
    Rank.SEVEN to listOf(.25f to .0f, .75f to .0f, .5f to .25f, .25f to .5f, .75f to .5f, .25f to 1f, .75f to 1f),               // 7
    Rank.EIGHT to listOf(.25f to .0f, .75f to .0f, .5f to .25f, .25f to .5f, .75f to .5f, .5f to .75f, .25f to 1f, .75f to 1f),  // 8
    Rank.NINE to listOf(.25f to .0f, .75f to .0f, .25f to .33f, .75f to .33f, .5f to .5f, .25f to .67f, .75f to .67f, .25f to 1f, .75f to 1f), // 9
    Rank.TEN to listOf(.25f to .0f, .75f to .0f, .5f to .17f, .25f to .33f, .75f to .33f, .25f to .67f, .75f to .67f, .5f to .83f, .25f to 1f, .75f to 1f), // 10
)

/**
 * A real-looking card: corner indices (rank over suit) top-left and, upside
 * down, bottom-right; the classic pip layout for 7-10, one large pip for the
 * ace, a framed court card with a crown for J / Q / K.
 */
@Composable
fun CardFace(card: Int, fourColor: Boolean, modifier: Modifier, width: Dp, height: Dp, small: Boolean = false,
                     enabled: Boolean = true, recommended: Boolean = false, glow: Boolean = false, onClick: (() -> Unit)? = null) {
    val shape = RoundedCornerShape(if (small) 4.dp else 7.dp)
    val interaction = if (enabled && onClick != null) Modifier.clickable(onClick = onClick) else Modifier
    val face = LocalDeck.current.face(card); val suit = face.suit; val rank = face.rank
    val color = suitTint(suit, fourColor)
    val idx: TextUnit = (width.value * .26f).coerceIn(9f, 20f).sp
    val idxPip: TextUnit = (width.value * .30f).coerceIn(9f, 22f).sp
    val t = LocalAppTheme.current
    Surface(
        modifier = modifier.then(Modifier.size(width, height)).then(interaction)
            .shadow(if (recommended) 14.dp else if (glow) 9.dp else 4.dp, shape,
                ambientColor = if (glow || recommended) t.accentBright else Color.Black,
                spotColor = if (glow || recommended) t.accentBright else Color.Black),
        shape = shape, color = Color.Transparent,
        border = BorderStroke(if (recommended) 3.dp else if (glow) 1.5.dp else 1.dp,
            if (recommended) t.accentBright else if (glow) t.accent else Color.Black.copy(alpha = .18f)),
    ) {
        Box(Modifier.fillMaxSize()
            .background(if (enabled) Brush.verticalGradient(listOf(t.paperTop, t.paperBottom))
                        else Brush.verticalGradient(listOf(Color(0xFFCFCBC1), Color(0xFFBDB8AD))))) {
            // fine inner frame in the theme's accent
            if (!small) Box(Modifier.matchParentSize().padding(2.5.dp).border(.8.dp, t.accent.copy(alpha = .35f), RoundedCornerShape(5.dp)))
            val corner = @Composable { m: Modifier ->
                Column(m.padding(horizontal = if (small) 2.dp else 3.dp, vertical = if (small) 1.dp else 2.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(rank.label, color = color, fontWeight = FontWeight.ExtraBold, fontSize = idx, lineHeight = idx)
                    Text(suitSymbol(suit), color = color, fontSize = idxPip, lineHeight = idxPip)
                }
            }
            corner(Modifier.align(Alignment.TopStart))
            if (!small) {
                corner(Modifier.align(Alignment.BottomEnd).rotate(180f))
                // the face itself: inside the corner indices
                val fx = width * .24f; val fy = height * .12f
                val faceW = width - fx * 2; val faceH = height - fy * 2
                Box(Modifier.offset(x = fx, y = fy).size(faceW, faceH)) {
                    when (rank) {
                        Rank.ACE -> if (t.style == "cafe") CafeAce(suit, color, faceW, faceH)
                             else Text(suitSymbol(suit), Modifier.align(Alignment.Center), color = color, fontSize = (width.value * .62f).sp)
                        Rank.JACK, Rank.QUEEN, Rank.KING -> if (t.style == "cafe") CafeCourt(rank, suit, color, faceW, faceH) else CourtFace(rank, suit, color, faceW, faceH)
                        else -> {
                            val pip = (width.value * .2f).coerceAtLeast(7f)
                            val pipBox = (pip * 1.25f).dp
                            PIPS.getValue(rank).forEach { (px, py) ->
                                Text(suitSymbol(suit), color = color, fontSize = pip.sp, lineHeight = pip.sp, textAlign = TextAlign.Center,
                                    modifier = Modifier.offset(x = faceW * px - pipBox / 2, y = (faceH - pipBox) * py)
                                        .size(pipBox).then(if (py > .5f) Modifier.rotate(180f) else Modifier))
                            }
                        }
                    }
                }
            }
        }
    }
}

/**
 * The café theme's court cards, after the surrealists: a little dream landscape with
 * the jack's upturned moustache, the queen's eye, the king's melting crown.
 */
@Composable
fun CafeCourt(rank: Rank, suit: Suit, color: Color, w: Dp, h: Dp) {
    val t = LocalAppTheme.current
    val shape = RoundedCornerShape(3.dp)
    Box(Modifier.size(w, h).clip(shape).border(1.dp, t.accent.copy(alpha = .8f), shape)
        .drawBehind {
            val hz = size.height * .62f
            drawRect(Brush.verticalGradient(listOf(Color(0xFF1B2E4F), Color(0xFF5B7DA6), Color(0xFFE9C38E)), endY = hz), size = androidx.compose.ui.geometry.Size(size.width, hz))
            drawRect(Brush.verticalGradient(listOf(Color(0xFFD8A866), Color(0xFFB07A3B)), startY = hz, endY = size.height),
                topLeft = Offset(0f, hz), size = androidx.compose.ui.geometry.Size(size.width, size.height - hz))
            // a long shadow across the sand
            drawLine(Color(0x552A1E14), Offset(size.width * .5f, hz + size.height * .1f), Offset(size.width * .05f, size.height * .97f), strokeWidth = size.width * .06f)
            val c = Offset(size.width / 2, size.height * .42f)
            when (rank) {
                Rank.JACK -> surrealMoustache(c, size.width * .82f, Color(0xFF17110C))
                Rank.QUEEN -> surrealEye(c, size.width * .84f, color)
                else -> meltingCrown(c, size.width * .74f)
            }
        }) {
        Text(suitSymbol(suit), color = Color(0xFFFFFBF2), fontSize = (w.value * .26f).sp,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 1.dp))
    }
}

/** The café theme's ace: the suit symbol, slowly dripping. */
@Composable
fun CafeAce(suit: Suit, color: Color, w: Dp, h: Dp) {
    Box(Modifier.size(w, h).drawBehind {
        val cx = size.width / 2; val top = size.height * .62f
        val drips = listOf(-.16f to .22f, -.02f to .34f, .12f to .16f)
        drips.forEach { (dx, len) ->
            val x = cx + dx * size.width; val r = size.width * .035f
            drawRoundRect(color, Offset(x - r, top - r), androidx.compose.ui.geometry.Size(r * 2, len * size.height),
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(r, r))
            drawCircle(color, r * 1.35f, Offset(x, top + len * size.height - r))
        }
    }, contentAlignment = Alignment.Center) {
        Text(suitSymbol(suit), color = color, fontSize = (w.value * .62f).sp, modifier = Modifier.offset(y = -h * .06f))
    }
}

fun androidx.compose.ui.graphics.drawscope.DrawScope.surrealMoustache(c: Offset, w: Float, ink: Color) {
    for (side in listOf(-1f, 1f)) {
        val p = androidx.compose.ui.graphics.Path()
        p.moveTo(c.x, c.y)
        p.cubicTo(c.x + side * w * .14f, c.y + w * .16f, c.x + side * w * .34f, c.y + w * .12f, c.x + side * w * .42f, c.y - w * .30f)
        p.cubicTo(c.x + side * w * .32f, c.y - w * .02f, c.x + side * w * .16f, c.y + w * .06f, c.x, c.y - w * .05f)
        p.close()
        drawPath(p, ink)
    }
    drawCircle(ink, w * .05f, Offset(c.x, c.y - w * .02f))
}

fun androidx.compose.ui.graphics.drawscope.DrawScope.surrealEye(c: Offset, w: Float, iris: Color) {
    val ink = Color(0xFF17110C)
    val p = androidx.compose.ui.graphics.Path()
    p.moveTo(c.x - w * .45f, c.y)
    p.quadraticBezierTo(c.x, c.y - w * .38f, c.x + w * .45f, c.y)
    p.quadraticBezierTo(c.x, c.y + w * .38f, c.x - w * .45f, c.y)
    p.close()
    drawPath(p, Color(0xFFFFFBF2))
    drawCircle(iris, w * .15f, c)
    drawCircle(ink, w * .065f, c)
    drawCircle(Color.White, w * .025f, Offset(c.x + w * .04f, c.y - w * .04f))
    drawPath(p, ink, style = androidx.compose.ui.graphics.drawscope.Stroke(w * .03f))
    for (k in 0..4) {
        val f = -.32f + k * .16f
        val x = c.x + f * w; val y = c.y - w * .19f * (1 - (f / .45f) * (f / .45f)) - w * .005f
        drawLine(ink, Offset(x, y), Offset(x + f * w * .25f, y - w * .12f), strokeWidth = w * .022f)
    }
    // a single tear
    val tx = c.x + w * .08f; val ty = c.y + w * .30f
    drawCircle(Color(0xFFBFD8F0), w * .045f, Offset(tx, ty))
    drawLine(Color(0xFFBFD8F0), Offset(tx, ty - w * .1f), Offset(tx, ty), strokeWidth = w * .03f)
}

fun androidx.compose.ui.graphics.drawscope.DrawScope.meltingCrown(c: Offset, w: Float) {
    val gold = Color(0xFFD9A93E); val ink = Color(0xFF4A3618)
    val base = c.y + w * .12f; val l = c.x - w * .45f; val r = c.x + w * .45f
    val p = androidx.compose.ui.graphics.Path()
    p.moveTo(l, base)
    p.lineTo(l, c.y - w * .22f); p.lineTo(c.x - w * .22f, c.y - w * .02f)
    p.lineTo(c.x, c.y - w * .36f); p.lineTo(c.x + w * .22f, c.y - w * .02f)
    p.lineTo(r, c.y - w * .22f); p.lineTo(r, base)
    // the rim melts: soft drips hanging from the band
    p.cubicTo(r - w * .05f, base + w * .30f, r - w * .14f, base + w * .30f, r - w * .17f, base + w * .06f)
    p.cubicTo(c.x + w * .05f, base + w * .1f, c.x + w * .06f, base + w * .48f, c.x - w * .04f, base + w * .44f)
    p.cubicTo(c.x - w * .10f, base + w * .40f, c.x - w * .10f, base + w * .08f, c.x - w * .2f, base + w * .07f)
    p.cubicTo(l + w * .12f, base + w * .2f, l + w * .04f, base + w * .2f, l, base)
    p.close()
    drawPath(p, gold)
    drawPath(p, ink, style = androidx.compose.ui.graphics.drawscope.Stroke(w * .025f))
    for (x in listOf(l, c.x, r)) drawCircle(Color(0xFFF6EEDD), w * .05f, Offset(x, if (x == c.x) c.y - w * .36f else c.y - w * .22f))
    drawCircle(Color(0xFFB0302A), w * .06f, Offset(c.x, base - w * .07f))
}

/**
 * The café theme's world behind the table, after the surrealists (all drawn here): a
 * vast dusk sky, a low horizon with cliffs and a far bare tree, sand with
 * long shadows. The table stands in it.
 */
fun androidx.compose.ui.graphics.drawscope.DrawScope.surrealBackdrop() {
    val w = size.width; val h = size.height; val hz = h * .78f
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    drawRect(Brush.verticalGradient(listOf(Color(0xFF0B1528), Color(0xFF1E3558), Color(0xFF557AA6), Color(0xFFE8BC88)), endY = hz), size = sz(w, hz))
    drawRect(Brush.verticalGradient(listOf(Color(0xFFD9AA68), Color(0xFFB27C40), Color(0xFF7A4E26)), startY = hz, endY = h),
        topLeft = Offset(0f, hz), size = sz(w, h - hz))
    val moon = Offset(w * .84f, h * .07f)
    drawCircle(Brush.radialGradient(listOf(Color(0x44F3E9D2), Color.Transparent), center = moon, radius = w * .16f), w * .16f, moon)
    drawCircle(Color(0xFFF3E9D2), w * .045f, moon)
    drawOval(Color(0x26FFFFFF), Offset(w * .08f, h * .5f), sz(w * .4f, h * .006f))
    drawOval(Color(0x1EFFFFFF), Offset(w * .5f, h * .62f), sz(w * .34f, h * .005f))
    val cliff = androidx.compose.ui.graphics.Path().apply {
        moveTo(0f, hz); lineTo(0f, hz - h * .06f); lineTo(w * .07f, hz - h * .075f); lineTo(w * .15f, hz - h * .04f)
        lineTo(w * .22f, hz - h * .05f); lineTo(w * .34f, hz); close()
    }
    drawPath(cliff, Color(0xFF6E482E))
    val far = androidx.compose.ui.graphics.Path().apply {
        moveTo(w * .72f, hz); lineTo(w * .82f, hz - h * .02f); lineTo(w * .9f, hz - h * .016f); lineTo(w, hz - h * .03f); lineTo(w, hz); close()
    }
    drawPath(far, Color(0xFF65607A))

}

/** Polished walnut: long, gently wavy grain lines with a few light streaks. */
fun androidx.compose.ui.graphics.drawscope.DrawScope.woodGrain() {
    val grain = Color(0x1F1E0E05); val light = Color(0x14FFE6C8)
    var y = 0f; var k = 0
    while (y < size.height) {
        val path = androidx.compose.ui.graphics.Path()
        for (i in 0..24) {
            val x = size.width * i / 24f
            val yy = y + 5.dp.toPx() * kotlin.math.sin(i * .55 + k * 1.7).toFloat()
            if (i == 0) path.moveTo(x, yy) else path.lineTo(x, yy)
        }
        drawPath(path, if (k % 3 == 0) light else grain, style = androidx.compose.ui.graphics.drawscope.Stroke((1 + k % 2).dp.toPx()))
        y += (9 + (k * 7) % 11).dp.toPx(); k++
    }
    // a soft sheen, as on a varnished top
    drawRect(Brush.linearGradient(listOf(Color(0x18FFFFFF), Color.Transparent, Color(0x10FFFFFF)), start = Offset.Zero, end = Offset(size.width, size.height)))
}

/** The table's bottom edge melts: slow drips of wood hanging under the rim. */
fun androidx.compose.ui.graphics.drawscope.DrawScope.meltingEdge(rim: Color) {
    val bottom = size.height
    listOf(.16f to 14f, .27f to 7f, .62f to 9f, .71f to 18f, .86f to 8f).forEach { (fx, len) ->
        val x = size.width * fx; val r = 4.dp.toPx(); val l = len.dp.toPx()
        drawRoundRect(rim, Offset(x - r, bottom - 6.dp.toPx()), androidx.compose.ui.geometry.Size(r * 2, l + 6.dp.toPx()),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(r, r))
        drawCircle(rim, r * 1.3f, Offset(x, bottom + l - r * .3f))
        drawCircle(Color(0x33FFFFFF), r * .4f, Offset(x - r * .4f, bottom + l - r * .6f))
    }
}

/** A closed book on the table, a soft clock draped over its edge (with ants). */
fun androidx.compose.ui.graphics.drawscope.DrawScope.drawClockOnBook() {
    val w = size.width; val h = size.height
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    val bx = w * .38f; val by = h * .45f; val bw = w * .6f; val bh = h * .3f
    drawOval(Color(0x55180C04), Offset(bx - w * .04f, by + bh - h * .05f), sz(bw * 1.12f, h * .14f))
    drawRoundRect(Color(0xFF7A1F22), Offset(bx, by), sz(bw, bh), cornerRadius = androidx.compose.ui.geometry.CornerRadius(3f, 3f))
    drawRect(Color(0xFFF1E6CC), Offset(bx + 2f, by + bh * .18f), sz(bw - 4f, bh * .64f))
    for (k in 1..4) drawLine(Color(0x33000000), Offset(bx + 2f, by + bh * (.18f + .13f * k)), Offset(bx + bw - 2f, by + bh * (.18f + .13f * k)), strokeWidth = .8f)
    drawRect(Color(0xFFD4AF5A), Offset(bx + bw * .1f, by + bh * .05f), sz(bw * .5f, 1.4f))
    // the clock: face up on the cover, its left half pouring over the spine edge
    val face = Offset(bx + bw * .45f, by - h * .01f); val rx = bw * .42f; val ry = h * .07f
    val clock = androidx.compose.ui.graphics.Path().apply {
        moveTo(face.x + rx, face.y)
        cubicTo(face.x + rx, face.y - ry * 1.5f, face.x - rx * .6f, face.y - ry * 1.5f, face.x - rx, face.y - ry * .2f)
        cubicTo(face.x - rx * 1.25f, face.y + ry, face.x - rx * 1.3f, by + bh * .9f, face.x - rx * 1.05f, by + bh * 1.25f)
        cubicTo(face.x - rx * .85f, by + bh * 1.35f, face.x - rx * .6f, by + bh * .8f, face.x - rx * .55f, by + 2f)
        cubicTo(face.x - rx * .1f, face.y + ry, face.x + rx * .7f, face.y + ry * 1.1f, face.x + rx, face.y)
        close()
    }
    drawPath(clock, Brush.verticalGradient(listOf(Color(0xFFF6EED2), Color(0xFFDCC88F)), startY = face.y - ry * 1.5f, endY = by + bh * 1.3f))
    drawPath(clock, Color(0xFFB08A3A), style = androidx.compose.ui.graphics.drawscope.Stroke(3f))
    drawPath(clock, Color(0xFF5A4220), style = androidx.compose.ui.graphics.drawscope.Stroke(1f))
    for (k in 0 until 12) {
        val a = kotlin.math.PI * 2 * k / 12
        val px = face.x + rx * .78f * kotlin.math.cos(a).toFloat(); val py = face.y - ry * .3f + ry * .75f * kotlin.math.sin(a).toFloat()
        if (px > face.x - rx * .55f) drawCircle(Color(0xFF3A2A14), if (k % 3 == 0) 1.8f else 1f, Offset(px, py))
    }
    val hub = Offset(face.x + rx * .05f, face.y - ry * .3f)
    drawLine(Color(0xFF3A2A14), hub, Offset(hub.x + rx * .45f, hub.y - ry * .3f), strokeWidth = 2f)
    drawLine(Color(0xFF3A2A14), hub, Offset(hub.x - rx * .3f, hub.y + ry * .45f), strokeWidth = 1.4f)
    drawCircle(Color(0xFF3A2A14), 2f, hub)
    for (k in 0 until 5) drawCircle(Color(0xFF17110C), 1.4f, Offset(face.x - rx * (1.0f + .05f * (k % 2)), by + bh * (.1f + .2f * k)))
}

/** A small elephant on impossibly long legs, walking across the table. */
fun androidx.compose.ui.graphics.drawscope.DrawScope.drawStiltElephant(step: Float) {
    val w = size.width; val h = size.height; val ink = Color(0xFF1E1511)
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    val foot = h * .95f; val body = h * .22f
    drawOval(Color(0x44180C04), Offset(w * .05f, foot - h * .03f), sz(w * .9f, h * .07f))
    drawOval(ink, Offset(w * .22f, body - h * .09f), sz(w * .5f, h * .17f))
    drawCircle(ink, w * .11f, Offset(w * .75f, body - h * .04f))
    drawLine(ink, Offset(w * .82f, body), Offset(w * .86f, body + h * .16f), strokeWidth = 2.2f)
    drawOval(Color(0xFFE9C38E), Offset(w * .28f, body - h * .14f), sz(w * .3f, h * .07f))   // the obelisk saddle
    val sw = kotlin.math.sin(step * 6.283f).toFloat()
    listOf(.28f, .38f, .55f, .65f).forEachIndexed { i, fx ->
        val swing = (if (i % 2 == 0) sw else -sw) * w * .04f
        drawLine(ink, Offset(w * fx, body + h * .06f), Offset(w * fx + swing, foot), strokeWidth = 1.2f)
    }
}


/** J / Q / K: a framed panel tinted in the suit colour with a crown and the letter. */
@Composable
fun CourtFace(rank: Rank, suit: Suit, color: Color, w: Dp, h: Dp) = Box(
    Modifier.size(w, h).clip(RoundedCornerShape(4.dp))
        .background(Brush.verticalGradient(listOf(color.copy(alpha = .16f), Color(0xFFFFF4D6), color.copy(alpha = .16f))))
        .border(1.dp, color.copy(alpha = .55f), RoundedCornerShape(4.dp)),
    contentAlignment = Alignment.Center,
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(arrayOf("♝", "♛", "♚")[rank.ordinal - Rank.JACK.ordinal] + "︎", color = Color(0xFFB8892E), fontSize = (w.value * .42f).sp, lineHeight = (w.value * .44f).sp)
        Text(rank.label, color = color, fontWeight = FontWeight.ExtraBold, fontSize = (w.value * .40f).sp, lineHeight = (w.value * .42f).sp)
        Text(suitSymbol(suit), color = color, fontSize = (w.value * .28f).sp, lineHeight = (w.value * .3f).sp)
    }
}

