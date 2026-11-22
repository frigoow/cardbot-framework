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

@Composable
fun ChoiceChip(label: String, selected: Boolean, onClick: () -> Unit) = Surface(
    modifier = Modifier.clickable(onClick = onClick), shape = RoundedCornerShape(16.dp),
    color = if (selected) GoldBright else Color.White.copy(alpha = .08f),
    border = BorderStroke(1.dp, if (selected) GoldBright else Gold.copy(alpha = .5f)),
) {
    Text(label, color = if (selected) Wood else Ink, fontSize = 12.sp, fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
        modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp))
}

@Composable
fun LobbyCard(step: String, title: String, name: String, sub: String, onClick: (() -> Unit)? = null) = Surface(
    modifier = Modifier.fillMaxWidth().widthIn(max = 480.dp).then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier),
    color = Panel, shape = RoundedCornerShape(14.dp), border = BorderStroke(1.dp, Color.White.copy(alpha = .12f)),
) {
    Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
        Surface(modifier = Modifier.size(28.dp), shape = CircleShape, color = Gold.copy(alpha = .14f), border = BorderStroke(1.dp, Gold.copy(alpha = .55f))) {
            Box(contentAlignment = Alignment.Center) { Text(step, color = GoldBright, fontWeight = FontWeight.Bold) }
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(title, color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp)
            Text(name, color = Ink, fontWeight = FontWeight.Bold, fontSize = 17.sp)
            Text(sub, color = GoldBright, fontSize = 11.sp)
        }
        if (onClick != null) Text("›", color = GoldBright, fontSize = 26.sp)
    }
}

@Composable
fun StatCell(label: String, value: String) = Column(horizontalAlignment = Alignment.CenterHorizontally) {
    Text(value, color = GoldBright, fontWeight = FontWeight.Bold, fontSize = 17.sp)
    Text(label, color = InkDim, fontSize = 10.sp)
}

@Composable
fun ModeButton(title: String, sub: String, modifier: Modifier, onClick: () -> Unit) = Button(
    modifier = modifier.height(92.dp), onClick = onClick,
    colors = ButtonDefaults.buttonColors(containerColor = Gold.copy(alpha = .14f), contentColor = Ink),
    border = BorderStroke(1.dp, Gold.copy(alpha = .65f)), shape = RoundedCornerShape(14.dp),
) { Column(horizontalAlignment = Alignment.CenterHorizontally) {
    Text(title, color = GoldBright, fontWeight = FontWeight.Bold, textAlign = TextAlign.Center)
    Text(sub, color = InkDim, fontSize = 10.sp, textAlign = TextAlign.Center)
} }

@Composable
fun StatSection(title: String, content: @Composable ColumnScope.() -> Unit) {
    Spacer(Modifier.height(12.dp))
    Surface(modifier = Modifier.widthIn(max = 560.dp).fillMaxWidth(), color = Panel, shape = RoundedCornerShape(14.dp),
        border = BorderStroke(1.dp, Color.White.copy(alpha = .12f))) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 12.dp)) {
            Text(title.uppercase(), color = InkDim, fontSize = 10.sp, letterSpacing = 1.sp)
            Spacer(Modifier.height(4.dp)); content()
        }
    }
}

@Composable
fun StatLine(label: String, value: String, extra: String? = null) = Row(Modifier.fillMaxWidth().padding(vertical = 3.dp),
    verticalAlignment = Alignment.CenterVertically) {
    Text(label, color = Ink, fontSize = 13.sp, modifier = Modifier.weight(1f))
    Text(value, color = GoldBright, fontSize = 13.sp, fontWeight = FontWeight.Bold)
    if (extra != null) Text("  $extra", color = InkDim, fontSize = 12.sp, modifier = Modifier.widthIn(min = 44.dp), textAlign = TextAlign.End)
}

@Composable
fun FloatingPrompt(title: String, sub: String, content: @Composable ColumnScope.() -> Unit) =
    Surface(shape = RoundedCornerShape(16.dp), color = Color.Transparent, border = BorderStroke(1.5.dp, Gold.copy(alpha = .85f)), shadowElevation = 12.dp) {
        // the theme's panel over a hint of its table colours
        Column(Modifier.background(Brush.verticalGradient(listOf(Panel.copy(alpha = .96f), Felt0.copy(alpha = .97f))))
            .border(.8.dp, GoldBright.copy(alpha = .25f), RoundedCornerShape(16.dp))
            .padding(horizontal = 12.dp, vertical = 8.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(title, color = GoldBright, fontWeight = FontWeight.ExtraBold, fontSize = 15.sp, letterSpacing = 1.5.sp)
            if (sub.isNotBlank()) { Spacer(Modifier.height(2.dp)); Text(sub, color = Ink, textAlign = TextAlign.Center, fontSize = 11.sp, maxLines = 2) }
            Spacer(Modifier.height(6.dp)); content()
        }
    }

@Composable
fun ScoreBox(label: String, score: Int, color: Color, modifier: Modifier, big: Boolean, target: Int) = Surface(
    modifier = modifier, shape = RoundedCornerShape(12.dp), color = Color.Black.copy(alpha = .35f),
    border = BorderStroke(1.dp, color.copy(alpha = .45f)),
) {
    Column(Modifier.padding(horizontal = 10.dp, vertical = 5.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(label, color = color, fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 2.sp)
        Text("$score", color = Ink, fontSize = if (big) 30.sp else 24.sp, fontWeight = FontWeight.ExtraBold, lineHeight = if (big) 32.sp else 26.sp, style = androidx.compose.ui.text.TextStyle(fontFeatureSettings = "tnum"))
        Box(Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)).background(Color.White.copy(alpha = .12f))) {
            Box(Modifier.fillMaxWidth((score / target.toFloat()).coerceIn(0f, 1f)).height(4.dp).background(color))
        }
        Text(tl("to $target", "tot $target", "до $target"), color = InkDim, fontSize = 8.sp)
    }
}

enum class Tail { BottomStart, BottomEnd, Start }

@Composable
fun SpeechBubble(text: String, color: Color, tail: Tail? = Tail.BottomStart) = Box(Modifier.keepOnScreen()) {
    // tail: a small rotated square peeking out towards the speaker's avatar
    val tailMod = when (tail) {
        Tail.BottomStart -> Modifier.align(Alignment.BottomStart).offset(x = 10.dp, y = 4.dp)
        Tail.BottomEnd -> Modifier.align(Alignment.BottomEnd).offset(x = (-10).dp, y = 4.dp)
        Tail.Start -> Modifier.align(Alignment.CenterStart).offset(x = (-4).dp)
        null -> null
    }
    if (tailMod != null) Box(tailMod.size(10.dp).rotate(45f).background(Color(0xFFFFFDF6)))
    Surface(shape = RoundedCornerShape(14.dp), color = Color(0xFFFFFDF6), shadowElevation = 5.dp) {
        Text(text, color = color, fontWeight = FontWeight.ExtraBold, fontSize = 13.sp,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp))
    }
}

/**
 * Keeps a bubble inside the screen: it measures where it landed (before its
 * own shift, so there is no feedback) and slides back in by the overflow.
 */
@Composable
fun Modifier.keepOnScreen(margin: Dp = 22.dp): Modifier {
    val density = LocalDensity.current
    val screenW = androidx.compose.ui.platform.LocalWindowInfo.current.containerSize.width.toFloat()
    val m = with(density) { margin.toPx() }
    var shift by remember { mutableStateOf(0) }
    return this
        .onGloballyPositioned { c ->
            val x = c.positionInWindow().x; val w = c.size.width
            val s = when {
                w > screenW - 2 * m -> (m - x)
                x < m -> m - x
                x + w > screenW - m -> screenW - m - x - w
                else -> 0f
            }.roundToInt()
            if (s != shift) shift = s
        }
        .offset { androidx.compose.ui.unit.IntOffset(shift, 0) }
}

/** A reaction: a single emoji in a round badge, a line of text in a speech bubble. */
@Composable
fun ReactionBubble(text: String) {
    val emojiOnly = text.count { !it.isLowSurrogate() } <= 2
    if (emojiOnly) Surface(modifier = Modifier.keepOnScreen(), shape = CircleShape, color = Color(0xFFFFFDF6), shadowElevation = 6.dp, border = BorderStroke(1.dp, Gold)) {
        Text(text, fontSize = 24.sp, modifier = Modifier.padding(6.dp))
    } else Surface(modifier = Modifier.keepOnScreen(), shape = RoundedCornerShape(14.dp), color = Color(0xFFFFFDF6), shadowElevation = 6.dp, border = BorderStroke(1.dp, Gold)) {
        Text(text, color = Color(0xFF1C1A16), fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
            modifier = Modifier.widthIn(max = 170.dp).padding(horizontal = 10.dp, vertical = 5.dp))
    }
}

/**
 * How long a card takes from [a] to [b]: [base] ms over a phone table's
 * distances (up to ~180 dp), then growing with the square root of the
 * distance, up to [max] ms on a large window, so a card never races across
 * a big table.
 */
fun glideMs(base: Int, a: DpOffset, b: DpOffset, max: Int): Int {
    val d = kotlin.math.hypot((a.x - b.x).value, (a.y - b.y).value)
    return (base * kotlin.math.sqrt(maxOf(1f, d / 180f))).roundToInt().coerceAtMost(max)
}

/** A one-line button label: longer words get a smaller size instead of wrapping. */
@Composable
fun ButtonLabel(text: String, bold: Boolean = false) = Text(
    text, maxLines = 1, softWrap = false, fontWeight = if (bold) FontWeight.Bold else FontWeight.Medium,
    fontSize = when { text.length <= 7 -> 15.sp; text.length <= 10 -> 13.sp; else -> 12.sp },
)

@Composable
fun Chip(text: String, color: Color) = Surface(shape = RoundedCornerShape(10.dp), color = Color.Black.copy(alpha = .3f),
    border = BorderStroke(1.dp, color.copy(alpha = .6f))) {
    Text(text, color = color, fontSize = 12.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp))
}

@Composable
fun Overlay(title: String, sub: String, content: @Composable ColumnScope.() -> Unit) = Box(Modifier.fillMaxSize().background(Color(0xB0000000)), contentAlignment = Alignment.Center) {
    Surface(modifier = Modifier.padding(18.dp).widthIn(max = 380.dp), shape = RoundedCornerShape(18.dp), color = Panel.copy(alpha = .97f),
        border = BorderStroke(1.5.dp, Gold.copy(alpha = .8f)), shadowElevation = 14.dp) {
        Column(Modifier.padding(horizontal = 18.dp, vertical = 18.dp), horizontalAlignment = Alignment.CenterHorizontally, content = {
            Text(title, color = GoldBright, fontWeight = FontWeight.ExtraBold, fontSize = 20.sp, letterSpacing = 2.sp, textAlign = TextAlign.Center)
            Spacer(Modifier.height(6.dp)); Text(sub, color = Ink, textAlign = TextAlign.Center, fontSize = 12.sp)
            Spacer(Modifier.height(14.dp)); content()
        })
    }
}

