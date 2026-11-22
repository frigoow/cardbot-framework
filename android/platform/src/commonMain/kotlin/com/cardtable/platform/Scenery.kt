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

/**
 * The café theme's table props, drawn (no stock art): the player's own pint
 * and a cup of tea float above the desert, their shadows on the sand; a music
 * box plays on the sand below the bare tree. Each change of [cheers] (a
 * trick made of a 7, 8, 9 and 10) raises the pint for a toast.
 */
@Composable
fun androidx.compose.foundation.layout.BoxScope.TableDecor(cheers: Int) {
    val anim = rememberInfiniteTransition(label = "decor")
    val time by anim.animateFloat(0f, 1f, infiniteRepeatable(tween(4000, easing = LinearEasing)), label = "time")
    // the toast: lift, clink, set down. The count seen at first composition
    // does not play, so coming back to the table does not toast again.
    val schol = remember { Animatable(0f) }
    var toasted by remember { mutableStateOf(cheers) }
    LaunchedEffect(cheers) {
        if (cheers == toasted) return@LaunchedEffect
        toasted = cheers
        schol.snapTo(0f); schol.animateTo(1f, tween(1300, easing = LinearEasing)); schol.snapTo(0f)
    }
    val p = schol.value
    fun ease(x: Float) = x.coerceIn(0f, 1f).let { it * it * (3 - 2 * it) }
    val up = when { p <= 0f -> 0f; p < .25f -> ease(p / .25f); p < .7f -> 1f; else -> ease((1f - p) / .3f) }
    // the clink: a quick forward jolt once the glass is up
    val clink = if (p in .25f..0.7f) kotlin.math.sin((p - .25f) / .45f * 2 * kotlin.math.PI.toFloat() * 1.5f) * (1f - (p - .25f) / .45f) else 0f
    // the player's pint, at their own corner of the table, on a coaster
    Box(Modifier.align(Alignment.BottomStart).offset(x = 8.dp, y = (-6).dp).size(58.dp, 72.dp)) {
        Canvas(Modifier.matchParentSize()) {
            drawOval(Color(0x55180C04).copy(alpha = .33f * (1f - .7f * up)), Offset(size.width * .06f, size.height * .80f), androidx.compose.ui.geometry.Size(size.width * .92f, size.height * .2f))
            drawOval(Brush.radialGradient(listOf(Color(0xFFD9C3A0), Color(0xFFB89A70))), Offset(size.width * .04f, size.height * .76f),
                androidx.compose.ui.geometry.Size(size.width * .92f, size.height * .2f))
        }
        Canvas(Modifier.align(Alignment.TopCenter).size(38.dp, 62.dp).graphicsLayer {
            // raised toward the table centre, pivoting on the glass's foot
            transformOrigin = TransformOrigin(.5f, 1f)
            translationY = -22.dp.toPx() * up
            translationX = 6.dp.toPx() * up + 3.dp.toPx() * clink
            rotationZ = 14f * up + 6f * clink
            scaleX = 1f + .14f * up; scaleY = scaleX
        }) { drawBeer(time, splash = if (p in .25f..0.75f) (p - .25f) / .5f else 0f) }
        if (p > 0f) {
            val a = ease((p - .22f) / .1f) * ease((.95f - p) / .15f)
            Text(tl("Cheers!", "Schol!", "Наздраве!"), color = Gold, fontSize = 15.sp, fontWeight = FontWeight.Black,
                modifier = Modifier.align(Alignment.TopStart).offset(x = 38.dp, y = (-58).dp)
                    .wrapContentSize(unbounded = true)
                    .graphicsLayer { alpha = a; scaleX = .7f + .3f * a; scaleY = scaleX; rotationZ = -8f })
        }
    }
    Box(Modifier.align(Alignment.TopEnd).offset(x = (-8).dp, y = 14.dp).size(64.dp, 56.dp)) {
        Canvas(Modifier.matchParentSize()) {
            drawOval(Color(0x55180C04), Offset(size.width * .02f, size.height * .82f), androidx.compose.ui.geometry.Size(size.width * .98f, size.height * .18f))
        }
        Canvas(Modifier.matchParentSize()) { drawTea(time) }
    }
    Canvas(Modifier.align(Alignment.TopStart).offset(x = 10.dp, y = 8.dp).size(60.dp, 66.dp)) { drawMusicBox(time) }
    Canvas(Modifier.align(Alignment.BottomEnd).offset(x = (-8).dp, y = (-6).dp).size(104.dp, 66.dp)) { drawClockOnBook() }
    // the elephant crosses slowly below the tea
    val walk by anim.animateFloat(0f, 1f, infiniteRepeatable(tween(26000, easing = LinearEasing), RepeatMode.Reverse), label = "walk")
    Canvas(Modifier.align(Alignment.TopEnd).offset(x = (-14 - 30 * walk).dp, y = 78.dp).size(42.dp, 58.dp)
        .graphicsLayer { scaleX = if (walk > .5f) 1f else -1f }) { drawStiltElephant(time * 4f % 1f) }
}

/** [splash] runs 0..1 over a toast: foam drops jump off the rim (0 = none). */
fun androidx.compose.ui.graphics.drawscope.DrawScope.drawBeer(time: Float, splash: Float = 0f) {
    val w = size.width; val h = size.height
    val topY = h * .04f; val botY = h * .97f; val inset = w * .11f
    fun xl(y: Float) = inset * (y - topY) / (botY - topY)
    val glass = androidx.compose.ui.graphics.Path().apply {
        moveTo(0f, topY); lineTo(w, topY); lineTo(w - inset, botY); lineTo(inset, botY); close()
    }
    // beer body
    val beerTop = h * .25f
    val beer = androidx.compose.ui.graphics.Path().apply {
        moveTo(xl(beerTop) + 1.5f, beerTop); lineTo(w - xl(beerTop) - 1.5f, beerTop); lineTo(w - inset - 1.5f, botY - h * .07f); lineTo(inset + 1.5f, botY - h * .07f); close()
    }
    drawPath(beer, Brush.verticalGradient(listOf(Color(0xFFF6C35A), Color(0xFFE09A22), Color(0xFFB06A10)), startY = beerTop, endY = botY))
    drawPath(beer, Brush.horizontalGradient(listOf(Color(0x66401E00), Color.Transparent, Color(0x33FFF3C0), Color.Transparent, Color(0x77401E00))))
    // rising bubbles
    for (k in 0 until 9) {
        val bx = w * (.25f + .5f * ((k * 37) % 9) / 9f)
        val phase = (time + k / 9f) % 1f
        val by = botY - h * .1f - phase * (botY - beerTop - h * .12f)
        drawCircle(Color.White.copy(alpha = .55f * (1 - phase) + .15f), 0.9f + (k % 3) * .4f, Offset(bx, by))
    }
    // foam head: a band with a bubbly top edge
    val foamTop = h * .10f
    drawRect(Brush.verticalGradient(listOf(Color(0xFFFFFCF2), Color(0xFFF1E3C4)), startY = foamTop, endY = beerTop + 2f),
        Offset(xl(foamTop) + 1f, foamTop), androidx.compose.ui.geometry.Size(w - 2 * xl(foamTop) - 2f, beerTop - foamTop + 2f))
    for (k in 0..5) drawCircle(Color(0xFFFFFDF6), w * .1f, Offset(w * (.1f + .16f * k), foamTop + h * .005f))
    drawCircle(Color(0xFFFFFDF6), w * .12f, Offset(w * .78f, foamTop - h * .035f))
    // glass: thick base, rim, highlights, condensation
    drawRect(Color(0x55DDEEFF), Offset(inset, botY - h * .07f), androidx.compose.ui.geometry.Size(w - 2 * inset, h * .07f))
    drawPath(glass, Color.White.copy(alpha = .55f), style = androidx.compose.ui.graphics.drawscope.Stroke(1.4f))
    drawLine(Color.White.copy(alpha = .55f), Offset(w * .2f, h * .3f), Offset(w * .24f, h * .86f), strokeWidth = w * .06f)
    drawLine(Color.White.copy(alpha = .25f), Offset(w * .78f, h * .32f), Offset(w * .74f, h * .8f), strokeWidth = w * .03f)
    for (k in 0 until 7) drawCircle(Color.White.copy(alpha = .45f), 1.1f, Offset(w * (.3f + .07f * ((k * 5) % 7)), h * (.4f + .07f * k)))
    if (splash > 0f) for (k in 0 until 6) {
        // each drop flies up and out, then falls back, fading
        val vx = (k - 2.5f) * .09f + .12f
        val x = w * (.5f + vx * splash * 2.2f)
        val y = topY - h * (.42f * splash - .5f * splash * splash) * (1f + .25f * (k % 3))
        drawCircle(Color(0xFFFFFDF6).copy(alpha = (1f - splash) * .95f), w * (.045f + .015f * (k % 2)), Offset(x, y))
    }
}

fun androidx.compose.ui.graphics.drawscope.DrawScope.drawTea(time: Float) {
    val w = size.width; val h = size.height
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    // saucer
    drawOval(Brush.verticalGradient(listOf(Color(0xFFFFFFFF), Color(0xFFD8D4CC)), startY = h * .78f, endY = h), Offset(0f, h * .78f), sz(w, h * .2f))
    drawOval(Color(0xFFC9A54E), Offset(0f, h * .78f), sz(w, h * .2f), style = androidx.compose.ui.graphics.drawscope.Stroke(1.2f))
    drawOval(Color(0x22000000), Offset(w * .25f, h * .82f), sz(w * .5f, h * .1f))
    // cup body
    val l = w * .14f; val r = w * .78f; val top = h * .36f; val foot = h * .86f
    val body = androidx.compose.ui.graphics.Path().apply {
        moveTo(l, top)
        cubicTo(l, h * .72f, w * .3f, foot, (l + r) / 2, foot)
        cubicTo(w * .62f, foot, r, h * .72f, r, top)
        close()
    }
    drawPath(body, Brush.horizontalGradient(listOf(Color(0xFFCFCBC2), Color(0xFFFFFFFF), Color(0xFFF4F1EA), Color(0xFFBDB8AE)), startX = l, endX = r))
    drawLine(Color(0xFFC9A54E), Offset(l + w * .03f, h * .5f), Offset(r - w * .03f, h * .5f), strokeWidth = 1.2f)
    // handle
    drawArc(Color(0xFFE6E2DA), -70f, 160f, false, Offset(r - w * .1f, h * .42f), sz(w * .22f, h * .26f),
        style = androidx.compose.ui.graphics.drawscope.Stroke(w * .05f))
    // tea surface and gilded rim
    drawOval(Brush.radialGradient(listOf(Color(0xFFC07A30), Color(0xFF8A4A14)), center = Offset((l + r) / 2, top), radius = (r - l) / 2),
        Offset(l + 1.5f, top - h * .06f), sz(r - l - 3f, h * .12f))
    drawOval(Color(0xFFD4AF5A), Offset(l, top - h * .065f), sz(r - l, h * .13f), style = androidx.compose.ui.graphics.drawscope.Stroke(1.6f))
    drawOval(Color.White.copy(alpha = .35f), Offset(l + w * .1f, top - h * .03f), sz(w * .16f, h * .03f))
    // steam
    for (k in 0..2) {
        val phase = (time + k / 3f) % 1f
        val x0 = w * (.32f + .14f * k); val y0 = top - h * .08f
        val path = androidx.compose.ui.graphics.Path()
        for (i in 0..12) {
            val f = i / 12f
            val x = x0 + w * .05f * kotlin.math.sin((f * 6 + phase * 6.28f + k).toDouble()).toFloat()
            val y = y0 - f * h * .34f - phase * h * .05f
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, Color.White.copy(alpha = .35f * (1 - phase) + .05f), style = androidx.compose.ui.graphics.drawscope.Stroke(2.2f, cap = androidx.compose.ui.graphics.StrokeCap.Round))
    }
}

fun androidx.compose.ui.graphics.drawscope.DrawScope.drawMusicBox(time: Float) {
    val w = size.width; val h = size.height
    val sz = { a: Float, b: Float -> androidx.compose.ui.geometry.Size(a, b) }
    val bx = w * .08f; val by = h * .55f; val bw = w * .78f; val bh = h * .34f
    // shadow on the sand
    drawOval(Color(0x442A1E14), Offset(bx - w * .02f, by + bh - h * .03f), sz(bw * 1.2f, h * .1f))
    // open lid, tilted back, with its mirror
    val lid = androidx.compose.ui.graphics.Path().apply {
        moveTo(bx, by); lineTo(bx + w * .06f, by - h * .38f); lineTo(bx + bw + w * .06f, by - h * .38f); lineTo(bx + bw, by); close()
    }
    drawPath(lid, Brush.verticalGradient(listOf(Color(0xFF6E431F), Color(0xFF4A2A12)), startY = by - h * .38f, endY = by))
    val mirror = androidx.compose.ui.graphics.Path().apply {
        moveTo(bx + w * .06f, by - h * .03f); lineTo(bx + w * .1f, by - h * .33f); lineTo(bx + bw - w * .0f, by - h * .33f); lineTo(bx + bw - w * .05f, by - h * .03f); close()
    }
    drawPath(mirror, Brush.linearGradient(listOf(Color(0xFFDCE6F0), Color(0xFF8FA3BA), Color(0xFFE8EEF4))))
    // the box
    drawRoundRect(Brush.verticalGradient(listOf(Color(0xFF9A6434), Color(0xFF5E3A1A)), startY = by, endY = by + bh),
        Offset(bx, by), sz(bw, bh), cornerRadius = androidx.compose.ui.geometry.CornerRadius(3f, 3f))
    drawRect(Color(0xFFD4AF5A), Offset(bx, by + bh * .12f), sz(bw, 1.6f))
    drawRect(Color(0xFFD4AF5A), Offset(bx, by + bh * .82f), sz(bw, 1.6f))
    drawCircle(Color(0xFF2A1A0C), 1.8f, Offset(bx + bw / 2, by + bh * .45f))
    drawLine(Color(0xFF2A1A0C), Offset(bx + bw / 2, by + bh * .45f), Offset(bx + bw / 2, by + bh * .6f), strokeWidth = 1.4f)
    // winding key
    drawLine(Color(0xFFD4AF5A), Offset(bx + bw, by + bh * .5f), Offset(bx + bw + w * .08f, by + bh * .5f), strokeWidth = 2f)
    drawOval(Color(0xFFD4AF5A), Offset(bx + bw + w * .06f, by + bh * .3f), sz(w * .06f, bh * .4f))
    // the dancer, turning (a squeezed width fakes the spin)
    val cx = bx + bw / 2; val spin = kotlin.math.cos(time * 6.283f * 2).toFloat()
    val sw = (.35f + .65f * kotlin.math.abs(spin))
    drawLine(Color(0xFFF2E6D6), Offset(cx, by - h * .02f), Offset(cx, by - h * .14f), strokeWidth = 1.6f)
    val tutu = androidx.compose.ui.graphics.Path().apply {
        moveTo(cx - w * .12f * sw, by - h * .14f); lineTo(cx + w * .12f * sw, by - h * .14f); lineTo(cx, by - h * .22f); close()
    }
    drawPath(tutu, Color(0xFFF4C6CF))
    drawLine(Color(0xFFF2E6D6), Offset(cx, by - h * .2f), Offset(cx, by - h * .28f), strokeWidth = 1.6f)
    drawLine(Color(0xFFF2E6D6), Offset(cx - w * .07f * sw, by - h * .31f), Offset(cx + w * .07f * sw, by - h * .31f), strokeWidth = 1.2f)
    drawCircle(Color(0xFFF2E6D6), w * .035f, Offset(cx, by - h * .31f))
    // a few notes drifting up
    for (k in 0..1) {
        val phase = (time + k * .5f) % 1f
        val nx = bx + bw * (.75f + .2f * k) + w * .08f * kotlin.math.sin(phase * 6.283f).toFloat()
        val ny = by - h * .1f - phase * h * .45f
        val a = (1 - phase) * .9f
        drawCircle(Color(0xFF2A1A0C).copy(alpha = a), 2.4f, Offset(nx, ny))
        drawLine(Color(0xFF2A1A0C).copy(alpha = a), Offset(nx + 2.2f, ny), Offset(nx + 2.2f, ny - h * .1f), strokeWidth = 1.2f)
    }
}

