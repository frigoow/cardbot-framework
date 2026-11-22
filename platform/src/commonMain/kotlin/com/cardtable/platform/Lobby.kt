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
fun ThemePicker(selectedId: String, choose: (String) -> Unit) = Column(Modifier.widthIn(max = 480.dp).fillMaxWidth()) {
    Text(tl("Table theme", "Tafelthema", "Тема на масата"), color = Ink, fontSize = 13.sp)
    Spacer(Modifier.height(6.dp))
    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        THEMES.forEach { t ->
            val selected = t.id == selectedId
            Column(horizontalAlignment = Alignment.CenterHorizontally, modifier = Modifier.clickable { choose(t.id) }) {
                Box(
                    Modifier.size(70.dp, 52.dp).clip(RoundedCornerShape(10.dp))
                        .background(Brush.radialGradient(listOf(t.feltCenter, t.feltMid, t.feltEdge)))
                        .border(if (selected) 3.dp else 1.dp, if (selected) t.accentBright else t.rim, RoundedCornerShape(10.dp)),
                    contentAlignment = Alignment.Center,
                ) {
                    Row(horizontalArrangement = Arrangement.spacedBy((-8).dp)) {
                        Box(Modifier.size(20.dp, 29.dp).clip(RoundedCornerShape(3.dp))
                            .background(Brush.linearGradient(listOf(t.backA, t.backB))).border(1.dp, t.accent, RoundedCornerShape(3.dp)),
                            contentAlignment = Alignment.Center) { Text(t.emblem, color = t.accentBright, fontSize = 9.sp) }
                        Box(Modifier.size(20.dp, 29.dp).clip(RoundedCornerShape(3.dp))
                            .background(Brush.verticalGradient(listOf(t.paperTop, t.paperBottom))).border(1.dp, t.accent.copy(alpha = .6f), RoundedCornerShape(3.dp)),
                            contentAlignment = Alignment.Center) { Text("A", color = CardRed, fontSize = 11.sp, fontWeight = FontWeight.Bold) }
                    }
                }
                Text(t.label(), color = if (selected) GoldBright else InkDim, fontSize = 11.sp, fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal)
            }
        }
    }
}

@Composable
fun ProfileDialog(
    profiles: List<Profile>, activeId: String, close: () -> Unit,
    select: (String) -> Unit, create: (String) -> Unit, rename: (String, String) -> Unit, delete: (String) -> Unit,
) {
    var newName by remember { mutableStateOf("") }
    var renaming by remember { mutableStateOf<Profile?>(null) }
    var renameText by remember { mutableStateOf("") }
    var deleting by remember { mutableStateOf<Profile?>(null) }
    AlertDialog(
        onDismissRequest = close,
        confirmButton = { TextButton(onClick = close) { Text(tl("Done", "Klaar", "Готово")) } },
        title = { Text(tl("Profiles", "Profielen", "Профили")) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                profiles.forEach { p ->
                    Row(Modifier.fillMaxWidth().clickable { select(p.id) }.padding(vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically) {
                        Text(if (p.id == activeId) "●" else "○", fontSize = 18.sp)
                        Spacer(Modifier.width(10.dp))
                        Text(p.name, Modifier.weight(1f), fontWeight = if (p.id == activeId) FontWeight.Bold else FontWeight.Normal)
                        TextButton(onClick = { renaming = p; renameText = p.name }) { Text(tl("Rename", "Hernoemen", "Преименувай")) }
                        if (profiles.size > 1) TextButton(onClick = { deleting = p }) { Text(tl("Delete", "Verwijderen", "Изтрий")) }
                    }
                }
                Spacer(Modifier.height(10.dp))
                OutlinedTextField(value = newName, onValueChange = { newName = it.take(24) }, singleLine = true,
                    label = { Text(tl("New profile name", "Naam nieuw profiel", "Име на новия профил")) }, modifier = Modifier.fillMaxWidth())
                TextButton(onClick = { create(newName); newName = "" }) { Text(tl("Create and switch to it", "Aanmaken en wisselen", "Създай и превключи")) }
                Text(tl("Each profile has its own saved game and statistics.", "Elk profiel heeft een eigen bewaard spel en statistieken.", "Всеки профил има своя записана игра и статистика."), fontSize = 12.sp)
            }
        },
    )
    renaming?.let { p ->
        AlertDialog(
            onDismissRequest = { renaming = null },
            confirmButton = { TextButton(onClick = { rename(p.id, renameText); renaming = null }) { Text(tl("Rename", "Hernoemen", "Преименувай")) } },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text(tl("Cancel", "Annuleren", "Отказ")) } },
            title = { Text(tl("Rename profile", "Profiel hernoemen", "Преименувай профила")) },
            text = { OutlinedTextField(value = renameText, onValueChange = { renameText = it.take(24) }, singleLine = true) },
        )
    }
    deleting?.let { p ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            confirmButton = { TextButton(onClick = { delete(p.id); deleting = null }) { Text(tl("Delete", "Verwijderen", "Изтрий")) } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text(tl("Cancel", "Annuleren", "Отказ")) } },
            title = { Text(tl("Delete ${p.name}?", "${p.name} verwijderen?", "Да се изтрие ${p.name}?")) },
            text = { Text(tl("Its saved game and all its statistics are removed. This cannot be undone.", "Het bewaarde spel en alle statistieken worden verwijderd. Dit kan niet ongedaan gemaakt worden.", "Записаната игра и цялата статистика се изтриват. Това не може да се отмени.")) },
        )
    }
}


/**
 * The screen behind the table: the theme's sky (café: a dream desert at
 * night; others: stars or a cloth weave), its glow and a vignette.
 */
@Composable
fun Modifier.tableBackdrop(): Modifier {
    val theme = LocalAppTheme.current
    val weave = Gold.copy(alpha = .035f)
    val vignette = Color.Black.copy(alpha = .55f)
    val stars = remember { val r = kotlin.random.Random(7); List(150) { floatArrayOf(r.nextFloat(), r.nextFloat(), r.nextFloat()) } }
    val felt = Brush.radialGradient(listOf(Felt2, Felt1, Felt0), radius = 1900f)
    return background(theme.bgTop?.let { top -> Brush.verticalGradient(listOf(top, theme.bgBottom ?: top)) } ?: felt)
        .drawBehind {
            if (theme.style == "cafe") {
                surrealBackdrop()
                // stars only in the upper sky
                stars.forEach { (fx, fy, b) ->
                    if (fy < .5f) drawCircle(Color.White.copy(alpha = (.10f + b * .4f) * (1 - fy * 1.6f).coerceAtLeast(0f)),
                        (.6f + b * .8f).dp.toPx(), Offset(fx * size.width, fy * size.height))
                }
            } else if (theme.style == "mehana") {
                mehanaBackdrop()
            } else if (theme.stars) {
                // a quiet night sky: small stars, a few brighter ones
                stars.forEach { (fx, fy, b) ->
                    val r = (if (b > .93f) 1.6f else .7f + b * .6f).dp.toPx()
                    drawCircle(Color.White.copy(alpha = .10f + b * (if (b > .93f) .55f else .25f)), r, Offset(fx * size.width, fy * size.height))
                }
            } else {
                // a faint diagonal weave, like cloth under the table
                val step = 26.dp.toPx()
                var x = -size.height
                while (x < size.width) {
                    drawLine(weave, Offset(x, 0f), Offset(x + size.height, size.height), strokeWidth = 1f)
                    drawLine(weave, Offset(x + size.height, 0f), Offset(x, size.height), strokeWidth = 1f)
                    x += step
                }
            }
            if (theme.glow != Color.Transparent) drawRect(Brush.radialGradient(listOf(theme.glow, Color.Transparent),
                center = Offset(size.width * .5f, 0f), radius = size.width * 1.1f))
            drawRect(Brush.radialGradient(listOf(Color.Transparent, Color.Transparent, vignette),
                center = center, radius = maxOf(size.width, size.height) * .75f))
        }
}

/** The table: felt in a wooden rim (café: grained, with a melting edge), the play area inside. */
@Composable
fun FeltTable(modifier: Modifier, content: @Composable androidx.compose.foundation.layout.BoxWithConstraintsScope.() -> Unit) {
    val shape = RoundedCornerShape(48.dp)
    val style = LocalAppTheme.current.style
    val melting = style == "cafe"
    val wooded = style == "cafe" || style == "mehana"
    val mehana = style == "mehana"
    val rim = Wood
    Box(modifier.padding(horizontal = 10.dp)
        .then(if (melting) Modifier.drawWithContent { drawContent(); meltingEdge(rim) } else Modifier)
        .shadow(20.dp, shape).clip(shape)
        .background(Brush.radialGradient(listOf(Felt2, Felt1, Felt0)))
        .then(if (wooded) Modifier.drawBehind { woodGrain(); if (mehana) shevitsaBorder() } else Modifier)
        .border(8.dp, Wood, shape).padding(8.dp)) {
        BoxWithConstraints(Modifier.fillMaxSize().border(1.dp, Gold.copy(alpha = if (mehana) .45f else .28f), RoundedCornerShape(38.dp)), content = content)
    }
}
